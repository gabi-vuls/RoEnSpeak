"""Resumable model downloader (Hugging Face Hub -> models_dir()/<model>, plain files).

Design notes
------------
* File list, sizes, SHA-256 and the commit hash come from ``HfApi.model_info(files_metadata=True)``,
  so the total byte count is known before the first byte is fetched.
* Files are streamed by us from the Hub ``/resolve/<commit>/<file>`` URL with huggingface_hub's own
  HTTP session and headers, into ``<file>.part`` with an HTTP Range header for resume. We do
  not use ``hf_hub_download``, because huggingface_hub 2.2.0 downloads into a process-unique temp file
  and deletes it on any failure, so a cancelled or interrupted download could never resume, and
  its xet backend cannot be cancelled mid-file. The Hub serves xet-backed files over plain
  HTTP (redirect to its CDN bridge), so hf-xet is never involved and byte progress is exact.
  We therefore do not need to set ``HF_HUB_DISABLE_XET``.
* LFS files (model.bin) are verified against their SHA-256 before being moved into place.
  ``MODEL_COMPLETE_MARKER`` is written last, after all required files are verified.
* The commit hash of a partial download is kept in ``.download_revision``. If the repo moves to a new
  commit, stale ``.part`` files are discarded instead of being mixed.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import shutil
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from app.contracts import (
    APP_NAME,
    APP_VERSION,
    MODEL_ALLOW_PATTERNS,
    MODEL_COMPLETE_MARKER,
    MODEL_REQUIRED_FILES,
    MODELS,
    DownloadCancelledError,
    DownloadError,
    DownloadProgress,
    ThreadingEvent,
    models_dir,
)

logger = logging.getLogger("roenspeak.engine.downloader")

PART_SUFFIX = ".part"
REVISION_FILE = ".download_revision"
CHUNK_SIZE = 1024 * 1024
PROGRESS_INTERVAL_S = 0.1  # <= 10 callbacks per second
MAX_STREAM_RETRIES = 5
CONNECT_TIMEOUT_S = 15.0
READ_TIMEOUT_S = 60.0


def model_dir_complete(path: Path) -> bool:
    """The completeness rule shared by the downloader and the engine: every
    MODEL_REQUIRED_FILES file and MODEL_COMPLETE_MARKER exist."""
    return all((path / f).is_file() for f in MODEL_REQUIRED_FILES) and (path / MODEL_COMPLETE_MARKER).is_file()


def _check_model(model: str) -> None:
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model!r}")


def _quiet_hf_hub() -> None:
    """Silence huggingface_hub's own progress bars / symlink warnings (must run before import)."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # httpx2 logs every request (including signed CDN URLs) at INFO; keep our log readable.
    for name in ("httpx2", "httpcore2"):
        lg = logging.getLogger(name)
        if lg.level == logging.NOTSET or lg.level < logging.WARNING:
            lg.setLevel(logging.WARNING)


@dataclass(frozen=True)
class _RemoteFile:
    name: str
    size: int
    sha256: Optional[str]  # LFS files only


_OFFLINE_HINT = (
    "Could not reach huggingface.co to download the speech model. "
    "Check your internet connection (and any proxy/firewall), then try again. "
    "The download resumes where it stopped."
)


def _is_connectivity_error(exc: BaseException) -> bool:
    """True if exc (or anything in its cause/context chain) is a network-level failure."""
    try:
        import httpx2

        net_types: tuple[type, ...] = (httpx2.TransportError,)
    except ImportError:  # pragma: no cover - httpx2 always ships with huggingface_hub 2.x
        net_types = ()
    try:
        from huggingface_hub.errors import OfflineModeIsEnabled

        net_types = net_types + (OfflineModeIsEnabled,)
    except ImportError:  # pragma: no cover
        pass
    net_types = net_types + (socket.gaierror, socket.timeout, ConnectionError, TimeoutError)
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, net_types):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


class _Progress:
    """Throttled progress reporting with a smoothed rate."""

    def __init__(self, model: str, total: int, done: int, cb: Callable[[DownloadProgress], None]) -> None:
        self.model = model
        self.total = total
        self.done = done
        self.cb = cb
        self.current_file = ""
        self.rate = 0.0
        self._last_t = time.monotonic()
        self._last_done = done

    def add(self, n: int) -> None:
        self.done += n
        self.maybe_emit()

    def set_done(self, done: int) -> None:
        self.done = done

    def maybe_emit(self, force: bool = False) -> None:
        now = time.monotonic()
        dt = now - self._last_t
        if not force and dt < PROGRESS_INTERVAL_S:
            return
        if dt > 0:
            inst = max(0, self.done - self._last_done) / dt
            self.rate = inst if self.rate == 0.0 else 0.7 * self.rate + 0.3 * inst
        self._last_t = now
        self._last_done = self.done
        try:
            self.cb(DownloadProgress(self.model, self.done, self.total, self.current_file, self.rate))
        except Exception:
            logger.exception("download progress callback failed")


class HfModelDownloader:
    """ModelDownloader implementation (see contracts.ModelDownloader)."""

    def __init__(self, models_root: Optional[Path] = None) -> None:
        self._models_root = Path(models_root) if models_root is not None else None

    @property
    def models_root(self) -> Path:
        return self._models_root if self._models_root is not None else models_dir()

    def model_path(self, model: str) -> Path:
        _check_model(model)
        return self.models_root / model

    def is_downloaded(self, model: str) -> bool:
        return model_dir_complete(self.model_path(model))

    # -- download -----------------------------------------------------------------

    def download(
        self,
        model: str,
        on_progress: Callable[[DownloadProgress], None],
        cancel: ThreadingEvent,
    ) -> Path:
        _check_model(model)
        spec = MODELS[model]
        dest = self.model_path(model)
        if self.is_downloaded(model):
            logger.info("model %s already downloaded at %s", model, dest)
            return dest
        if cancel.is_set():
            raise DownloadCancelledError("Download cancelled.")

        t0 = time.monotonic()
        files, revision = self._list_files(spec.hf_repo)
        total = sum(f.size for f in files)
        logger.info(
            "download start model=%s repo=%s revision=%s files=%d total_bytes=%d dest=%s",
            model, spec.hf_repo, revision, len(files), total, dest,
        )

        try:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / MODEL_COMPLETE_MARKER).unlink(missing_ok=True)
            self._reset_if_revision_changed(dest, revision, files)
        except OSError as e:
            raise DownloadError(f"Could not create the model folder {dest}: {e}", details=repr(e)) from e

        already = sum(self._present_bytes(dest, f) for f in files)
        self._check_disk_space(dest, total - already)

        progress = _Progress(model, total, already, on_progress)
        progress.current_file = files[0].name if files else ""
        progress.maybe_emit(force=True)

        downloaded_now = 0
        for f in files:
            progress.current_file = f.name
            before = progress.done
            self._download_file(spec.hf_repo, revision, dest, f, progress, cancel)
            downloaded_now += progress.done - before

        self._verify_and_mark(dest, files, spec.hf_repo, revision)
        progress.current_file = ""
        progress.set_done(total)
        progress.maybe_emit(force=True)
        dt = time.monotonic() - t0
        logger.info(
            "download done model=%s bytes=%d fetched_bytes=%d seconds=%.1f",
            model, total, downloaded_now, dt,
        )
        return dest

    # -- helpers ------------------------------------------------------------------

    def _list_files(self, repo: str) -> tuple[list[_RemoteFile], str]:
        _quiet_hf_hub()
        try:
            from huggingface_hub import HfApi
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()
            info = HfApi(library_name=APP_NAME, library_version=APP_VERSION).model_info(
                repo, files_metadata=True, timeout=CONNECT_TIMEOUT_S, token=False
            )
        except Exception as e:
            raise self._wrap_error(e, f"Could not get the file list of {repo}") from e

        files: list[_RemoteFile] = []
        for s in info.siblings or []:
            name = s.rfilename
            if "/" in name or not any(fnmatch.fnmatchcase(name, p) for p in MODEL_ALLOW_PATTERNS):
                continue
            size = s.size if s.size is not None else (s.lfs.size if s.lfs is not None else None)
            if size is None:
                raise DownloadError(
                    f"The model repository did not report the size of {name}.",
                    details=f"repo={repo}",
                )
            sha = s.lfs.sha256 if s.lfs is not None else None
            files.append(_RemoteFile(name, int(size), sha))
        names = {f.name for f in files}
        missing = [r for r in MODEL_REQUIRED_FILES if r not in names]
        if missing:
            raise DownloadError(
                f"The model repository {repo} is missing required files: {', '.join(missing)}.",
                details=f"files={sorted(names)}",
            )
        if not info.sha:
            raise DownloadError(f"The model repository {repo} did not report a revision.")
        # Small files first so the big model.bin dominates the tail of the progress bar.
        files.sort(key=lambda f: (f.size, f.name))
        return files, info.sha

    @staticmethod
    def _present_bytes(dest: Path, f: _RemoteFile) -> int:
        final = dest / f.name
        if final.is_file() and final.stat().st_size == f.size:
            return f.size
        part = dest / (f.name + PART_SUFFIX)
        if part.is_file():
            return min(part.stat().st_size, f.size)
        return 0

    @staticmethod
    def _reset_if_revision_changed(dest: Path, revision: str, files: list[_RemoteFile]) -> None:
        rev_file = dest / REVISION_FILE
        old = rev_file.read_text(encoding="utf-8").strip() if rev_file.is_file() else None
        if old is not None and old != revision:
            logger.info("model repo moved from %s to %s; discarding partial/old files", old, revision)
            for f in files:
                (dest / (f.name + PART_SUFFIX)).unlink(missing_ok=True)
                (dest / f.name).unlink(missing_ok=True)
        elif old is None:
            # Unknown provenance: keep complete files (they are size/hash checked) but drop parts.
            for f in files:
                (dest / (f.name + PART_SUFFIX)).unlink(missing_ok=True)
        rev_file.write_text(revision, encoding="utf-8")

    @staticmethod
    def _check_disk_space(dest: Path, needed: int) -> None:
        try:
            free = shutil.disk_usage(dest).free
        except OSError:
            return
        margin = 50 * 1024 * 1024
        if needed > 0 and free < needed + margin:
            raise DownloadError(
                f"Not enough disk space in {dest}: {needed / 1e9:.2f} GB needed, {free / 1e9:.2f} GB free.",
                details=f"needed={needed} free={free}",
            )

    def _download_file(
        self,
        repo: str,
        revision: str,
        dest: Path,
        f: _RemoteFile,
        progress: _Progress,
        cancel: ThreadingEvent,
    ) -> None:
        final = dest / f.name
        part = dest / (f.name + PART_SUFFIX)
        if final.is_file() and final.stat().st_size == f.size:
            if f.sha256 is None or self._sha256_ok(final, f.sha256, cancel):
                return
            logger.warning("%s failed its checksum; downloading it again", f.name)
            progress.set_done(progress.done - f.size)
            final.unlink()

        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers, get_session
        import httpx2

        url = hf_hub_url(repo, f.name, revision=revision)
        base_headers = build_hf_headers(
            token=False, library_name=APP_NAME, library_version=APP_VERSION
        )
        base_headers["Accept-Encoding"] = "identity"  # exact bytes, so Range/size math holds
        timeout = httpx2.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)
        session = get_session()

        retries = 0
        while True:
            if cancel.is_set():
                raise DownloadCancelledError("Download cancelled.")
            have = part.stat().st_size if part.is_file() else 0
            if have > f.size:
                part.unlink()
                progress.set_done(progress.done - f.size)
                have = 0
            if have == f.size:
                break
            headers = dict(base_headers)
            if have:
                headers["Range"] = f"bytes={have}-"
            try:
                with session.stream("GET", url, headers=headers, timeout=timeout, follow_redirects=True) as resp:
                    if resp.status_code == 416 and have:
                        # Range not satisfiable: our .part is unusable; start over.
                        part.unlink(missing_ok=True)
                        progress.set_done(progress.done - have)
                        continue
                    if resp.status_code not in (200, 206):
                        resp.read()
                        raise DownloadError(
                            f"Download of {f.name} failed: the server answered HTTP {resp.status_code}.",
                            details=f"url={url} body={resp.text[:300]!r}",
                        )
                    mode = "ab"
                    if have and resp.status_code == 200:
                        logger.info("server ignored Range for %s; restarting the file", f.name)
                        progress.set_done(progress.done - have)
                        mode = "wb"
                    with open(part, mode) as out:
                        for chunk in resp.iter_bytes(CHUNK_SIZE):
                            if cancel.is_set():
                                raise DownloadCancelledError("Download cancelled.")
                            if chunk:
                                out.write(chunk)
                                progress.add(len(chunk))
                                retries = 0
            except (DownloadCancelledError, DownloadError):
                raise
            except OSError as e:
                if _is_connectivity_error(e):
                    retries = self._retry_or_raise(retries, e, f.name, cancel)
                    continue
                raise DownloadError(f"Could not write {part}: {e}", details=repr(e)) from e
            except Exception as e:
                if _is_connectivity_error(e):
                    retries = self._retry_or_raise(retries, e, f.name, cancel)
                    continue
                raise self._wrap_error(e, f"Download of {f.name} failed") from e
            # Stream ended: loop re-checks the size (short reads resume on the next pass).
            got = part.stat().st_size if part.is_file() else 0
            if got < f.size and got == have:
                retries = self._retry_or_raise(
                    retries, ConnectionError("connection closed without data"), f.name, cancel
                )

        if f.sha256 is not None and not self._sha256_ok(part, f.sha256, cancel):
            part.unlink(missing_ok=True)
            progress.set_done(progress.done - f.size)
            raise DownloadError(
                f"{f.name} was corrupted during download and has been discarded. Please try again.",
                details=f"sha256 mismatch for {f.name}",
            )
        try:
            os.replace(part, final)
        except OSError as e:
            raise DownloadError(f"Could not move {part} into place: {e}", details=repr(e)) from e

    @staticmethod
    def _retry_or_raise(retries: int, exc: BaseException, name: str, cancel: ThreadingEvent) -> int:
        retries += 1
        if retries > MAX_STREAM_RETRIES:
            raise DownloadError(
                _OFFLINE_HINT, offline=True, details=f"{name}: {type(exc).__name__}: {exc}"
            ) from exc
        delay = min(2 ** (retries - 1), 8)
        logger.warning("network error while downloading %s (%s); retry %d/%d in %ds",
                       name, exc, retries, MAX_STREAM_RETRIES, delay)
        end = time.monotonic() + delay
        while time.monotonic() < end:
            if cancel.is_set():
                raise DownloadCancelledError("Download cancelled.")
            time.sleep(0.05)
        return retries

    @staticmethod
    def _sha256_ok(path: Path, expected: str, cancel: ThreadingEvent) -> bool:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                if cancel.is_set():
                    raise DownloadCancelledError("Download cancelled.")
                buf = fh.read(8 * 1024 * 1024)
                if not buf:
                    break
                h.update(buf)
        return h.hexdigest() == expected.lower()

    @staticmethod
    def _verify_and_mark(dest: Path, files: list[_RemoteFile], repo: str, revision: str) -> None:
        for f in files:
            p = dest / f.name
            if not p.is_file() or p.stat().st_size != f.size:
                raise DownloadError(
                    f"{f.name} is incomplete after download. Please try again.",
                    details=f"path={p}",
                )
        for r in MODEL_REQUIRED_FILES:
            if not (dest / r).is_file():
                raise DownloadError(f"Required model file {r} is missing after download.")
        marker = {
            "repo": repo,
            "revision": revision,
            "files": {f.name: f.size for f in files},
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        try:
            tmp = dest / (MODEL_COMPLETE_MARKER + ".tmp")
            tmp.write_text(json.dumps(marker, indent=2), encoding="utf-8")
            os.replace(tmp, dest / MODEL_COMPLETE_MARKER)
        except OSError as e:
            raise DownloadError(f"Could not finalize the model folder {dest}: {e}", details=repr(e)) from e

    @staticmethod
    def _wrap_error(exc: BaseException, what: str) -> DownloadError:
        if _is_connectivity_error(exc):
            return DownloadError(_OFFLINE_HINT, offline=True, details=f"{what}: {type(exc).__name__}: {exc}")
        if isinstance(exc, OSError):
            return DownloadError(f"{what}: {exc}", details=repr(exc))
        try:
            from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

            if isinstance(exc, RepositoryNotFoundError):
                return DownloadError(f"{what}: the model repository was not found.", details=str(exc))
            if isinstance(exc, HfHubHTTPError):
                code = exc.response.status_code if exc.response is not None else "?"
                return DownloadError(f"{what}: the server answered HTTP {code}.", details=str(exc))
        except ImportError:  # pragma: no cover
            pass
        return DownloadError(f"{what}: {exc}", details=f"{type(exc).__name__}: {exc}")
