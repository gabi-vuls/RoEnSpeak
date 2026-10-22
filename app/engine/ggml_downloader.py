"""Resumable ggml model downloader (macOS): one file from GGML_REPO per model + compute type.

The Windows engine loads a *directory* of CTranslate2 files; whisper.cpp loads a *single*
``ggml-*.bin``. So this is HfModelDownloader with a one-file manifest:
``models_root/<model>/<ggml_model_file(model, compute_type)>`` plus MODEL_COMPLETE_MARKER.

It subclasses HfModelDownloader deliberately rather than copying it. Everything that is
genuinely hard and already proven on Windows -- the ``Range:`` resume loop, the 416 and
"server ignored Range" recoveries, the retry/backoff with cancellation, the SHA-256 check
before the final ``os.replace``, the throttled EMA progress, the connectivity detection and
the DownloadError/DownloadCancelledError contract -- is inherited unchanged, so the two
platforms cannot drift. Only the manifest, the completeness rule and the marker differ:

* ``_list_files`` (a repo-wide file list keyed off MODEL_ALLOW_PATTERNS) is replaced by one
  ``HfApi.get_paths_info`` lookup for the one file we want, which also gives its size and
  its LFS sha256 without downloading anything.
* ``is_downloaded`` asks about the **currently selected** compute type, so switching
  quantization correctly reports "not downloaded" and re-downloads, while the previously
  downloaded file stays on disk and keeps working if the user switches back.
* MODEL_COMPLETE_MARKER is shared by the two quantizations living in one model directory,
  so (unlike the Windows downloader) it is *not* unlinked when a download starts -- that
  would make an already complete sibling file look incomplete. It is instead rewritten
  atomically at the very end with the previous entries merged in. This is safe because the
  final file only appears via ``os.replace`` after its hash has been verified, so
  "marker exists AND selected file exists" can never be true for a partial download.

REPORT note on resume: both hops (huggingface.co -> the signed CDN URL) advertise
``accept-ranges: bytes``, and the signed URL expires in about an hour, which is why every
pass re-requests the ``huggingface.co`` ``/resolve/<commit>/<file>`` URL instead of caching
the redirect target. That is what the inherited ``_download_file`` does.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Union

from app.contracts import (
    APP_NAME,
    APP_VERSION,
    GGML_REPO,
    MODEL_COMPLETE_MARKER,
    MODELS,
    ComputeType,
    DownloadCancelledError,
    DownloadError,
    DownloadProgress,
    ThreadingEvent,
    ggml_model_file,
)
from app.engine.downloader import (
    CONNECT_TIMEOUT_S,
    HfModelDownloader,
    _Progress,
    _quiet_hf_hub,
    _RemoteFile,
)

logger = logging.getLogger("roenspeak.engine.ggml_downloader")


class GgmlModelDownloader(HfModelDownloader):
    """ModelDownloader implementation for macOS (see contracts.ModelDownloader).

    ``compute_type`` selects which ggml file is the target. The controller changes it
    through ``set_compute_type`` (looked up with ``hasattr``, so the name is part of the
    contract) whenever the configured compute type changes; it may do so from the UI thread
    while a download runs on a worker thread, so the value is lock-protected and every
    download works from one snapshot taken at its start.
    """

    def __init__(
        self,
        models_root: Optional[Path] = None,
        compute_type: Union[ComputeType, str] = ComputeType.FLOAT16,
    ) -> None:
        super().__init__(models_root)
        self._ct_lock = threading.Lock()
        self._compute_type = ComputeType(compute_type)  # ValueError for an unknown value

    # -- compute type -------------------------------------------------------------

    @property
    def compute_type(self) -> ComputeType:
        with self._ct_lock:
            return self._compute_type

    def set_compute_type(self, compute_type: Union[ComputeType, str]) -> None:
        """Switch which quantization is the download target. Any thread; takes effect for
        the next download() and immediately for is_downloaded()."""
        ct = ComputeType(compute_type)  # ValueError for an unknown value
        with self._ct_lock:
            if ct == self._compute_type:
                return
            self._compute_type = ct
        logger.info("ggml download target compute_type=%s", ct.value)

    # -- paths --------------------------------------------------------------------

    def model_file(self, model: str, compute_type: Optional[ComputeType] = None) -> Path:
        """The single ggml file for a model, for the selected (or given) compute type."""
        ct = self.compute_type if compute_type is None else ComputeType(compute_type)
        return self.model_path(model) / ggml_model_file(model, ct)

    def is_downloaded(self, model: str) -> bool:
        """True only when the file for the *currently selected* compute type is present and
        the completion marker exists. Both quantizations may coexist in the directory."""
        dest = self.model_path(model)  # ValueError for an unknown model
        try:
            target = dest / ggml_model_file(model, self.compute_type)
        except ValueError:  # pragma: no cover - every model has both ggml files
            return False
        return target.is_file() and (dest / MODEL_COMPLETE_MARKER).is_file()

    # -- download -----------------------------------------------------------------

    def download(
        self,
        model: str,
        on_progress: Callable[[DownloadProgress], None],
        cancel: ThreadingEvent,
    ) -> Path:
        if model not in MODELS:
            raise ValueError(f"Unknown model: {model!r}")
        ct = self.compute_type
        dest = self.model_path(model)
        filename = ggml_model_file(model, ct)
        if self.is_downloaded(model):
            logger.info("model %s already downloaded at %s", model, dest / filename)
            return dest
        if cancel.is_set():
            raise DownloadCancelledError("Download cancelled.")

        t0 = time.monotonic()
        remote, revision = self._lookup_file(filename)
        total = remote.size
        logger.info(
            "download start model=%s compute_type=%s repo=%s revision=%s file=%s total_bytes=%d dest=%s",
            model, ct.value, GGML_REPO, revision, filename, total, dest,
        )

        try:
            dest.mkdir(parents=True, exist_ok=True)
            # NOTE: the marker is deliberately left alone here; see the module docstring.
            self._reset_if_revision_changed(dest, revision, [remote])
        except OSError as e:
            raise DownloadError(f"Could not create the model folder {dest}: {e}", details=repr(e)) from e

        already = self._present_bytes(dest, remote)
        self._check_disk_space(dest, total - already)

        progress = _Progress(model, total, already, on_progress)
        progress.current_file = filename
        progress.maybe_emit(force=True)

        before = progress.done
        self._download_file(GGML_REPO, revision, dest, remote, progress, cancel)
        fetched = progress.done - before

        self._verify_and_mark(dest, [remote], GGML_REPO, revision)
        progress.current_file = ""
        progress.set_done(total)
        progress.maybe_emit(force=True)
        logger.info(
            "download done model=%s compute_type=%s bytes=%d fetched_bytes=%d seconds=%.1f",
            model, ct.value, total, fetched, time.monotonic() - t0,
        )
        return dest

    # -- helpers ------------------------------------------------------------------

    def _lookup_file(self, filename: str) -> tuple[_RemoteFile, str]:
        """Size + LFS sha256 + the repo commit, with no bytes downloaded."""
        _quiet_hf_hub()
        try:
            from huggingface_hub import HfApi
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()
            api = HfApi(library_name=APP_NAME, library_version=APP_VERSION)
            info = api.model_info(GGML_REPO, timeout=CONNECT_TIMEOUT_S, token=False)
            paths = api.get_paths_info(
                GGML_REPO, [filename], revision=info.sha, token=False
            )
        except Exception as e:
            raise self._wrap_error(e, f"Could not get the file list of {GGML_REPO}") from e

        entry = next((p for p in paths if getattr(p, "path", None) == filename), None)
        if entry is None:
            raise DownloadError(
                f"The model repository {GGML_REPO} does not contain {filename}.",
                details=f"repo={GGML_REPO} file={filename}",
            )
        size = getattr(entry, "size", None)
        lfs = getattr(entry, "lfs", None)
        if size is None and lfs is not None:
            size = getattr(lfs, "size", None)
        if size is None:
            raise DownloadError(
                f"The model repository did not report the size of {filename}.",
                details=f"repo={GGML_REPO}",
            )
        if not info.sha:
            raise DownloadError(f"The model repository {GGML_REPO} did not report a revision.")
        sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
        return _RemoteFile(filename, int(size), sha256), str(info.sha)

    @staticmethod
    def _verify_and_mark(dest: Path, files: list[_RemoteFile], repo: str, revision: str) -> None:
        """Re-check the downloaded file, then write MODEL_COMPLETE_MARKER last and
        atomically, keeping the entries an earlier compute type's download recorded."""
        for f in files:
            p = dest / f.name
            if not p.is_file() or p.stat().st_size != f.size:
                raise DownloadError(
                    f"{f.name} is incomplete after download. Please try again.",
                    details=f"path={p}",
                )
        marker_path = dest / MODEL_COMPLETE_MARKER
        known: dict[str, int] = {}
        if marker_path.is_file():
            try:
                old = json.loads(marker_path.read_text(encoding="utf-8"))
                if isinstance(old, dict) and isinstance(old.get("files"), dict):
                    known = {
                        str(k): int(v)
                        for k, v in old["files"].items()
                        if isinstance(v, int) and (dest / str(k)).is_file()
                    }
            except (OSError, ValueError, TypeError):
                logger.debug("ignoring an unreadable %s", marker_path)
        known.update({f.name: f.size for f in files})
        marker = {
            "repo": repo,
            "revision": revision,
            "files": known,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        try:
            tmp = dest / (MODEL_COMPLETE_MARKER + ".tmp")
            tmp.write_text(json.dumps(marker, indent=2), encoding="utf-8")
            os.replace(tmp, marker_path)
        except OSError as e:
            raise DownloadError(f"Could not finalize the model folder {dest}: {e}", details=repr(e)) from e
