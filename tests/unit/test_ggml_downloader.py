"""Spec/contract tests for app.engine.ggml_downloader.GgmlModelDownloader (ModelDownloader).

Expectations come from contracts.ModelDownloader / DownloadProgress / DownloadError /
DownloadCancelledError / MODEL_COMPLETE_MARKER / ggml_model_file / GGML_REPO, from SPEC.md
("First run shows a download dialog with progress ... resumable, clear error if offline")
and from ARCHITECTURE.md section 12.1 (one single-file ggml model per model + compute type,
the quantization is baked into the file).

The Hub is faked at two seams -- HfApi (the file manifest) and huggingface_hub's HTTP
session (the bytes) -- so nothing here touches the network and no model is ever fetched.
The clock is faked where throttling is measured, so there are no real sleeps either.
"""

from __future__ import annotations

import hashlib
import json
import socket
import threading
import time
from pathlib import Path

import httpx2
import huggingface_hub
import huggingface_hub.utils
import pytest

from app.contracts import (
    GGML_REPO,
    MODEL_COMPLETE_MARKER,
    ComputeType,
    DownloadCancelledError,
    DownloadError,
    DownloadProgress,
    ModelDownloader,
    ggml_model_file,
)
from app.engine import downloader as dl
from app.engine.ggml_downloader import GgmlModelDownloader

MODEL = "large-v3"
REVISION = "0123456789abcdef0123456789abcdef01234567"

# ---------------------------------------------------------------------------
# Fake Hugging Face Hub (manifest + bytes)
# ---------------------------------------------------------------------------


class FakeLfs:
    def __init__(self, size: int, sha256: str | None) -> None:
        self.size = size
        self.sha256 = sha256


class FakeEntry:
    """What HfApi.get_paths_info returns for one repo file."""

    def __init__(self, path: str, size: int, sha256: str | None) -> None:
        self.path = path
        self.size = size
        self.lfs = FakeLfs(size, sha256)


class FakeModelInfo:
    def __init__(self, sha: str) -> None:
        self.sha = sha


class FakeResponse:
    """The slice of an httpx streaming response the downloader uses."""

    def __init__(self, status_code: int, body: bytes, chunk_size: int, on_chunk=None) -> None:
        self.status_code = status_code
        self._body = body
        self._chunk_size = chunk_size
        self._on_chunk = on_chunk

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def read(self) -> bytes:
        return self._body

    @property
    def text(self) -> str:
        return self._body.decode("utf-8", "replace")

    def iter_bytes(self, size: int):
        step = min(size, self._chunk_size)
        for i, start in enumerate(range(0, len(self._body), step)):
            if self._on_chunk is not None:
                self._on_chunk(i)
            yield self._body[start : start + step]


class FakeHub:
    """One repo file, served over a fake HTTP session. Everything is scriptable."""

    #: Sentinel: "derive the hash from whatever content is currently served".
    DERIVE = object()

    def __init__(self, content: bytes = b"") -> None:
        self.content = content
        self._sha256: object = self.DERIVE
        self.revision = REVISION
        self.filename: str | None = None  # None = serve whatever is asked for
        self.reported_size: int | None = None  # None = len(content)
        self.manifest_error: BaseException | None = None
        self.stream_error: BaseException | None = None
        self.force_status: int | None = None
        self.status_once: int | None = None  # answered once, then back to normal
        self.error_body = b"gateway timeout"
        self.ignore_range = False
        self.chunk_size = 1 << 20
        self.on_chunk = None
        self.requests: list[dict[str, str]] = []
        self.urls: list[str] = []
        self.manifest_calls = 0

    # -- manifest ----------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self.content) if self.reported_size is None else self.reported_size

    @property
    def sha256(self) -> str | None:
        if self._sha256 is self.DERIVE:
            return hashlib.sha256(self.content).hexdigest()
        return self._sha256  # type: ignore[return-value]

    @sha256.setter
    def sha256(self, value: str | None) -> None:
        self._sha256 = value

    def model_info(self, repo, **kw):
        self.manifest_calls += 1
        if self.manifest_error is not None:
            raise self.manifest_error
        assert repo == GGML_REPO
        return FakeModelInfo(self.revision)

    def paths_info(self, repo, paths, **kw):
        if self.manifest_error is not None:
            raise self.manifest_error
        name = paths[0]
        if self.filename is not None and name != self.filename:
            return []
        return [FakeEntry(name, self.size, self.sha256)]

    # -- bytes -------------------------------------------------------------------

    def open_stream(self, url: str, headers: dict) -> FakeResponse:
        self.urls.append(url)
        self.requests.append(dict(headers))
        if self.stream_error is not None:
            raise self.stream_error
        if self.status_once is not None:
            status, self.status_once = self.status_once, None
            return FakeResponse(status, b"", self.chunk_size)
        if self.force_status is not None:
            return FakeResponse(self.force_status, self.error_body, self.chunk_size)
        rng = headers.get("Range")
        if not rng:
            return FakeResponse(200, self.content, self.chunk_size, self.on_chunk)
        start = int(rng.removeprefix("bytes=").split("-")[0])
        if self.ignore_range:
            return FakeResponse(200, self.content, self.chunk_size, self.on_chunk)
        if start >= len(self.content):
            return FakeResponse(416, b"", self.chunk_size)
        return FakeResponse(206, self.content[start:], self.chunk_size, self.on_chunk)


class FakeSession:
    def __init__(self, hub: FakeHub) -> None:
        self.hub = hub

    def stream(self, method, url, headers=None, timeout=None, follow_redirects=False):
        assert method == "GET"
        return self.hub.open_stream(url, headers or {})


class FakeTime:
    """app.engine.downloader's clock, so progress throttling is measured deterministically."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def strftime(self, fmt: str) -> str:
        return time.strftime(fmt)

    def advance(self, dt: float) -> None:
        self.now += dt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hub(monkeypatch) -> FakeHub:
    """The fake Hub, wired into both seams. Any real network call would fail loudly."""
    h = FakeHub(content=b"")

    class _Api:
        def __init__(self, *a, **kw) -> None:
            pass

        def model_info(self, repo, **kw):
            return h.model_info(repo, **kw)

        def get_paths_info(self, repo, paths, **kw):
            return h.paths_info(repo, paths, **kw)

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(huggingface_hub.utils, "get_session", lambda: FakeSession(h))
    return h


@pytest.fixture
def models_root(tmp_path) -> Path:
    return tmp_path / "models"


@pytest.fixture
def down(models_root) -> GgmlModelDownloader:
    return GgmlModelDownloader(models_root, ComputeType.FLOAT16)


@pytest.fixture
def cancel() -> threading.Event:
    return threading.Event()


class Recorder:
    """Progress sink that also snapshots the on-disk state at every callback."""

    def __init__(self, path: Path, downloader: GgmlModelDownloader | None = None, model: str = MODEL,
                 raises: bool = False) -> None:
        self.path = path
        self.downloader = downloader
        self.model = model
        self.raises = raises
        self.events: list[DownloadProgress] = []
        self.marker_seen: list[bool] = []
        self.complete_seen: list[bool] = []
        self.calls = 0

    def __call__(self, p: DownloadProgress) -> None:
        self.calls += 1
        self.events.append(p)
        self.marker_seen.append((self.path / MODEL_COMPLETE_MARKER).is_file())
        if self.downloader is not None:
            self.complete_seen.append(self.downloader.is_downloaded(self.model))
        if self.raises:
            raise RuntimeError("the progress dialog exploded")


def content(n: int, seed: bytes = b"ggml") -> bytes:
    """Deterministic, compressible-but-not-degenerate bytes."""
    return (seed * ((n // len(seed)) + 1))[:n]


def mark_complete(models_root: Path, model: str, compute_type: ComputeType, size: int = 64) -> Path:
    d = models_root / model
    d.mkdir(parents=True, exist_ok=True)
    f = d / ggml_model_file(model, compute_type)
    f.write_bytes(content(size))
    marker = d / MODEL_COMPLETE_MARKER
    files = {}
    if marker.is_file():
        files = json.loads(marker.read_text(encoding="utf-8"))["files"]
    files[f.name] = size
    marker.write_text(json.dumps({"repo": GGML_REPO, "revision": REVISION, "files": files}), encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# Contract shape and paths
# ---------------------------------------------------------------------------


def test_is_a_model_downloader(down):
    assert isinstance(down, ModelDownloader)


def test_default_compute_type_is_float16(models_root):
    assert GgmlModelDownloader(models_root).compute_type is ComputeType.FLOAT16


def test_an_unknown_compute_type_is_rejected(models_root):
    with pytest.raises(ValueError):
        GgmlModelDownloader(models_root, "float32")
    with pytest.raises(ValueError):
        GgmlModelDownloader(models_root).set_compute_type("bfloat16")


@pytest.mark.parametrize(
    "model,compute_type,filename",
    [
        ("large-v3", ComputeType.FLOAT16, "ggml-large-v3.bin"),
        ("large-v3", ComputeType.INT8_FLOAT16, "ggml-large-v3-q5_0.bin"),
        ("medium", ComputeType.FLOAT16, "ggml-medium.bin"),
        ("large-v3-turbo", ComputeType.INT8_FLOAT16, "ggml-large-v3-turbo-q5_0.bin"),
    ],
)
def test_model_path_and_model_file(models_root, model, compute_type, filename):
    d = GgmlModelDownloader(models_root, compute_type)
    assert d.model_path(model) == models_root / model
    assert d.model_file(model) == models_root / model / filename
    assert d.model_file(model).name == ggml_model_file(model, compute_type)


def test_model_file_accepts_an_explicit_compute_type(down):
    assert down.model_file(MODEL, ComputeType.INT8_FLOAT16).name == "ggml-large-v3-q5_0.bin"


@pytest.mark.parametrize("model", ["tiny", "", "LARGE-V3", "distil-small.en"])
def test_unknown_model_raises_value_error(down, model):
    with pytest.raises(ValueError):
        down.model_path(model)
    with pytest.raises(ValueError):
        down.model_file(model)
    with pytest.raises(ValueError):
        down.is_downloaded(model)


def test_unknown_model_raises_value_error_from_download(down, cancel):
    with pytest.raises(ValueError):
        down.download("tiny", lambda p: None, cancel)


def test_models_root_defaults_to_models_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("ROENSPEAK_MODELS_DIR", str(tmp_path / "m"))
    assert GgmlModelDownloader().model_path(MODEL) == tmp_path / "m" / MODEL


# ---------------------------------------------------------------------------
# is_downloaded(): the selected compute type decides
# ---------------------------------------------------------------------------


def test_not_downloaded_when_nothing_is_there(down):
    assert down.is_downloaded(MODEL) is False


def test_not_downloaded_without_the_marker(down, models_root):
    (models_root / MODEL).mkdir(parents=True)
    (models_root / MODEL / "ggml-large-v3.bin").write_bytes(b"x")
    assert down.is_downloaded(MODEL) is False


def test_not_downloaded_without_the_file(down, models_root):
    (models_root / MODEL).mkdir(parents=True)
    (models_root / MODEL / MODEL_COMPLETE_MARKER).write_text("{}", encoding="utf-8")
    assert down.is_downloaded(MODEL) is False


def test_downloaded_when_the_selected_file_and_the_marker_exist(down, models_root):
    mark_complete(models_root, MODEL, ComputeType.FLOAT16)
    assert down.is_downloaded(MODEL) is True


def test_set_compute_type_flips_is_downloaded(down, models_root):
    mark_complete(models_root, MODEL, ComputeType.FLOAT16)
    assert down.is_downloaded(MODEL) is True

    down.set_compute_type(ComputeType.INT8_FLOAT16)
    assert down.compute_type is ComputeType.INT8_FLOAT16
    # The q5_0 file is a different file, so the model counts as not downloaded.
    assert down.is_downloaded(MODEL) is False

    down.set_compute_type(ComputeType.FLOAT16)
    assert down.is_downloaded(MODEL) is True


def test_both_quantizations_coexist_in_one_directory(down, models_root):
    """Switching quantization must not invalidate the other file (ARCHITECTURE 12.1)."""
    mark_complete(models_root, MODEL, ComputeType.FLOAT16)
    mark_complete(models_root, MODEL, ComputeType.INT8_FLOAT16)
    names = sorted(p.name for p in (models_root / MODEL).iterdir())
    assert names == [MODEL_COMPLETE_MARKER, "ggml-large-v3-q5_0.bin", "ggml-large-v3.bin"]
    for ct in ComputeType:
        down.set_compute_type(ct)
        assert down.is_downloaded(MODEL) is True


def test_set_compute_type_accepts_a_plain_string(down, models_root):
    mark_complete(models_root, MODEL, ComputeType.INT8_FLOAT16)
    down.set_compute_type("int8_float16")
    assert down.is_downloaded(MODEL) is True


def test_is_downloaded_is_per_model(down, models_root):
    mark_complete(models_root, MODEL, ComputeType.FLOAT16)
    assert down.is_downloaded("medium") is False


# ---------------------------------------------------------------------------
# download(): the happy path
# ---------------------------------------------------------------------------


def test_download_writes_the_file_the_engine_loads(down, models_root, hub, cancel):
    hub.content = content(40_000)
    rec = Recorder(models_root / MODEL, down)
    dest = down.download(MODEL, rec, cancel)

    assert dest == models_root / MODEL
    target = dest / "ggml-large-v3.bin"
    assert target.read_bytes() == hub.content
    assert down.is_downloaded(MODEL) is True
    # No .part left behind.
    assert list(dest.glob("*" + dl.PART_SUFFIX)) == []


def test_download_requests_the_file_for_the_selected_compute_type(down, models_root, hub, cancel):
    hub.content = content(1_000)
    down.set_compute_type(ComputeType.INT8_FLOAT16)
    down.download(MODEL, lambda p: None, cancel)
    assert (models_root / MODEL / "ggml-large-v3-q5_0.bin").is_file()
    assert "ggml-large-v3-q5_0.bin" in hub.urls[0]
    assert REVISION in hub.urls[0]


def test_download_returns_immediately_when_already_downloaded(down, models_root, hub, cancel):
    mark_complete(models_root, MODEL, ComputeType.FLOAT16, size=128)
    hub.manifest_error = AssertionError("the Hub must not be contacted")
    rec = Recorder(models_root / MODEL, down)

    dest = down.download(MODEL, rec, cancel)
    assert dest == models_root / MODEL
    assert hub.manifest_calls == 0
    assert rec.events == []
    # The existing bytes were not touched.
    assert (dest / "ggml-large-v3.bin").read_bytes() == content(128)


def test_a_cancel_that_is_already_set_does_not_start(down, hub, cancel):
    hub.content = content(1_000)
    cancel.set()
    with pytest.raises(DownloadCancelledError):
        down.download(MODEL, lambda p: None, cancel)
    assert hub.manifest_calls == 0


def test_the_marker_records_the_file_and_revision(down, models_root, hub, cancel):
    hub.content = content(2_048)
    down.download(MODEL, lambda p: None, cancel)
    marker = json.loads((models_root / MODEL / MODEL_COMPLETE_MARKER).read_text(encoding="utf-8"))
    assert marker["repo"] == GGML_REPO
    assert marker["revision"] == REVISION
    assert marker["files"] == {"ggml-large-v3.bin": 2_048}


def test_downloading_the_second_quantization_keeps_the_first_in_the_marker(
    down, models_root, hub, cancel
):
    hub.content = content(1_024)
    down.download(MODEL, lambda p: None, cancel)
    down.set_compute_type(ComputeType.INT8_FLOAT16)
    hub.content = content(512, b"q5_0")
    down.download(MODEL, lambda p: None, cancel)

    marker = json.loads((models_root / MODEL / MODEL_COMPLETE_MARKER).read_text(encoding="utf-8"))
    assert marker["files"] == {"ggml-large-v3.bin": 1_024, "ggml-large-v3-q5_0.bin": 512}
    for ct in ComputeType:
        down.set_compute_type(ct)
        assert down.is_downloaded(MODEL) is True


# ---------------------------------------------------------------------------
# download(): progress
# ---------------------------------------------------------------------------


def test_progress_is_forced_first_and_last(down, models_root, hub, cancel):
    hub.content = content(4_096)
    rec = Recorder(models_root / MODEL, down)
    down.download(MODEL, rec, cancel)

    assert len(rec.events) >= 2
    first, last = rec.events[0], rec.events[-1]
    assert first.model == MODEL
    assert first.downloaded_bytes == 0
    assert first.total_bytes == 4_096
    assert first.current_file == "ggml-large-v3.bin"
    # The last emission reports the finished download and no current file.
    assert last.current_file == ""
    assert last.downloaded_bytes == last.total_bytes == 4_096


def test_progress_is_throttled_to_ten_per_second(down, models_root, hub, cancel, monkeypatch):
    """contracts.ModelDownloader: "on_progress is called ... at most ~10x/s"."""
    clock = FakeTime()
    monkeypatch.setattr(dl, "time", clock)
    chunks, chunk = 200, 1_000
    hub.content = content(chunks * chunk)
    hub.chunk_size = chunk
    hub.on_chunk = lambda i: clock.advance(0.01)  # 200 chunks over 2.0 simulated seconds

    rec = Recorder(models_root / MODEL, down)
    down.download(MODEL, rec, cancel)

    elapsed = clock.now - 1_000.0
    assert elapsed == pytest.approx(2.0)
    # 10/s over the elapsed time, plus the two forced emissions.
    assert len(rec.events) <= int(elapsed * 10) + 2
    assert len(rec.events) < chunks  # i.e. it really is throttled
    assert len(rec.events) >= 2
    assert clock.slept == []  # no retries, no sleeping


def test_progress_bytes_never_go_backwards(down, models_root, hub, cancel):
    hub.content = content(50_000)
    hub.chunk_size = 997
    rec = Recorder(models_root / MODEL, down)
    down.download(MODEL, rec, cancel)

    seen = [p.downloaded_bytes for p in rec.events]
    assert seen == sorted(seen)
    assert all(0 <= b <= p.total_bytes for b, p in zip(seen, rec.events))
    assert all(p.total_bytes == 50_000 for p in rec.events)


def test_progress_rate_is_sane(down, models_root, hub, cancel, monkeypatch):
    clock = FakeTime()
    monkeypatch.setattr(dl, "time", clock)
    hub.content = content(100_000)
    hub.chunk_size = 1_000
    hub.on_chunk = lambda i: clock.advance(0.01)
    rec = Recorder(models_root / MODEL, down)
    down.download(MODEL, rec, cancel)

    rates = [p.bytes_per_s for p in rec.events]
    assert all(r >= 0.0 for r in rates)
    assert all(r == r and r != float("inf") for r in rates)  # finite, not NaN
    # 1000 bytes per 0.01 s = 100 kB/s; the EMA must land in that ballpark.
    assert any(10_000 < r < 1_000_000 for r in rates)


def test_the_marker_is_written_last(down, models_root, hub, cancel):
    """An interrupted download must never look complete, so the marker appears only after
    the verified file is in place -- it is absent at every mid-download callback."""
    hub.content = content(30_000)
    hub.chunk_size = 1_000
    rec = Recorder(models_root / MODEL, down)
    down.download(MODEL, rec, cancel)

    assert len(rec.events) >= 2
    mid = [seen for seen, p in zip(rec.marker_seen, rec.events) if p.current_file]
    assert mid and not any(mid)
    assert rec.marker_seen[-1] is True  # only the final, forced emission sees it
    assert rec.complete_seen[-1] is True
    assert not any(rec.complete_seen[:-1])


def test_a_sibling_quantization_marker_never_makes_a_partial_download_look_complete(
    down, models_root, hub, cancel
):
    """The marker is shared by both quantizations, so it is deliberately *not* removed when
    a download starts. is_downloaded() stays False anyway, because the selected file only
    appears (via os.replace) after its hash was verified."""
    mark_complete(models_root, MODEL, ComputeType.FLOAT16)
    down.set_compute_type(ComputeType.INT8_FLOAT16)
    hub.content = content(30_000, b"q5_0")
    hub.chunk_size = 1_000
    rec = Recorder(models_root / MODEL, down)
    down.download(MODEL, rec, cancel)

    assert all(rec.marker_seen)  # the sibling's marker was there all along
    assert not any(rec.complete_seen[:-1])  # yet the model never looked downloaded
    assert rec.complete_seen[-1] is True


def test_a_progress_callback_that_raises_does_not_break_the_download(
    down, models_root, hub, cancel
):
    hub.content = content(20_000)
    hub.chunk_size = 1_000
    rec = Recorder(models_root / MODEL, down, raises=True)
    dest = down.download(MODEL, rec, cancel)

    assert rec.calls >= 2
    assert (dest / "ggml-large-v3.bin").read_bytes() == hub.content
    assert down.is_downloaded(MODEL) is True


# ---------------------------------------------------------------------------
# download(): cancellation
# ---------------------------------------------------------------------------


def test_cancelling_mid_download_keeps_the_partial_data(down, models_root, hub, cancel):
    hub.content = content(50_000)
    hub.chunk_size = 1_000
    hub.on_chunk = lambda i: cancel.set() if i == 10 else None

    with pytest.raises(DownloadCancelledError):
        down.download(MODEL, lambda p: None, cancel)

    dest = models_root / MODEL
    part = dest / ("ggml-large-v3.bin" + dl.PART_SUFFIX)
    assert part.is_file()
    assert 0 < part.stat().st_size < len(hub.content)  # partial data kept for resume
    assert not (dest / "ggml-large-v3.bin").exists()
    assert not (dest / MODEL_COMPLETE_MARKER).exists()
    assert down.is_downloaded(MODEL) is False


def test_a_cancelled_download_resumes_from_the_part_file(down, models_root, hub, cancel):
    hub.content = content(50_000)
    hub.chunk_size = 1_000
    hub.on_chunk = lambda i: cancel.set() if i == 10 else None
    with pytest.raises(DownloadCancelledError):
        down.download(MODEL, lambda p: None, cancel)
    part = models_root / MODEL / ("ggml-large-v3.bin" + dl.PART_SUFFIX)
    already = part.stat().st_size

    cancel.clear()
    hub.on_chunk = None
    hub.requests.clear()
    dest = down.download(MODEL, lambda p: None, cancel)

    assert hub.requests[0].get("Range") == f"bytes={already}-"
    assert (dest / "ggml-large-v3.bin").read_bytes() == hub.content
    assert down.is_downloaded(MODEL) is True


# ---------------------------------------------------------------------------
# download(): failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("Name or service not known"),
        ConnectionResetError("connection reset by peer"),
        TimeoutError("timed out"),
        socket.gaierror("nodename nor servname provided"),
        httpx2.ConnectError("could not connect"),
        httpx2.ReadError("read error"),
        httpx2.ProxyError("proxy refused"),
    ],
)
def test_a_connectivity_failure_is_an_offline_download_error(down, hub, cancel, error):
    """SPEC: "clear error if offline". DownloadError.offline drives that message."""
    hub.manifest_error = error
    with pytest.raises(DownloadError) as e:
        down.download(MODEL, lambda p: None, cancel)
    assert e.value.offline is True
    assert e.value.user_message
    assert "connection" in e.value.user_message.lower()


def test_a_wrapped_connectivity_failure_is_still_offline(down, hub, cancel):
    """The cause/context chain is what identifies a network failure hidden in a wrapper."""
    try:
        raise ConnectionError("connection refused")
    except ConnectionError as cause:
        hub.manifest_error = RuntimeError("the Hub client failed")
        hub.manifest_error.__cause__ = cause
    with pytest.raises(DownloadError) as e:
        down.download(MODEL, lambda p: None, cancel)
    assert e.value.offline is True


def test_a_connectivity_failure_mid_stream_is_offline_too(down, hub, cancel, monkeypatch):
    monkeypatch.setattr(dl, "MAX_STREAM_RETRIES", 0)  # no real backoff sleeping
    hub.content = content(10_000)
    hub.stream_error = ConnectionResetError("connection reset by peer")
    with pytest.raises(DownloadError) as e:
        down.download(MODEL, lambda p: None, cancel)
    assert e.value.offline is True


@pytest.mark.parametrize(
    "error",
    [
        ValueError("the Hub answered something unexpected"),
        KeyError("sha"),
        RuntimeError("boom"),
    ],
)
def test_other_manifest_failures_are_not_offline(down, hub, cancel, error):
    hub.manifest_error = error
    with pytest.raises(DownloadError) as e:
        down.download(MODEL, lambda p: None, cancel)
    assert e.value.offline is False


def test_an_http_error_is_a_download_error_that_is_not_offline(down, hub, cancel):
    hub.content = content(1_000)
    hub.force_status = 503
    with pytest.raises(DownloadError) as e:
        down.download(MODEL, lambda p: None, cancel)
    assert e.value.offline is False
    assert "503" in e.value.user_message


def test_a_repo_without_the_file_is_a_download_error(down, hub, cancel):
    hub.content = content(1_000)
    hub.filename = "ggml-something-else.bin"
    with pytest.raises(DownloadError) as e:
        down.download(MODEL, lambda p: None, cancel)
    assert e.value.offline is False
    assert "ggml-large-v3.bin" in e.value.user_message


def test_a_missing_revision_is_a_download_error(down, hub, cancel):
    hub.content = content(1_000)
    hub.revision = ""
    with pytest.raises(DownloadError):
        down.download(MODEL, lambda p: None, cancel)


def test_a_hash_mismatch_discards_the_download(down, models_root, hub, cancel):
    hub.content = content(20_000)
    hub.sha256 = hashlib.sha256(b"a completely different file").hexdigest()
    rec = Recorder(models_root / MODEL, down)

    with pytest.raises(DownloadError) as e:
        down.download(MODEL, rec, cancel)
    assert e.value.offline is False
    assert "try again" in e.value.user_message.lower()

    dest = models_root / MODEL
    # Neither the final file nor the marker is left behind, and the bad bytes are gone.
    assert not (dest / "ggml-large-v3.bin").exists()
    assert not (dest / MODEL_COMPLETE_MARKER).exists()
    assert not (dest / ("ggml-large-v3.bin" + dl.PART_SUFFIX)).exists()
    assert down.is_downloaded(MODEL) is False
    assert not any(rec.marker_seen)


def test_a_short_file_never_completes(down, models_root, hub, cancel, monkeypatch):
    """The manifest promises more bytes than the server delivers: the download must end in
    a DownloadError, never in a model directory that looks complete."""
    monkeypatch.setattr(dl, "MAX_STREAM_RETRIES", 0)  # no real backoff sleeping
    hub.content = content(1_000)
    hub.reported_size = 5_000
    hub.sha256 = None  # no LFS hash to catch it; the size check must
    hub.ignore_range = True
    with pytest.raises(DownloadError):
        down.download(MODEL, lambda p: None, cancel)
    assert not (models_root / MODEL / MODEL_COMPLETE_MARKER).exists()
    assert not (models_root / MODEL / "ggml-large-v3.bin").exists()
    assert down.is_downloaded(MODEL) is False


def test_an_oversized_part_file_is_discarded(down, models_root, hub, cancel):
    dest = models_root / MODEL
    dest.mkdir(parents=True)
    hub.content = content(5_000)
    (dest / ("ggml-large-v3.bin" + dl.PART_SUFFIX)).write_bytes(content(5_000) + b"junk")
    (dest / dl.REVISION_FILE).write_text(REVISION, encoding="utf-8")

    down.download(MODEL, lambda p: None, cancel)
    assert (dest / "ggml-large-v3.bin").read_bytes() == hub.content


def test_an_unsatisfiable_range_restarts_the_file(down, models_root, hub, cancel):
    """HTTP 416 on resume means our .part is unusable; it must be dropped and refetched."""
    dest = models_root / MODEL
    dest.mkdir(parents=True)
    hub.content = content(5_000)
    hub.status_once = 416
    (dest / ("ggml-large-v3.bin" + dl.PART_SUFFIX)).write_bytes(content(2_500))
    (dest / dl.REVISION_FILE).write_text(REVISION, encoding="utf-8")

    down.download(MODEL, lambda p: None, cancel)
    assert (dest / "ggml-large-v3.bin").read_bytes() == hub.content
    assert down.is_downloaded(MODEL) is True


def test_a_server_that_ignores_range_restarts_the_file(down, models_root, hub, cancel):
    dest = models_root / MODEL
    dest.mkdir(parents=True)
    hub.content = content(5_000)
    hub.ignore_range = True
    (dest / ("ggml-large-v3.bin" + dl.PART_SUFFIX)).write_bytes(content(1_000))
    (dest / dl.REVISION_FILE).write_text(REVISION, encoding="utf-8")

    down.download(MODEL, lambda p: None, cancel)
    assert (dest / "ggml-large-v3.bin").read_bytes() == hub.content


def test_a_new_repo_revision_discards_stale_parts(down, models_root, hub, cancel):
    dest = models_root / MODEL
    dest.mkdir(parents=True)
    (dest / ("ggml-large-v3.bin" + dl.PART_SUFFIX)).write_bytes(b"stale bytes from an old commit")
    (dest / dl.REVISION_FILE).write_text("f" * 40, encoding="utf-8")
    hub.content = content(3_000)

    down.download(MODEL, lambda p: None, cancel)
    assert (dest / "ggml-large-v3.bin").read_bytes() == hub.content
    assert (dest / dl.REVISION_FILE).read_text(encoding="utf-8") == REVISION


# ---------------------------------------------------------------------------
# threading
# ---------------------------------------------------------------------------


def test_set_compute_type_is_safe_while_a_download_runs(down, models_root, hub, cancel):
    """The UI thread may switch compute type mid-download; the running download must keep
    working from the snapshot it started with."""
    hub.content = content(20_000)
    hub.chunk_size = 1_000
    hub.on_chunk = lambda i: down.set_compute_type(
        ComputeType.INT8_FLOAT16 if i % 2 else ComputeType.FLOAT16
    )
    down.download(MODEL, lambda p: None, cancel)
    # The file that was being fetched is the one that landed.
    assert (models_root / MODEL / "ggml-large-v3.bin").read_bytes() == hub.content
    assert len(hub.urls) == 1
    assert all("ggml-large-v3.bin" in u for u in hub.urls)


# ---------------------------------------------------------------------------
# a server that can neither finish nor resume must fail, not spin
# ---------------------------------------------------------------------------


def test_a_server_that_keeps_refusing_to_resume_fails_instead_of_looping(
    down, models_root, hub, cancel
):
    """Regression: a manifest size larger than the bytes actually served, plus 416 on the
    resume request, used to loop forever (full GET -> short write -> 416 -> discard -> ...).
    `retries` could never trip because every written chunk resets it, so the download neither
    finished nor failed and the user only saw downloaded_bytes oscillate. The restart counter
    must stop it with a DownloadError instead."""
    hub.content = content(1_000)
    hub.reported_size = 5_000  # the server will never serve this much

    with pytest.raises(DownloadError) as excinfo:
        down.download(MODEL, lambda p: None, cancel)

    assert "resume" in excinfo.value.user_message
    assert excinfo.value.offline is False
    # It gave up after a bounded number of restarts rather than hammering the server.
    assert len(hub.requests) <= 2 * (dl.MAX_STREAM_RETRIES + 1) + 2
    # Nothing incomplete is left behind that could later look finished.
    dest = models_root / MODEL
    assert not (dest / "ggml-large-v3.bin").exists()
    assert not (dest / MODEL_COMPLETE_MARKER).exists()


def test_a_server_that_keeps_ignoring_range_fails_instead_of_looping(
    down, models_root, hub, cancel
):
    """Same guard for the other restart path: a server that answers a Range request with a
    full 200 body that is still short of the manifest size."""
    hub.content = content(1_000)
    hub.reported_size = 5_000
    hub.ignore_range = True

    with pytest.raises(DownloadError) as excinfo:
        down.download(MODEL, lambda p: None, cancel)

    assert "restarting" in excinfo.value.user_message or "resume" in excinfo.value.user_message
    assert not (models_root / MODEL / "ggml-large-v3.bin").exists()
