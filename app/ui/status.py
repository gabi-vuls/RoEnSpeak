"""Pure formatting helpers: AppState -> status text / tray icon state, sizes, rates, latency."""

from __future__ import annotations

from typing import Optional

from app.contracts import IS_MACOS, AppState, EngineInfo

#: Status-line wording per state (spec: idle / downloading / loading model / ready / listening /
#: processing / error).
STATE_LABELS: dict[AppState, str] = {
    AppState.DISARMED: "Idle",
    AppState.DOWNLOADING: "Downloading model",
    AppState.LOADING_MODEL: "Loading model",
    AppState.READY: "Ready",
    AppState.LISTENING: "Listening",
    AppState.PROCESSING: "Processing",
    AppState.UNLOADED: "Ready (model unloaded)",
    AppState.ERROR: "Error",
}

#: Tray icon (icon_art.TRAY_STATES) per state.
TRAY_STATE: dict[AppState, str] = {
    AppState.DISARMED: "disarmed",
    AppState.DOWNLOADING: "loading",
    AppState.LOADING_MODEL: "loading",
    AppState.READY: "ready",
    AppState.LISTENING: "listening",
    AppState.PROCESSING: "processing",
    AppState.UNLOADED: "ready",
    AppState.ERROR: "error",
}

#: Status dot colour for a non-fatal ERROR (needs attention, not broken).
SOFT_ERROR_COLOR = "#f59e0b"

#: Status dot colours per state (same palette as the tray icons).
STATE_COLORS: dict[AppState, str] = {
    AppState.DISARMED: "#8a8f98",
    AppState.DOWNLOADING: "#2f7fd8",
    AppState.LOADING_MODEL: "#2f7fd8",
    AppState.READY: "#1f9d55",
    AppState.LISTENING: "#e02424",
    AppState.PROCESSING: "#f59e0b",
    AppState.UNLOADED: "#1f9d55",
    AppState.ERROR: "#b91c1c",
}


def status_text(
    state: AppState,
    *,
    detail: str = "",
    hotkey: Optional[str] = None,
    download_pct: Optional[float] = None,
    hard_error: bool = True,
) -> str:
    """hard_error=False renders ERROR as a recoverable "needs attention" state (e.g. the model
    is not downloaded yet) using the detail text instead of an "Error:" label."""
    label = STATE_LABELS[state]
    if state == AppState.DISARMED:
        return f"{label} - hotkey is off"
    if state == AppState.DOWNLOADING:
        pct = f" {download_pct:.0f}%" if download_pct is not None else ""
        return f"{label}{pct}…"
    if state == AppState.LOADING_MODEL:
        return f"{label} {detail}…" if detail else f"{label}…"
    if state in (AppState.READY, AppState.UNLOADED):
        return f"{label} - press {hotkey} to dictate" if hotkey else label
    if state == AppState.LISTENING:
        return f"{label} - press {hotkey} to stop" if hotkey else f"{label}…"
    if state == AppState.PROCESSING:
        return f"{label}…"
    if not hard_error:
        return detail or "Action needed"
    return f"{label}: {detail}" if detail else label


#: Shown when the engine reports a GPU device but no device name. "CUDA GPU" would be a lie
#: on macOS, where the engine runs on Metal (and normally reports the real chip, e.g.
#: "Apple M2 Pro").
DEFAULT_GPU_NAME = "Metal GPU" if IS_MACOS else "CUDA GPU"


def engine_text(info: Optional[EngineInfo]) -> str:
    if info is None:
        return "Model not loaded"
    if info.device == "cpu":
        return f"CPU mode (slow) · {info.model} · {info.compute_type}"
    gpu = info.gpu_name or DEFAULT_GPU_NAME
    return f"{gpu} · {info.model} · {info.compute_type}"


def latency_text(ms: Optional[float]) -> str:
    return "Last latency: -" if ms is None else f"Last latency: {ms:.0f} ms"


def format_bytes(n: float) -> str:
    """Decimal (SI) units, matching ModelSpec.download_mb."""
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.0f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.0f} KB"
    return f"{int(n)} B"


def format_rate(bytes_per_s: float) -> str:
    return f"{bytes_per_s / 1e6:.1f} MB/s" if bytes_per_s >= 1e6 else f"{bytes_per_s / 1e3:.0f} KB/s"


def format_eta(remaining_bytes: float, bytes_per_s: float) -> str:
    if bytes_per_s <= 0 or remaining_bytes <= 0:
        return ""
    s = remaining_bytes / bytes_per_s
    if s < 60:
        return "less than a minute left"
    if s < 3600:
        return f"about {s / 60:.0f} min left"
    return f"about {s / 3600:.1f} h left"
