# Bundled Silero VAD asset

`silero_vad_v6.onnx` is the exact asset that faster-whisper bundles
(`faster_whisper/assets/silero_vad_v6.onnx`), vendored here for the macOS build.

- Source: https://raw.githubusercontent.com/SYSTRAN/faster-whisper/9e472ae9de8bb374552f531f752654554b76b2d6/faster_whisper/assets/silero_vad_v6.onnx
- faster-whisper commit: `9e472ae9de8bb374552f531f752654554b76b2d6` (v1.2.1) — the same commit the Windows build pins.
- sha256: `914fd98ac0a73d69ba1e70c9b1d66acb740eff90500dfde08b89a961b168a6a9`
- ONNX signature: inputs `input [seq_len, 576]`, `h [1,1,128]`, `c [1,1,128]`;
  outputs `speech_probs`, `hn`, `cn` — identical to what `app/audio/vad.py` feeds.

The Windows build keeps loading it from the installed faster-whisper package
(`faster_whisper.utils.get_assets_path()`); macOS has no faster-whisper installed,
so it loads this copy instead. Both paths are resolved by `app/audio/vad_asset.py`.
