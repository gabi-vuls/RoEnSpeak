# RoEnSpeak

Speak Romanian, type English. RoEnSpeak is a small Windows tray app for dictation. Press a hotkey (Alt+Q by
default) and speak Romanian. After each phrase, the English translation is typed into whatever window has the
keyboard focus: a browser, a chat app, a terminal, the Start menu search box, and so on. Press the hotkey again to
stop. It can also type plain Romanian instead of translating.

How it works:

```
microphone (16 kHz) -> Silero voice detection (splits speech into phrases) -> faster-whisper large-v3 on your
NVIDIA GPU (Romanian -> English) -> typed into the focused window
```

**Privacy.** Everything runs locally on your PC. Your audio and text never leave the machine. The app goes online
only once, to download the speech model from Hugging Face the first time it runs (or when you pick a different
model).

---

## Contents

- [Requirements](#requirements)
- [Install](#install)
- [Using RoEnSpeak](#using-roenspeak)
- [Settings](#settings)
- [Changing the hotkey](#changing-the-hotkey)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Uninstall](#uninstall)
- [For developers](#for-developers)
- [License and credits](#license-and-credits)

---

## Requirements

| What | Details |
|---|---|
| Operating system | Windows 10 or 11, 64-bit |
| Graphics card | An **NVIDIA** GPU with driver **528.33 or newer**. The installer already includes the CUDA libraries it needs (cuBLAS, cuDNN), so you don't have to install CUDA or anything else. Just keep the NVIDIA driver up to date. AMD and Intel GPUs are not supported. |
| Video memory (VRAM) | See the table below. The default (large-v3, float16) needs about 4.7 GB. |
| Disk space | About 2 GB for the app itself, which is mostly the bundled NVIDIA libraries ([BUILD.md](BUILD.md) has the exact installer size), plus the model download: about 3.1 GB for large-v3. |
| Microphone | Any Windows recording device |
| Internet | Only once, for the model download |

Approximate VRAM needed (including CUDA overhead) and download size for each model:

| Model | float16 | int8_float16 | Download | Can translate? |
|---|---|---|---|---|
| large-v3 (best quality, default) | ~4.7 GB | ~3.1 GB | ~3.1 GB | Yes |
| large-v2 | ~4.7 GB | ~3.1 GB | ~3.1 GB | Yes |
| medium (faster, lower quality) | ~2.6 GB | ~1.8 GB | ~1.5 GB | Yes |
| large-v3-turbo | ~2.6 GB | ~1.8 GB | ~1.6 GB | No, Romanian transcription only |

A 6 GB card runs large-v3 with float16. On a 4 GB card, choose int8_float16 (or medium).

---

## Install

1. Download `RoEnSpeak-Setup.exe` and run it.
2. **Windows SmartScreen may show a warning.** The installer is not code-signed, so Windows doesn't recognize the
   publisher. Click **More info**, then **Run anyway**.
3. The installer is per-user and does not need administrator rights. It creates a Start menu shortcut and, if you
   want, a desktop shortcut.
4. Start RoEnSpeak. It goes straight to the system tray (the notification area next to the clock), so no window
   opens. If you don't see the icon, click the **^** arrow to show the hidden icons.
5. **First run: model download.** A **Download speech model** dialog appears. Click **Download**. The dialog
   shows progress, speed and time left. Things to know:
   - The download is about 3.1 GB for large-v3. Files are saved to `%LOCALAPPDATA%\RoEnSpeak\models`.
   - The download is resumable. You can choose **Cancel download**, **Hide** or **Later**, and if your connection
     drops, click **Retry**. It continues where it stopped.
   - If you're offline, the dialog says so ("No internet connection…"). Connect and click Retry.
6. **First run: GPU check.** RoEnSpeak checks your NVIDIA driver and how much video memory you have. If the
   driver is missing or too old, you'll see an **NVIDIA GPU required** dialog (see
   [Troubleshooting](#troubleshooting)). If your GPU has less video memory than the selected model needs, a
   notification ("Not enough GPU memory?") suggests switching the compute type to int8_float16 or choosing a
   smaller model.
7. Once the model has loaded, the tray icon turns green and you can dictate.

---

## Using RoEnSpeak

### Dictating

1. Click into the text field where you want the text to go.
2. Press **Alt+Q**. You'll hear a start sound, and a small red **● REC** badge appears at the top of the screen.
3. Speak Romanian in normal phrases. When you pause for about half a second, that phrase is translated and the
   English text is typed at the cursor, usually within a second or two.
4. Keep going, phrase after phrase. Phrases are separated by spaces automatically.
5. Press **Alt+Q** again to stop. Whatever you said last is still translated and typed, so you don't need to wait
   for it before pressing the hotkey.

Output is **append-only**. Each phrase is typed once and never corrected or deleted afterwards. If a phrase
comes out wrong, fix it by hand.

You can switch to another window between phrases. Each phrase goes to whichever window has focus at the moment
it is typed.

If you turn the main ON/OFF switch off in the middle of a dictation, the dictation stops and any speech that
hasn't been processed yet is **discarded**. Pressing the hotkey to stop does not lose anything.

### Tray icon

The microphone icon in the tray changes color to show the state:

| Color | Meaning |
|---|---|
| Grey | Off (disarmed): the hotkey is not registered |
| Blue | Downloading or loading the model |
| Green | Ready: press the hotkey to dictate |
| Red | Listening |
| Amber | Processing: finishing the last phrase after you stopped |
| Dark red with "!" | Error: open the window to see what's wrong |

Hover over the icon to see the state and hotkey. Click it to open the main window. Right-click it for the menu:
**Open RoEnSpeak**, **Hotkey armed (Alt+Q)** (on/off), and **Quit**.

Closing the main window only hides it. RoEnSpeak keeps running in the tray, and the hotkey only works while it's
running. Use **Quit** from the tray menu to exit completely. This also frees the GPU memory. If you start
RoEnSpeak again while it's already running, the existing window just comes to the front.

### Main window

- **Dictation hotkey ON/OFF switch.** ON registers the global hotkey. OFF unregisters it completely, so Alt+Q goes
  back to other apps, and it stops any dictation in progress. The setting is remembered across restarts.
- **Hotkey** field and **Reset to default** (Alt+Q). See [Changing the hotkey](#changing-the-hotkey).
- **Status line.** Shows *Idle - hotkey is off*, *Downloading model NN%…*, *Loading model…*,
  *Ready - press Alt+Q to dictate*, *Listening - press Alt+Q to stop*, *Processing…*, or *Error: …*. Below it is
  the GPU name, model and compute type (for example `NVIDIA GeForce RTX 4060 Ti · large-v3 · float16`) and
  **Last latency**: the time from the end of your last phrase to the text being typed.
- **Settings** in three tabs (General, Model, Advanced). Settings changes take effect when you click **Apply**.
  **Revert** throws away unsaved changes.
- **Open log folder** button.
- The version number is shown in the top-right corner.

### Translate or transcribe

**Settings → General → Mode** has two choices:

- **Translate Romanian → English** (default): speak Romanian, English is typed.
- **Romanian transcription only**: speak Romanian, Romanian is typed (with diacritics). This mode can also use
  large-v3-turbo.

---

## Settings

Settings are saved to `%APPDATA%\RoEnSpeak\config.json`. Labels below are exactly as they appear in the window.

### General tab

| Setting | Default | What it does |
|---|---|---|
| Mode | Translate Romanian → English | Translate to English, or type Romanian ("Romanian transcription only"). |
| Microphone | System default | Which recording device to use. **↻** refreshes the list. A device that isn't plugged in is shown as "(not connected)". |
| Typing method | Type keystrokes (SendInput) | How text gets into the window. **Type keystrokes (SendInput)** types each character and leaves your clipboard alone; it works almost everywhere. **Paste via clipboard** puts the text on the clipboard, presses Ctrl+V, then restores what was on your clipboard before. It is faster for long text and helps in some apps that don't accept typed Unicode characters. |
| Show "● REC" overlay while listening | On | A small always-on-top badge while listening. It never takes focus and clicks pass through it. |
| Play start/stop sounds | On | A short beep when dictation starts and stops. |
| Start with Windows | Off | Starts RoEnSpeak in the tray when you sign in to Windows. |

### Model tab

| Setting | Default | What it does |
|---|---|---|
| Model | large-v3 (best quality) | Speech model. Translation mode offers large-v3, large-v2 and medium. Transcription mode also offers large-v3-turbo. The hint underneath shows the VRAM it needs and the download size. Picking a model that isn't downloaded yet opens the download dialog. |
| Compute type | float16 (best, more VRAM) | **int8_float16 (less VRAM)** uses about a third less video memory, with a small quality cost. Ignored in CPU mode. |
| Beam size | 5 | 1–10. 5 gives the best quality. 1 is fastest. |
| Unload when idle | Off (keep model loaded) | 0–1440 minutes. After this many idle minutes the model is unloaded to free GPU memory, and the status shows *Ready (model unloaded)*. The next hotkey press starts listening right away and reloads the model in the background, so the first phrase takes a few seconds longer. |
| CPU mode (slow) | Off | Runs the model on the processor instead of an NVIDIA GPU. Expect several seconds per phrase. Use it only on a PC without a supported NVIDIA GPU. It is **never** turned on automatically: if the GPU doesn't work, RoEnSpeak shows an error instead of quietly switching to the CPU. |

### Advanced tab

| Setting | Default | What it does |
|---|---|---|
| Silence ends phrase | 500 ms | 100–5000 ms. How long a pause has to be before a phrase is sent for translation. Shorter means faster output but more fragmented sentences. |
| Max phrase length | 15.0 s | 2–30 s. Continuous speech without a pause is cut into a separate phrase after this long. |
| Speech threshold | 0.50 | 0.05–0.95. Voice detection sensitivity: lower is more sensitive (picks up quieter speech, but also more noise). |
| Hallucination blocklist | built-in list | Speech models sometimes "hear" stock phrases in near-silence (for example "Thank you for watching", "Subtitles by…"). Text that matches a line here is never typed. Put one phrase per line, and end a line with `*` to match anything that starts with it. **Defaults** restores the built-in list. |

A few expert values exist only in `config.json`: the remaining voice-detection parameters and the hallucination
filter thresholds (`no_speech_prob_threshold`, `avg_logprob_threshold`, `compression_ratio_threshold`). Quit
RoEnSpeak before you edit the file. If a value is invalid, it is reset to its default and you'll see a
"Some settings were reset" notification. If the file can't be read at all, it is renamed to
`config.json.bad-<date>` and defaults are used.

### File locations

| What | Where |
|---|---|
| Settings | `%APPDATA%\RoEnSpeak\config.json` |
| Downloaded models | `%LOCALAPPDATA%\RoEnSpeak\models\<model name>\` |
| Logs | `%LOCALAPPDATA%\RoEnSpeak\logs\roenspeak.log` (rotating, 5 × 2 MB). The **Open log folder** button opens this folder. |

You can paste these paths into the Explorer address bar.

---

## Changing the hotkey

1. Click the **Hotkey** field in the main window.
2. Press the new combination, for example Ctrl+Alt+D. Press **Esc** to cancel.
3. The new hotkey takes effect immediately. You don't need to click Apply or restart.

Rules:

- The combination needs at least one modifier (Ctrl, Alt, Shift or Win) plus exactly one other key.
- Shift as the **only** modifier is not allowed with a key that types something (letters, digits, punctuation,
  Space), because you would no longer be able to type that character. Shift+F9 is fine, but Shift+A is not.
- Supported keys: A–Z, 0–9, F1–F24, Space, Enter, Tab, Esc, Backspace, Insert, Delete, Home, End, PageUp,
  PageDown, arrow keys, Pause, Numpad0–Numpad9, and the punctuation keys `; = , - . / ` [ \ ] '`.
- If another app (or Windows itself) already uses the combination, you'll see "… is already in use by another
  application or by Windows. Choose a different combination." Your previous hotkey stays active. Many Win+key
  combinations are reserved by Windows.
- If the ON/OFF switch is off, the new hotkey is saved and is registered the next time you turn the switch on.

**Reset to default** sets Alt+Q again.

---

## Known limitations

- **Administrator windows.** Windows does not let a normal app type into a window that runs as administrator
  (this protection is called UIPI). This also applies to admin terminals, Task Manager, and some installers.
  RoEnSpeak detects this and shows a tray notification, "Text not typed: administrator window", that contains
  the text, so nothing is lost silently. To fix it, either run the target app without admin rights, or run
  RoEnSpeak as administrator. Switching to clipboard mode does not help here.
- **Apps that block simulated keyboard input.** Some programs ignore keystrokes that don't come from a physical
  keyboard. Examples are games with anti-cheat, some remote desktop and virtual machine consoles, the Windows
  secure desktop (UAC prompts, the Ctrl+Alt+Del screen, the sign-in screen), and password fields in some apps.
  Try **Typing method → Paste via clipboard**. Some of these places will still refuse the text.
- **Alt+Q overrides Microsoft Office's search shortcut.** In Microsoft 365 apps, Alt+Q normally jumps to the
  "Tell me / Search" box. While RoEnSpeak is running with the switch ON, Alt+Q starts dictation instead. If you
  need the Office shortcut, change RoEnSpeak's hotkey or turn the switch OFF.
- **NVIDIA GPUs only.** The fast path needs CUDA. On other hardware, only the slow CPU mode works.
- **No leading space at the start of a dictation.** The first phrase of each new dictation is typed without a
  space in front. Type the space yourself if you're continuing an existing sentence. Later phrases in the same
  dictation are separated automatically.
- **Long speech without pauses is cut every ~15 seconds** (see *Max phrase length*). The cut can land
  mid-sentence or even mid-word, which can hurt translation quality around it. Short natural pauses work best.
- **Typing waits while you hold modifier keys.** If you're holding Ctrl, Shift, Alt or Win when a phrase is ready,
  RoEnSpeak waits until you release them, because typing with them held would trigger shortcuts. If they're still
  held after 5 seconds, that phrase is skipped and shown in a tray notification ("Text not typed: keys held down").
- **large-v3-turbo is offered only for Romanian transcription.** It was not trained to translate. The distil-*
  models only handle English and are not offered.
- **Append-only output.** Text that has already been typed is never revised.

---

## Troubleshooting

**"NVIDIA GPU required" dialog.**
Either RoEnSpeak found no NVIDIA GPU, the driver is older than 528.33, or the CUDA libraries didn't load. The
dialog shows the exact problem. Install or update the driver from nvidia.com/drivers (or with the NVIDIA app or
GeForce Experience), then restart RoEnSpeak. If the PC has no NVIDIA GPU, you can enable **Settings → Model →
CPU mode (slow)**.

**The model download fails.**
"No internet connection" means Hugging Face couldn't be reached. Check your connection, a VPN or firewall, then
click **Retry**. The download resumes, and partial files are kept. If you closed the dialog with **Later**, turn
the switch OFF and ON again to bring it back.

**"Hotkey unavailable" / "… is already in use".**
Another program owns that combination. Choose a different one (see [Changing the hotkey](#changing-the-hotkey)).

**Nothing gets typed.**
- Check the tray icon. It should be red while you're speaking. If it stays green, the hotkey didn't reach
  RoEnSpeak. Check that the switch is ON and that the hotkey shown is the one you're pressing.
- Make sure the text cursor is actually in a text field and that window has focus.
- Is the target an administrator window? Look for a "Text not typed" tray notification
  ([Known limitations](#known-limitations)).
- Try **Typing method → Paste via clipboard**.
- Check **Microphone**: pick your mic explicitly instead of "System default", and make sure Windows allows
  microphone access (Windows Settings → Privacy → Microphone).
- If you're speaking very quietly, lower **Speech threshold** (for example to 0.35).

**It's slow.**
Look at **Last latency** in the main window. To speed things up, try these in order:
1. **Beam size** 1.
2. **Compute type** int8_float16.
3. **Model** medium.

Also check that the status line shows your GPU name and not "CPU mode (slow)". On a GPU that's too small, the
model may be swapping memory or failing to load. Close other GPU-heavy programs (games, video editing).

**Strange phrases appear when you didn't say anything.**
Add them to the **Hallucination blocklist** (Advanced tab), or raise **Speech threshold** a little.

**Logs.**
Click **Open log folder**, or go to `%LOCALAPPDATA%\RoEnSpeak\logs`. `roenspeak.log` records GPU, driver and CUDA
library versions (`cuda_probe`, `engine_loaded` lines) and per-phrase timings (`segment_latency` lines). Include
it when you report a problem.

**Diagnostic test without the microphone (CLI test mode).**
RoEnSpeak includes a command-line tool that runs a `.wav` file through the same speech pipeline and prints the
result and timings. It doesn't type anything. Run it from a Command Prompt in the install folder:

```
RoEnSpeak-cli.exe path\to\file.wav
```

It uses your saved settings unless you override them:

| Option | Meaning |
|---|---|
| `--model NAME` | `large-v3`, `large-v2`, `medium` or `large-v3-turbo` |
| `--compute-type T` | `float16` or `int8_float16` |
| `--beam N` | Beam size 1–10 |
| `--task T` | `translate` (Romanian → English) or `transcribe` (Romanian) |
| `--cpu` | Run on the CPU (slow); prints `WARNING: CPU mode (slow)` |
| `--download` | Download the model first if it's missing |
| `--json` | Print one JSON object instead of text lines |
| `--realtime` | Feed the audio at normal speed instead of as fast as possible |

The output has an `ENGINE …` line (device, compute type, GPU, load time), one `SEGMENT …` line per phrase (with
timings and the text), and a `SUMMARY …` line. Exit codes:

| Code | Meaning |
|---|---|
| 0 | OK |
| 1 | Other error |
| 2 | CUDA unavailable (and `--cpu` not given) |
| 3 | Model not downloaded (and `--download` not given) |
| 4 | Bad input: missing or unreadable .wav, or invalid options |

---

## Uninstall

Use **Settings → Apps → Installed apps → RoEnSpeak → Uninstall** in Windows, or the uninstaller in the Start
menu. The uninstaller asks whether to also delete the downloaded models (several GB in
`%LOCALAPPDATA%\RoEnSpeak\models`). Keep them if you plan to reinstall. To remove every trace, also delete
`%APPDATA%\RoEnSpeak` (settings) and `%LOCALAPPDATA%\RoEnSpeak` (logs and models) if they're still there.

---

## For developers

- **Python 3.12** on Windows x64 (the code requires 3.11+). UI is PySide6 (Essentials). All Win32 calls go
  through ctypes.
- Speech engine: [SYSTRAN faster-whisper](https://github.com/SYSTRAN/faster-whisper), installed from a local
  clone and pinned to commit `9e472ae9de8bb374552f531f752654554b76b2d6` (v1.2.1). See `requirements.in`.
  `requirements.txt` is the pinned lock, which includes ctranslate2 4.8.3 and the `nvidia-cublas-cu12` and
  `nvidia-cudnn-cu12` wheels whose DLLs get bundled.
- [ARCHITECTURE.md](ARCHITECTURE.md) covers threads, queues, the state machine, the CUDA DLL policy and the latency
  budget. `app/contracts.py` is the single source of truth for interfaces, defaults and paths.
- [BUILD.md](BUILD.md) has the exact steps from a clean checkout to `RoEnSpeak-Setup.exe` (PyInstaller onedir and
  Inno Setup, plus the optional signing hook).
- Run from source: `python -m app.main` (tray app) and `python -m app.cli file.wav` (CLI test mode).
- Tests. The repo lives in WSL. `scripts/win.sh` syncs it to `C:\dev\ro-en-speak` and runs the Windows venv's
  Python there:

  ```
  scripts/win.sh -m pytest                 # default: unit tests, hardware mocked
  scripts/win.sh -m pytest -m gpu          # real CUDA + downloaded model (CLI end-to-end)
  scripts/win.sh -m pytest -m desktop      # types real keystrokes into a test window
  scripts/win.sh -m pytest -m mic          # needs a human speaking into a microphone
  ```

  Set `ROENSPEAK_CONFIG_DIR`, `ROENSPEAK_DATA_DIR` and `ROENSPEAK_MODELS_DIR` to isolate config, logs and models
  from your real installation.

---

## License and credits

**License:** TBD by the owner.

RoEnSpeak is built on:

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) by SYSTRAN, MIT License, and
  [CTranslate2](https://github.com/OpenNMT/CTranslate2), MIT License.
- [Whisper](https://github.com/openai/whisper) models by OpenAI, MIT License, in the CTranslate2 conversions
  published on Hugging Face (`Systran/faster-whisper-*`, and `mobiuslabsgmbh/faster-whisper-large-v3-turbo`).
  Models are downloaded at first run and are not included in the installer.
- [Silero VAD](https://github.com/snakers4/silero-vad), MIT License, the copy bundled with faster-whisper.
- [PySide6 / Qt for Python](https://www.qt.io/qt-for-python), LGPL v3.
- NVIDIA CUDA libraries (cuBLAS, cuDNN), redistributed under the NVIDIA software license terms that come with
  those packages.
- Other Python dependencies (onnxruntime, sounddevice/PortAudio, soxr, huggingface_hub, nvidia-ml-py) are used
  under their own licenses.
