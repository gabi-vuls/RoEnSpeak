# RoEnSpeak

Speak Romanian, type English. RoEnSpeak is a small dictation app that lives in the Windows system tray or the
macOS menu bar. Press a hotkey (Alt+Q by default, which is Option+Q on a Mac) and speak Romanian. After each
phrase, the English translation is typed into whatever window has the keyboard focus: a browser, a chat app, a
terminal, the Start menu or Spotlight search box, and so on. Press the hotkey again to stop. It can also type
plain Romanian instead of translating.

How it works:

```
microphone (16 kHz) -> Silero voice detection (splits speech into phrases) -> large-v3 (Romanian -> English)
  on Windows: faster-whisper on your NVIDIA GPU
  on macOS:   whisper.cpp on your Mac's Metal GPU
-> typed into the focused window
```

Everything above the engine is the same code on both platforms, and one `config.json` is valid on both.

**Privacy.** Everything runs locally on your computer. Your audio and text never leave the machine. The app goes
online only once, to download the speech model from Hugging Face the first time it runs (or when you pick a
different model).

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

RoEnSpeak runs on Windows and on Apple Silicon Macs. Where the two differ, the section has a **Windows** and a
**macOS** part - Requirements, Install, Troubleshooting and Uninstall all do. Everything else is the same on
both.

---

## Requirements

### Windows

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

### macOS

| What | Details |
|---|---|
| Operating system | macOS 14 (Sonoma) or newer. |
| Mac model | **Apple Silicon only** - any M1, M2, M3 or M4 Mac. Intel Macs are not supported: the app is built for arm64 and the speech engine runs on Apple's Metal GPU. |
| Memory | See the table below. There is no separate graphics card: Apple Silicon shares one pool of memory between the CPU and the GPU, so the "memory" column is taken out of your Mac's total RAM. The default (large-v3, float16) needs about 4 GB of it, so 16 GB of RAM is comfortable and 8 GB is tight. |
| Disk space | About 185 MB for the app, plus the model download (about 3 GB for large-v3). |
| Microphone | Any input device macOS recognises, built-in included. |
| Internet | Only once, for the model download. |
| Permissions | **Two permission prompts you have to accept** - Microphone and Accessibility. See [Install](#install). |

Approximate memory needed and download size on macOS. The downloads are different files from the Windows ones
(single-file `ggml-*.bin` models), so the sizes differ slightly:

| Model | float16 | int8_float16 | Download (float16) | Download (int8_float16) | Can translate? |
|---|---|---|---|---|---|
| large-v3 (best quality, default) | ~4.0 GB | ~1.4 GB | 3.0 GB | 1.1 GB | Yes |
| large-v2 | ~4.0 GB | ~1.4 GB | 3.0 GB | 1.1 GB | Yes |
| medium (faster, lower quality) | ~2.0 GB | ~0.7 GB | 1.5 GB | 539 MB | Yes |
| large-v3-turbo | ~2.0 GB | ~0.7 GB | 1.6 GB | 574 MB | No, Romanian transcription only |

On an 8 GB Mac, choose **int8_float16** (or medium): large-v3 at float16 needs about 4 GB of working set next to
the rest of the system. On 16 GB or more, large-v3 at float16 is the right default. For reference, on an M1 Max
`medium` translates a 3.5 second phrase in about 380 ms.

---

## Install

### Windows

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

### macOS

1. Download `RoEnSpeak-<version>.dmg` and double-click it.
2. Drag **RoEnSpeak** onto the **Applications** folder shown next to it, then eject the disk image.
3. **The first launch needs a right-click.** The app is not signed with a paid Apple developer
   certificate, so double-clicking it shows *"RoEnSpeak" cannot be opened because the developer cannot be
   verified*. **Right-click (or Control-click) RoEnSpeak in Applications → Open → Open.** You only have to do
   this once. (Equivalent, from Terminal:
   `xattr -dr com.apple.quarantine /Applications/RoEnSpeak.app`.)
4. RoEnSpeak has **no Dock icon on purpose**. It appears as a small microphone in the **menu bar** at the top
   right of the screen, next to the clock, Wi-Fi and battery. If you cannot see it, your menu bar may be full -
   quit an app or use a menu-bar manager to reveal it.
5. **Two permission prompts - this is the part with no Windows equivalent, and it is the single most common
   thing to get stuck on.** macOS will not let any app listen to your microphone or type into other apps
   without your explicit permission:

   | Prompt | When it appears | What happens if you say no |
   |---|---|---|
   | **"RoEnSpeak would like to access the microphone."** | The first time you start a dictation. | Nothing is heard. The status line shows a microphone error and no phrase is ever produced. |
   | **"RoEnSpeak would like to control this computer using accessibility features."** | The first time RoEnSpeak has text to type. | Speech is recognised but **nothing is typed**. You get a notification: *"RoEnSpeak needs Accessibility permission to type. Open System Settings > Privacy & Security > Accessibility and switch RoEnSpeak on, then try again."* |

   Click **Allow** / **Open System Settings** on both. For Accessibility, macOS opens
   **System Settings → Privacy & Security → Accessibility**, where you switch **RoEnSpeak** on (you may need to
   unlock the padlock with Touch ID or your password). If you dismissed a prompt, you can grant them later in
   the same place, or in **Privacy & Security → Microphone**, and RoEnSpeak's own buttons take you straight
   there.

   > After installing an **update**, macOS may ask for Accessibility again. That is normal: macOS ties the
   > permission to the app's exact code signature, and a new build has a new one. Switch RoEnSpeak off and on
   > again in the Accessibility list if the toggle looks enabled but typing does not work.

6. **First run: model download.** A **Download speech model** dialog appears; click **Download**. About 3 GB for
   large-v3, saved to `~/Library/Application Support/RoEnSpeak/models`. It is resumable and shows progress,
   speed and time left, exactly as on Windows.
7. Once the model has loaded, the menu-bar icon turns green and you can dictate.

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

### Tray icon (menu-bar icon on macOS)

The microphone icon in the Windows tray, or in the macOS menu bar, changes color to show the state:

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

On macOS the icon is a template image, so it follows the menu bar's light or dark appearance, and the state
colour shows as a small dot. A left click opens the menu (macOS menu-bar items have no separate right-click
menu), where **Open RoEnSpeak** brings up the window.

Closing the main window only hides it. RoEnSpeak keeps running in the tray (menu bar), and the hotkey only works
while it's running. Use **Quit** from the menu to exit completely. This also frees the GPU memory. If you start
RoEnSpeak again while it's already running, the existing window just comes to the front. On macOS there is no
Dock icon and no entry in ⌘-Tab, so the menu bar is the way back to the window.

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
| Start with Windows<br>(**Open at login** on macOS) | Off | Starts RoEnSpeak minimised when you sign in. On macOS this writes `~/Library/LaunchAgents/com.roenspeak.RoEnSpeak.plist`, and removing the login item deletes it again. |

### Model tab

| Setting | Default | What it does |
|---|---|---|
| Model | large-v3 (best quality) | Speech model. Translation mode offers large-v3, large-v2 and medium. Transcription mode also offers large-v3-turbo. The hint underneath shows the VRAM it needs and the download size. Picking a model that isn't downloaded yet opens the download dialog. |
| Compute type | float16 (best, more VRAM) | **int8_float16 (less VRAM)** uses about a third less video memory, with a small quality cost. Ignored in CPU mode. |
| Beam size | 5 | 1–10. 5 gives the best quality. 1 is fastest. |
| Unload when idle | Off (keep model loaded) | 0–1440 minutes. After this many idle minutes the model is unloaded to free GPU memory, and the status shows *Ready (model unloaded)*. The next hotkey press starts listening right away and reloads the model in the background, so the first phrase takes a few seconds longer. |
| CPU mode (slow) | Off | Runs the model on the processor instead of the GPU. Expect several seconds per phrase. On Windows, use it only on a PC without a supported NVIDIA GPU; on macOS, only if Metal fails (whisper.cpp on the CPU is roughly 3x slower than on Metal). It is **never** turned on automatically: if the GPU doesn't work, RoEnSpeak shows an error instead of quietly switching to the CPU. |

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

**Windows**

| What | Where |
|---|---|
| Settings | `%APPDATA%\RoEnSpeak\config.json` |
| Downloaded models | `%LOCALAPPDATA%\RoEnSpeak\models\<model name>\` |
| Logs | `%LOCALAPPDATA%\RoEnSpeak\logs\roenspeak.log` (rotating, 5 × 2 MB). The **Open log folder** button opens this folder. |

You can paste these paths into the Explorer address bar.

**macOS**

| What | Where |
|---|---|
| Settings | `~/Library/Application Support/RoEnSpeak/config.json` |
| Downloaded models | `~/Library/Application Support/RoEnSpeak/models/<model name>/` |
| Logs | `~/Library/Logs/RoEnSpeak/roenspeak.log` (rotating, 5 × 2 MB). The **Open log folder** button opens this folder in Finder. |
| Login item | `~/Library/LaunchAgents/com.roenspeak.RoEnSpeak.plist` (only when **Open at login** is on) |

`~/Library` is hidden in Finder. Use **Go → Go to Folder…** (⇧⌘G) and paste the path, or hold ⌥ while opening
Finder's **Go** menu to reveal **Library**. The settings file is the same format on both platforms, so you can
copy a `config.json` from a PC to a Mac.

---

## Changing the hotkey

1. Click the **Hotkey** field in the main window.
2. Press the new combination, for example Ctrl+Alt+D. Press **Esc** to cancel.
3. The new hotkey takes effect immediately. You don't need to click Apply or restart.

Rules:

- The combination needs at least one modifier plus exactly one other key. On Windows the modifiers are Ctrl,
  Alt, Shift and Win; on macOS they are Control, Option, Shift and Command.
- Shift as the **only** modifier is not allowed with a key that types something (letters, digits, punctuation,
  Space), because you would no longer be able to type that character. Shift+F9 is fine, but Shift+A is not.
- Supported keys: A–Z, 0–9, F1–F24, Space, Enter, Tab, Esc, Backspace, Insert, Delete, Home, End, PageUp,
  PageDown, arrow keys, Pause, Numpad0–Numpad9, and the punctuation keys `; = , - . / ` [ \ ] '`.
- If another app (or the operating system) already uses the combination, you'll see "… is already in use by
  another application or by Windows. Choose a different combination." Your previous hotkey stays active. Many
  Win+key combinations are reserved by Windows.
- If the ON/OFF switch is off, the new hotkey is saved and is registered the next time you turn the switch on.

**Reset to default** sets Alt+Q again.

### On a Mac

The names in the hotkey field are the same words as on Windows, so one `config.json` works on both machines.
This is how they map to the keys printed on an Apple keyboard:

| Shown in RoEnSpeak | Key on your Mac | Symbol |
|---|---|---|
| `Ctrl` | Control | ⌃ |
| `Alt` | Option (also labelled Alt) | ⌥ |
| `Shift` | Shift | ⇧ |
| `Cmd` | Command | ⌘ |
| `Win` | read as Command, so a hotkey saved on a PC still works | ⌘ |

The default **Alt+Q** therefore means **⌥Q** (Option+Q). Combinations are displayed in the order
Ctrl, Alt, Shift, Cmd — so ⌃⌥D shows as `Ctrl+Alt+D`.

**Combinations macOS keeps for itself.** Unlike Windows, macOS does not always say no: some combinations are
taken by the system before any app sees them, and RoEnSpeak simply never fires. Avoid:

- **⌘Space** and **⌃⌘Space** (Spotlight, emoji picker) and **⌥⌘Space** (Finder search).
- **⌘Tab**, **⌘`**, **⌘Q**, **⌘W**, **⌘H**, **⌘M**, **⌘,** and the other standard application shortcuts -
  these go to the focused app, not to RoEnSpeak.
- **⌃↑ / ⌃↓ / ⌃← / ⌃→** (Mission Control and Spaces) and **F11** (Show Desktop).
- **⇧⌘3 / ⇧⌘4 / ⇧⌘5** (screenshots) and **⌃⌘Q** (lock screen).
- The media and brightness keys (F1-F12 with the `fn` behaviour your Mac is set to). If F-key hotkeys do not
  fire, turn on **System Settings → Keyboard → Keyboard Shortcuts → Function Keys → Use F1, F2, etc. as
  standard function keys**, or hold `fn`.

If a combination is already registered by another running app, you get the normal "already in use" message and
your previous hotkey stays active. If instead the hotkey is accepted but nothing ever happens, macOS is
swallowing it - pick a different one. **⌥** combinations with a letter (like the default ⌥Q) are the most
reliable choice.

> ⌥ plus a letter normally types a special character (⌥Q is `œ`). While RoEnSpeak holds the hotkey, that
> character is no longer typed by that combination. Change the hotkey if you need it.

---

## Known limitations

Items marked **(Windows)** or **(macOS)** apply to that platform only; the rest apply to both.

- **(macOS) Password fields and other "secure input" fields.** macOS turns on *secure event input* whenever a
  password field has the keyboard focus, and that blocks every synthetic keystroke - there is no way around it,
  not even clipboard mode. RoEnSpeak detects it and shows *"Can't type into a password field (macOS blocks
  synthetic keystrokes there). Click into a normal text field and dictate again."*, with the text included so
  nothing is lost. This is the macOS counterpart of the Windows administrator-window limitation below: some
  apps (a few terminals, password managers and remote-desktop clients) leave secure input on for longer than
  you would expect, so if typing stops working everywhere, close the app that last had a password field.
- **(macOS) Apps that ignore synthetic keystrokes.** Accessibility permission covers almost everything, but
  some apps refuse events that did not come from real hardware: games and apps with anti-cheat, some
  virtual-machine and remote-desktop consoles (Parallels, VMware, Citrix, VNC clients), Java-based apps with
  their own input handling, and the macOS login window / FileVault unlock screen, which no app can reach. Try
  **Typing method → Paste via clipboard** (which uses ⌘V instead of per-character events) - some of these
  accept that when they reject typing.
- **(macOS) Apple Silicon only.** The app is arm64 and the engine runs on Metal; it does not install on Intel
  Macs.
- **(macOS) No Dock icon.** RoEnSpeak is a menu-bar app, so it is not in the Dock, not in ⌘-Tab and not in
  Mission Control. Use the menu-bar icon.
- **(macOS) Accessibility has to be re-granted after an update.** macOS ties the permission to the app's code
  signature, and each build has a different one. Switch RoEnSpeak off and on again in
  **Privacy & Security → Accessibility**.
- **(Windows) Administrator windows.** Windows does not let a normal app type into a window that runs as administrator
  (this protection is called UIPI). This also applies to admin terminals, Task Manager, and some installers.
  RoEnSpeak detects this and shows a tray notification, "Text not typed: administrator window", that contains
  the text, so nothing is lost silently. To fix it, either run the target app without admin rights, or run
  RoEnSpeak as administrator. Switching to clipboard mode does not help here.
- **(Windows) Apps that block simulated keyboard input.** Some programs ignore keystrokes that don't come from a physical
  keyboard. Examples are games with anti-cheat, some remote desktop and virtual machine consoles, the Windows
  secure desktop (UAC prompts, the Ctrl+Alt+Del screen, the sign-in screen), and password fields in some apps.
  Try **Typing method → Paste via clipboard**. Some of these places will still refuse the text.
- **(Windows) Alt+Q overrides Microsoft Office's search shortcut.** In Microsoft 365 apps, Alt+Q normally jumps to the
  "Tell me / Search" box. While RoEnSpeak is running with the switch ON, Alt+Q starts dictation instead. If you
  need the Office shortcut, change RoEnSpeak's hotkey or turn the switch OFF.
- **(Windows) NVIDIA GPUs only.** The fast path needs CUDA. On other hardware, only the slow CPU mode works.
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

### macOS only

**Nothing is typed, but the status line looks fine, or you get "RoEnSpeak needs Accessibility permission to
type".**
This is the most common problem on a Mac, and it is always the Accessibility permission. Open
**System Settings → Privacy & Security → Accessibility** and make sure **RoEnSpeak** is listed *and* switched
on. If the switch already looks on but typing still fails, switch it **off and on again** - this happens after
every app update, because macOS ties the permission to the app's exact code signature. If RoEnSpeak is not in
the list at all, click **+**, then pick `/Applications/RoEnSpeak.app`.

**No speech is ever detected, or "microphone access denied".**
Open **System Settings → Privacy & Security → Microphone** and switch RoEnSpeak on. If you clicked "Don't
Allow" at the first prompt, macOS will not ask again - this is the only way to change it. Also check
**Settings → General → Microphone** in RoEnSpeak and pick your device explicitly instead of "System default".

**"RoEnSpeak" cannot be opened because the developer cannot be verified.**
The app is not signed with a paid Apple certificate. Right-click (Control-click) RoEnSpeak in Applications →
**Open** → **Open**. Once only. Or run `xattr -dr com.apple.quarantine /Applications/RoEnSpeak.app`.

**"RoEnSpeak is damaged and can't be opened."**
The download was incomplete or the bundle was modified after it was signed. Download the `.dmg` again; do not
edit anything inside `RoEnSpeak.app`.

**The hotkey does nothing.**
Either another app owns it, or macOS swallows it before any app sees it. See
[On a Mac](#on-a-mac) under Changing the hotkey for the combinations to avoid. Try the default ⌥Q first. Note
that the hotkey needs no extra permission (RoEnSpeak uses the system hotkey API, not keyboard monitoring), so
there is nothing to grant for it.

**I can't find the app - there's no Dock icon.**
By design: it is a menu-bar app. Look for the small microphone at the top right of the screen. If your menu bar
is full, the icon can be pushed off the edge; quit another menu-bar app or use a menu-bar manager.

**Typing stops working in one particular app.**
That app probably has secure input on (it had a password field focused) or it rejects synthetic events. See
[Known limitations](#known-limitations).

**Logs.**
Click **Open log folder**, or **Go → Go to Folder…** (⇧⌘G) and paste `~/Library/Logs/RoEnSpeak`.
`roenspeak.log` records the Mac model, macOS version and whisper.cpp/Metal status (`cuda_probe`,
`whispercpp_loaded`, `engine_loaded` lines) and per-phrase timings (`segment_latency` lines). Include it when
you report a problem.

### Windows only

**"NVIDIA GPU required" dialog.**
Either RoEnSpeak found no NVIDIA GPU, the driver is older than 528.33, or the CUDA libraries didn't load. The
dialog shows the exact problem. Install or update the driver from nvidia.com/drivers (or with the NVIDIA app or
GeForce Experience), then restart RoEnSpeak. If the PC has no NVIDIA GPU, you can enable **Settings → Model →
CPU mode (slow)**.

**The model download fails.**
"No internet connection" means Hugging Face couldn't be reached. Check your connection, a VPN or firewall, then
click **Retry**. The download resumes, and partial files are kept. If you closed the dialog with **Later**, turn
the switch OFF and ON again to bring it back.

### Both platforms

**"Hotkey unavailable" / "… is already in use".**
Another program owns that combination. Choose a different one (see [Changing the hotkey](#changing-the-hotkey)).

**Nothing gets typed.**
- Check the tray icon. It should be red while you're speaking. If it stays green, the hotkey didn't reach
  RoEnSpeak. Check that the switch is ON and that the hotkey shown is the one you're pressing.
- Make sure the text cursor is actually in a text field and that window has focus.
- Is the target an administrator window? Look for a "Text not typed" tray notification
  ([Known limitations](#known-limitations)).
- Try **Typing method → Paste via clipboard**.
- Check **Microphone**: pick your mic explicitly instead of "System default", and make sure the system allows
  microphone access (Windows Settings → Privacy → Microphone; macOS System Settings → Privacy & Security →
  Microphone).
- On a Mac, check **Privacy & Security → Accessibility** first - see [macOS only](#macos-only) above.
- If you're speaking very quietly, lower **Speech threshold** (for example to 0.35).

**It's slow.**
Look at **Last latency** in the main window. To speed things up, try these in order:
1. **Beam size** 1.
2. **Compute type** int8_float16.
3. **Model** medium.

Also check that the status line shows your GPU name and not "CPU mode (slow)". On a GPU that's too small, the
model may be swapping memory or failing to load. Close other GPU-heavy programs (games, video editing). On a
Mac, the very first phrase after installing can take an extra half second while macOS compiles and caches the
Metal shaders; it is fast from then on, including after a restart.

**Strange phrases appear when you didn't say anything.**
Add them to the **Hallucination blocklist** (Advanced tab), or raise **Speech threshold** a little.

**Logs.**
Click **Open log folder**, or go to `%LOCALAPPDATA%\RoEnSpeak\logs`. `roenspeak.log` records GPU, driver and CUDA
library versions (`cuda_probe`, `engine_loaded` lines) and per-phrase timings (`segment_latency` lines). Include
it when you report a problem.

**Diagnostic test without the microphone (CLI test mode).**
RoEnSpeak includes a command-line tool that runs a `.wav` file through the same speech pipeline and prints the
result and timings. It doesn't type anything.

On Windows, run it from a Command Prompt in the install folder:

```
RoEnSpeak-cli.exe path\to\file.wav
```

On macOS it lives inside the app bundle, so run it from Terminal:

```
/Applications/RoEnSpeak.app/Contents/MacOS/RoEnSpeak-cli path/to/file.wav
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
| 2 | GPU unavailable - CUDA on Windows, Metal on macOS (and `--cpu` not given) |
| 3 | Model not downloaded (and `--download` not given) |
| 4 | Bad input: missing or unreadable .wav, or invalid options |

---

## Uninstall

### Windows

Use **Settings → Apps → Installed apps → RoEnSpeak → Uninstall** in Windows, or the uninstaller in the Start
menu. The uninstaller asks whether to also delete the downloaded models (several GB in
`%LOCALAPPDATA%\RoEnSpeak\models`). Keep them if you plan to reinstall. To remove every trace, also delete
`%APPDATA%\RoEnSpeak` (settings) and `%LOCALAPPDATA%\RoEnSpeak` (logs and models) if they're still there.

### macOS

A disk image has no uninstaller, so **use the script** - dragging the app to the Trash leaves the login item,
the models (several GB) and the logs behind. From the repo, or from wherever you saved it:

```sh
bash installer/macos/uninstall.sh
```

It quits a running RoEnSpeak, removes the login item (only if it points at the app being removed), removes
`/Applications/RoEnSpeak.app`, and then **asks** whether to delete the downloaded models and the logs. The
default answer is **No**, so nothing is deleted unless you say yes, and your settings
(`~/Library/Application Support/RoEnSpeak/config.json`) are always kept so a reinstall remembers them. Useful
options: `--dry-run` (show the plan, change nothing), `--yes` (answer yes to everything, models included),
`--all` (also delete the settings). Running it twice is harmless.
[installer/macos/README.md](installer/macos/README.md) has the details.

By hand, if you prefer: drag `/Applications/RoEnSpeak.app` to the Trash, then delete
`~/Library/LaunchAgents/com.roenspeak.RoEnSpeak.plist`, `~/Library/Application Support/RoEnSpeak` and
`~/Library/Logs/RoEnSpeak`.

Either way, macOS keeps the Microphone and Accessibility entries in its own protected database, which no
uninstaller may touch. Remove them yourself in **System Settings → Privacy & Security → Accessibility** and
**→ Microphone** (select RoEnSpeak, click **-**). Leaving them is harmless.

---

## For developers

- **Python 3.12** (the code requires 3.11+): x64 on Windows, **arm64** on macOS. UI is PySide6 (Essentials) on
  both. Every platform call goes through ctypes - Win32 in `app/win32/`, Carbon/CoreGraphics/AppKit in
  `app/macos/` - and `app/core/platform.py` is the only place that chooses between them.
- Speech engines:
  - *Windows*: [SYSTRAN faster-whisper](https://github.com/SYSTRAN/faster-whisper), installed from a local clone
    and pinned to commit `9e472ae9de8bb374552f531f752654554b76b2d6` (v1.2.1). See `requirements.in`.
    `requirements.txt` is the pinned lock, which includes ctranslate2 4.8.3 and the `nvidia-cublas-cu12` and
    `nvidia-cudnn-cu12` wheels whose DLLs get bundled.
  - *macOS*: [whisper.cpp](https://github.com/ggml-org/whisper.cpp) pinned to tag **v1.9.5** (commit
    `d1be6fde11ac6e0407606b4e42fe72d34add8037`), built with Metal and embedded shaders by
    `native/whispercpp/build.sh` and loaded through a small C shim with ctypes. There is **no** faster-whisper,
    ctranslate2 or `nvidia-*` package on macOS. See `requirements-macos.in`; `requirements-macos.txt` is the
    pinned lock.
- [ARCHITECTURE.md](ARCHITECTURE.md) covers threads, queues, the state machine, the GPU library policy and the
  latency budget. `app/contracts.py` is the single source of truth for interfaces, defaults and paths, on both
  platforms.
- [BUILD.md](BUILD.md) has the exact steps from a clean checkout to `RoEnSpeak-Setup.exe` (PyInstaller onedir +
  Inno Setup) and to `RoEnSpeak-<version>.dmg` (PyInstaller `BUNDLE` + `hdiutil`), including signing,
  notarization and the bundle layout.
- Run from source, on either platform: `python -m app.main` (tray app) and `python -m app.cli file.wav` (CLI
  test mode).
- **Build.**

  ```
  powershell -NoProfile -ExecutionPolicy Bypass -File build\build.ps1     # Windows -> dist\RoEnSpeak-Setup.exe
  bash build/build_macos.sh                                              # macOS   -> dist/RoEnSpeak-<v>.dmg
  ```

- **Tests on macOS.** No indirection: the repo is the working copy and the venv is local.

  ```
  .venv/bin/python -m pytest                 # default: unit tests, hardware mocked
  .venv/bin/python -m pytest -m metal        # real Metal + a downloaded ggml model (CLI end-to-end)
  .venv/bin/python -m pytest -m desktop      # types real keystrokes (needs Accessibility permission)
  .venv/bin/python -m pytest -m mic          # needs a human speaking into a microphone
  ```

- **Tests on Windows.** The repo lives in WSL. `scripts/win.sh` syncs it to `C:\dev\ro-en-speak` and runs the
  Windows venv's Python there:

  ```
  scripts/win.sh -m pytest                 # default: unit tests, hardware mocked
  scripts/win.sh -m pytest -m gpu          # real CUDA + downloaded model (CLI end-to-end)
  scripts/win.sh -m pytest -m desktop      # types real keystrokes into a test window
  scripts/win.sh -m pytest -m mic          # needs a human speaking into a microphone
  ```

  Set `ROENSPEAK_CONFIG_DIR`, `ROENSPEAK_DATA_DIR` and `ROENSPEAK_MODELS_DIR` to isolate config, logs and models
  from your real installation. This works the same on both platforms.

---

## License and credits

**License:** TBD by the owner.

RoEnSpeak is built on:

- *(Windows)* [faster-whisper](https://github.com/SYSTRAN/faster-whisper) by SYSTRAN, MIT License, and
  [CTranslate2](https://github.com/OpenNMT/CTranslate2), MIT License.
- *(macOS)* [whisper.cpp](https://github.com/ggml-org/whisper.cpp) and [ggml](https://github.com/ggml-org/ggml)
  by Georgi Gerganov and contributors, MIT License, built from the pinned v1.9.5 source with the Metal backend.
- [Whisper](https://github.com/openai/whisper) models by OpenAI, MIT License - as CTranslate2 conversions on
  Hugging Face (`Systran/faster-whisper-*`, `mobiuslabsgmbh/faster-whisper-large-v3-turbo`) on Windows, and as
  ggml conversions (`ggerganov/whisper.cpp`) on macOS. Models are downloaded at first run and are not included
  in the installer or the disk image.
- [Silero VAD](https://github.com/snakers4/silero-vad), MIT License - the copy bundled with faster-whisper on
  Windows, and the same model vendored at `assets/vad/silero_vad_v6.onnx` on macOS.
- [PySide6 / Qt for Python](https://www.qt.io/qt-for-python), LGPL v3.
- *(Windows)* NVIDIA CUDA libraries (cuBLAS, cuDNN), redistributed under the NVIDIA software license terms that
  come with those packages.
- Other Python dependencies (onnxruntime, sounddevice/PortAudio, soxr, huggingface_hub, nvidia-ml-py) are used
  under their own licenses.
