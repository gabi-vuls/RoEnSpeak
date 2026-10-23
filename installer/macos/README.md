# macOS packaging

The Windows build ships an Inno Setup installer (`../RoEnSpeak.iss`). macOS ships a
**drag-install disk image** instead, so this directory is much smaller than its Windows sibling:
a `.dmg` needs no installer program, and the only thing it cannot do is uninstall.

| | Windows | macOS |
|---|---|---|
| Package | `dist/RoEnSpeak-Setup.exe` (Inno Setup) | `dist/RoEnSpeak-<version>.dmg` (`hdiutil`) |
| Built by | `build/build.ps1` → `ISCC` | `build/build_macos.sh` → `build/make_dmg.sh` |
| Install | wizard, per-user, no admin | drag `RoEnSpeak.app` to `/Applications` |
| Install script | `installer/RoEnSpeak.iss` | none needed — the image is just the app + an `/Applications` symlink |
| Uninstall | Settings → Apps → Uninstall | **`installer/macos/uninstall.sh`** (this directory) |

## Why a script, and why only for uninstalling

A `.dmg` has no install or uninstall hooks at all. Everything `RoEnSpeak.iss` does *on install* —
copy files, make shortcuts, offer to launch — a drag-install does for free: the Finder copies the
bundle, Launchpad and Spotlight index it automatically, and the user double-clicks it.

The four things `RoEnSpeak.iss` does *on uninstall* have no equivalent, because dragging the app to
the Trash removes the bundle and nothing else. `uninstall.sh` is the counterpart, step for step:

| `RoEnSpeak.iss` | `uninstall.sh` |
|---|---|
| `CloseRunningApp` — `taskkill /IM RoEnSpeak.exe`, then `/F` | `pkill -x RoEnSpeak`, wait 3 s, then `pkill -9`; aborts with a clear message if it survives |
| `RemoveAutostartValue` — delete the `HKCU\…\Run\RoEnSpeak` value **only if it points into this install dir** | `launchctl bootout` + remove `~/Library/LaunchAgents/com.roenspeak.RoEnSpeak.plist` **only if its `ProgramArguments[0]` is inside the bundle being removed** |
| uninstall `{app}` | remove `/Applications/RoEnSpeak.app` |
| `MaybeDeleteUserData` — ask once about models + logs, **default No**; settings always kept | ask once about `~/Library/Application Support/RoEnSpeak/models` and `~/Library/Logs/RoEnSpeak`, **default No**; `config.json` always kept |
| `Type: dirifempty` | `rmdir` (fails harmlessly when not empty), plus the Qt preference domain and saved window state |

## Usage

```sh
# From the repo:
bash installer/macos/uninstall.sh

# Or, after installing, straight out of the bundle you are about to delete:
bash /Applications/RoEnSpeak.app/Contents/Resources/uninstall.sh   # only if shipped inside the app

# See exactly what would happen, change nothing:
bash installer/macos/uninstall.sh --dry-run

# Unattended, including the models (the --VERYSILENT /SUPPRESSMSGBOXES analogue):
bash installer/macos/uninstall.sh --yes

# Full reset, settings included:
bash installer/macos/uninstall.sh --yes --all

# A copy that is not in /Applications:
bash installer/macos/uninstall.sh --app ~/Applications/RoEnSpeak.app
```

| Option | Meaning |
|---|---|
| `--app PATH` | The bundle to remove. Default: `/Applications/RoEnSpeak.app`, then `~/Applications/RoEnSpeak.app`. |
| `--yes`, `-y` | Answer yes to every prompt, **including** deleting models and logs. |
| `--all` | Also delete `config.json`. Still prompted unless `--yes` is given. |
| `--dry-run` | Print the plan, change nothing. |

## Guarantees

- **Non-destructive by default.** Every deletion asks, and every prompt defaults to **No**. With no
  terminal attached and no `--yes`, the answer is No, so piping the script into `sh` can never
  delete your models.
- **Settings survive.** `~/Library/Application Support/RoEnSpeak/config.json` is only ever removed
  with `--all`, matching the Windows uninstaller, which never touches `%APPDATA%\RoEnSpeak`.
- **Idempotent.** Each step checks first and prints "not found" rather than failing, so running it
  twice, or after a manual drag-to-Trash, is fine.
- **Never touches a second installation.** The login item is removed only when it points at the
  bundle being uninstalled.
- **No `sudo`.** Everything it touches is in `$HOME` or in `/Applications`, which the installing
  user owns. If `/Applications/RoEnSpeak.app` belongs to someone else, the script says so instead
  of escalating.

## What it cannot do

macOS stores the **Accessibility** and **Microphone** grants in the TCC database, which is
protected by System Integrity Protection: no script, with or without `sudo`, may edit it. The
script therefore prints the two System Settings panes to clear by hand. Leaving the entries is
harmless — they refer to an app that no longer exists, and a reinstall prompts again (and *must*
prompt again, because an ad-hoc signature's code requirement changes on every rebuild; see
[BUILD.md](../../BUILD.md)).
