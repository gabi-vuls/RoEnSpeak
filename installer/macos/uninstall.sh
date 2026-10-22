#!/usr/bin/env bash
# RoEnSpeak uninstaller for macOS.
#
#   bash uninstall.sh [--app PATH] [--yes] [--all] [--dry-run]
#
# A .dmg is a drag-install: dragging RoEnSpeak.app to the Trash removes the application but leaves
# the login item, the downloaded models and the logs behind. This script does what
# installer/RoEnSpeak.iss's uninstaller does on Windows, in the same order and with the same
# defaults:
#
#   1. quits a running RoEnSpeak                      (Inno: CloseRunningApp / taskkill)
#   2. unloads + removes the "Open at login" LaunchAgent, but ONLY if it points at the app being
#      removed                                        (Inno: RemoveAutostartValue)
#   3. removes the application bundle                 (Inno: uninstall of {app})
#   4. ASKS whether to delete the downloaded models and the logs, default No
#                                                     (Inno: MaybeDeleteUserData)
#   5. always keeps ~/Library/Application Support/RoEnSpeak/config.json, so a reinstall keeps your
#      settings                                       (Inno: settings are never touched)
#
# Options
#   --app PATH   the bundle to remove (default: /Applications/RoEnSpeak.app, then ~/Applications)
#   --yes        answer yes to every prompt, INCLUDING deleting models and logs. Non-interactive.
#                Without it, nothing is ever deleted without a y/N answer.
#   --all        also delete config.json (a full reset). Still prompted unless --yes is given.
#   --dry-run    print what would happen and change nothing.
#
# Safe to run twice: every step checks first and reports "not found" instead of failing. Needs no
# administrator rights unless the bundle is in /Applications and owned by another user.
set -euo pipefail

APP_NAME="RoEnSpeak"
BUNDLE_ID="com.roenspeak.RoEnSpeak"
APP=""
ASSUME_YES=0
REMOVE_CONFIG=0
DRY_RUN=0

SUPPORT_DIR="$HOME/Library/Application Support/$APP_NAME"
MODELS_DIR="$SUPPORT_DIR/models"
CONFIG_FILE="$SUPPORT_DIR/config.json"
LOGS_DIR="$HOME/Library/Logs/$APP_NAME"
AGENT_PLIST="$HOME/Library/LaunchAgents/$BUNDLE_ID.plist"

die()  { printf '%s: %s\n' "$APP_NAME uninstaller" "$*" >&2; exit 1; }
say()  { printf '%s\n' "$*"; }
step() { printf '\n==> %s\n' "$*"; }

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; $d'; exit 0; }

while [ $# -gt 0 ]; do
    case "$1" in
        --app)     [ $# -ge 2 ] || die "--app needs a path"; APP="$2"; shift 2 ;;
        --yes|-y)  ASSUME_YES=1; shift ;;
        --all)     REMOVE_CONFIG=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage ;;
        *) die "unknown option: $1 (try --help)" ;;
    esac
done

[ "$(uname -s)" = "Darwin" ] || die "this uninstaller is for macOS"

# ask PROMPT -> 0 for yes, 1 for no. Default is NO, exactly like the Inno Setup MB_DEFBUTTON2 box.
# With --yes the answer is yes; with no tty and no --yes the answer is no (nothing is deleted).
ask() {
    if [ "$ASSUME_YES" -eq 1 ]; then
        say "   $1 [y/N] y   (--yes)"
        return 0
    fi
    if [ ! -t 0 ]; then
        say "   $1 [y/N] n   (not a terminal; pass --yes to delete)"
        return 1
    fi
    local reply=""
    printf '   %s [y/N] ' "$1"
    read -r reply || reply=""
    case "$reply" in [yY]|[yY][eE][sS]) return 0 ;; *) return 1 ;; esac
}

do_rm() {  # do_rm PATH DESCRIPTION
    if [ "$DRY_RUN" -eq 1 ]; then
        say "   would remove $1"
        return 0
    fi
    rm -rf -- "$1" || die "could not remove $1 ($2). Check permissions, or move it to the Trash by hand."
    say "   removed $1"
}

human_size() {  # human_size PATH... -> e.g. "3.1 GB"
    local total=0 kb path
    for path in "$@"; do
        if [ -e "$path" ]; then
            kb="$(du -sk "$path" 2>/dev/null | awk '{print $1}')"
            total=$(( total + ${kb:-0} ))
        fi
    done
    awk -v kb="$total" 'BEGIN {
        if (kb >= 1048576) printf "%.1f GB", kb/1048576;
        else if (kb >= 1024) printf "%.0f MB", kb/1024;
        else printf "%d KB", kb
    }'
}

# ---------------------------------------------------------------------------------------------
# Locate the bundle.
# ---------------------------------------------------------------------------------------------
if [ -z "$APP" ]; then
    for candidate in "/Applications/$APP_NAME.app" "$HOME/Applications/$APP_NAME.app"; do
        if [ -d "$candidate" ]; then APP="$candidate"; break; fi
    done
fi
if [ -n "$APP" ]; then
    APP="${APP%/}"
    [ -d "$APP" ] || die "no application bundle at $APP"
    say "Application : $APP"
else
    say "Application : not found in /Applications or ~/Applications (will still clean up the rest)"
fi
[ "$DRY_RUN" -eq 1 ] && say "DRY RUN: nothing will be changed."

# ---------------------------------------------------------------------------------------------
step "Quit $APP_NAME if it is running"
# The tray app holds no unsaved user data (config is written on change), so a polite quit is
# enough; pkill -TERM lets Qt shut down, unregister the hotkey and release the model.
if pgrep -x "$APP_NAME" >/dev/null 2>&1; then
    if [ "$DRY_RUN" -eq 1 ]; then
        say "   would quit the running $APP_NAME"
    else
        pkill -x "$APP_NAME" 2>/dev/null || true
        for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
            pgrep -x "$APP_NAME" >/dev/null 2>&1 || break
            sleep 0.2
        done
        if pgrep -x "$APP_NAME" >/dev/null 2>&1; then
            say "   it did not quit; forcing"
            pkill -9 -x "$APP_NAME" 2>/dev/null || true
            sleep 0.5
        fi
        if pgrep -x "$APP_NAME" >/dev/null 2>&1; then
            die "$APP_NAME is still running. Quit it from its menu-bar icon and run this script again."
        fi
        say "   quit"
    fi
else
    say "   not running"
fi

# ---------------------------------------------------------------------------------------------
step "Login item (\"Open at login\")"
# Remove the LaunchAgent only when it points at the bundle being removed. A developer checkout can
# own the same label, and a second copy of the app elsewhere must keep working -- this is the exact
# counterpart of the Inno uninstaller's "only if the Run value points into this install dir" check.
if [ -f "$AGENT_PLIST" ]; then
    AGENT_TARGET="$(/usr/libexec/PlistBuddy -c 'Print :ProgramArguments:0' "$AGENT_PLIST" 2>/dev/null || true)"
    [ -n "$AGENT_TARGET" ] || AGENT_TARGET="$(/usr/libexec/PlistBuddy -c 'Print :Program' "$AGENT_PLIST" 2>/dev/null || true)"
    say "   $AGENT_PLIST -> ${AGENT_TARGET:-<unreadable>}"
    if [ -n "$APP" ] && [ -n "$AGENT_TARGET" ] && [ "${AGENT_TARGET#"$APP"}" = "$AGENT_TARGET" ]; then
        say "   kept: it points somewhere else, not at $APP"
    elif [ -z "$APP" ] && [ -n "$AGENT_TARGET" ] && [ -e "$AGENT_TARGET" ]; then
        say "   kept: no bundle given to remove and its target still exists"
    else
        if [ "$DRY_RUN" -eq 1 ]; then
            say "   would unload and remove $AGENT_PLIST"
        else
            launchctl bootout "gui/$(id -u)/$BUNDLE_ID" 2>/dev/null \
                || launchctl unload "$AGENT_PLIST" 2>/dev/null || true
            do_rm "$AGENT_PLIST" "login item"
        fi
    fi
else
    say "   no login item installed"
fi

# ---------------------------------------------------------------------------------------------
step "Application bundle"
if [ -n "$APP" ]; then
    if ask "Remove $APP?"; then
        do_rm "$APP" "application bundle"
    else
        say "   kept $APP"
    fi
else
    say "   nothing to remove"
fi

# ---------------------------------------------------------------------------------------------
step "Downloaded speech models and logs"
if [ -d "$MODELS_DIR" ] || [ -d "$LOGS_DIR" ]; then
    [ -d "$MODELS_DIR" ] && say "   $MODELS_DIR"
    [ -d "$LOGS_DIR" ]   && say "   $LOGS_DIR"
    say "   ($(human_size "$MODELS_DIR" "$LOGS_DIR") in total)"
    say "   Keep them and a reinstall will not download the model again. Your settings are kept either way."
    if ask "Delete the models and logs?"; then
        [ -d "$MODELS_DIR" ] && do_rm "$MODELS_DIR" "models"
        [ -d "$LOGS_DIR" ]   && do_rm "$LOGS_DIR"   "logs"
    else
        say "   kept"
    fi
else
    say "   none found"
fi

# ---------------------------------------------------------------------------------------------
step "Settings"
if [ -f "$CONFIG_FILE" ]; then
    if [ "$REMOVE_CONFIG" -eq 1 ]; then
        if ask "Also delete your settings ($CONFIG_FILE)?"; then
            do_rm "$CONFIG_FILE" "settings"
        else
            say "   kept $CONFIG_FILE"
        fi
    else
        say "   kept $CONFIG_FILE (pass --all to delete it too)"
    fi
else
    say "   no settings file"
fi

# ---------------------------------------------------------------------------------------------
step "Tidy up"
if [ "$DRY_RUN" -eq 0 ]; then
    # Only ever remove these directories when they are empty -- the Inno uninstaller's
    # "Type: dirifempty" behaviour. rmdir on a non-empty directory fails, which is the point.
    rmdir "$SUPPORT_DIR" 2>/dev/null && say "   removed the now-empty $SUPPORT_DIR" || true
    rmdir "$LOGS_DIR" 2>/dev/null && say "   removed the now-empty $LOGS_DIR" || true
    # Reset the saved window position / Qt state that macOS keeps for the bundle identifier.
    defaults delete "$BUNDLE_ID" >/dev/null 2>&1 && say "   cleared the $BUNDLE_ID preference domain" || true
    rm -f "$HOME/Library/Saved Application State/$BUNDLE_ID.savedState"/* 2>/dev/null || true
    rmdir "$HOME/Library/Saved Application State/$BUNDLE_ID.savedState" 2>/dev/null || true
fi
say "   done"

printf '\n%s\n' "Uninstall complete."
cat <<'TAIL'

One thing this script cannot do for you: macOS keeps the Accessibility and Microphone permissions
in its own protected database (TCC), which no script may edit. If you want to clear them, remove
the RoEnSpeak entries by hand:

    System Settings > Privacy & Security > Accessibility   (select RoEnSpeak, click "-")
    System Settings > Privacy & Security > Microphone      (select RoEnSpeak, click "-")

Leaving them is harmless: with the app gone the entries simply do nothing, and a later reinstall
will ask again.
TAIL
