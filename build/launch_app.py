"""PyInstaller entry script for the windowed RoEnSpeak.exe (tray app): runs app.main.main()."""

import sys

from app.main import main

if __name__ == "__main__":
    sys.exit(main())
