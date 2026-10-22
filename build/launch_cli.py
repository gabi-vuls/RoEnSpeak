"""PyInstaller entry script for the console RoEnSpeak-cli.exe (.wav test mode): runs app.cli.main()."""

import sys

from app.cli import main

if __name__ == "__main__":
    sys.exit(main())
