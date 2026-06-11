"""PyInstaller entry point for the desktop builds."""

import multiprocessing
import sys

from voynich.launcher import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
