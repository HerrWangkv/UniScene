#!/bin/bash
set -e  # Exit on any error

# Start Xvfb for headless rendering
Xvfb :1 -screen 0 1280x1024x24 &
export DISPLAY=:1
export QT_QPA_PLATFORM=offscreen
export ETS_TOOLKIT=qt
export QT_API=pyqt5

# Execute the command
exec "$@"
