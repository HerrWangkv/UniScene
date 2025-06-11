#!/bin/bash
set -e  # Exit on any error

# Start Xvfb for headless rendering
Xvfb :1 -screen 0 1280x1024x24 &

# Execute the command
exec "$@"
