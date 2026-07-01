#!/bin/bash
Xvfb :99 -screen 0 1920x1080x24 &
XVFB_PID=$!
sleep 1
export DISPLAY=:99
export QT_QPA_PLATFORM=xcb
uv run python run_napari_opym.py &
NAPARI_PID=$!
sleep 8
import -window root /tmp/napari_screenshot.png
kill $NAPARI_PID
kill $XVFB_PID
