#!/usr/bin/env bash
set -e

echo "================================================================="
echo "   ISRO SAC DepthWizard: Single-View Height Estimation & 3D Flythrough"
echo "   SIH26175 | Space Applications Centre (SAC), Ahmedabad"
echo "================================================================="

# Create virtual environment if missing
if [ ! -d ".venv" ]; then
    echo "[*] Initializing virtual environment..."
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

echo "[*] Running DepthWizard Self-Verification Test Suite..."
.venv/bin/python -m unittest tests/test_depth_wizard.py

echo "[*] Launching DepthWizard 3D Flythrough Cockpit on http://127.0.0.1:8099 ..."
.venv/bin/uvicorn src.depth_wizard.server:app --host 127.0.0.1 --port 8099 --reload
