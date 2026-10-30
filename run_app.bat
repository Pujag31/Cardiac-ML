@echo off
title CardioGuard AI - Healthcare Clinical Decision Support System
cd /d "%~dp0"
echo ==============================================================
echo Starting CardioGuard AI Clinical Decision Support System...
echo ==============================================================
python -m streamlit run app.py
pause
