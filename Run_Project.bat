
@echo off
title Spam Email Detection Project
cd /d "%~dp0"

if not exist "spam_model.pkl" (
    echo Training the machine learning model...
    python train_model.py
    if errorlevel 1 (
        echo Model training failed. Check Python and dataset.
        pause
        exit /b 1
    )
)

python -m streamlit run app.py

pause
