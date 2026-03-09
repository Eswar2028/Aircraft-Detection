@echo off
setlocal
cd /d "e:\Aircarft-Detection-YOLO"
echo Launching Aircraft Detection Web App...
.\venv\Scripts\python.exe -m streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Failed to start the application. 
    echo Please make sure the virtual environment exists in e:\Aircarft-Detection-YOLO\venv
    pause
)
endlocal
