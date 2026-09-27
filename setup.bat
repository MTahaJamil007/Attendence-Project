@echo off
REM One-time environment setup. Run from the project root.
setlocal

echo ============================================================
echo  Hand-Raise Attendance - environment setup
echo ============================================================

if not exist .venv (
    echo [1/4] creating virtual environment...
    py -3.14 -m venv .venv || py -3.12 -m venv .venv || goto :nopython
) else (
    echo [1/4] .venv already exists, reusing it
)

echo [2/4] upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet

echo [3/4] installing CUDA PyTorch (large download, please wait)...
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto :torchfail

echo [4/4] installing remaining dependencies...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :depsfail

echo.
echo ============================================================
echo  verifying the runtime
echo ============================================================
.venv\Scripts\python.exe src\runtime_check.py
if errorlevel 1 goto :runtimefail

echo.
echo Setup complete. Next steps:
echo   run.bat test        - logic tests (no camera needed)
echo   run.bat cameras     - list cameras
echo   run.bat benchmark   - pick model + imgsz
echo   run.bat preview     - live detector preview
exit /b 0

:nopython
echo ERROR: no suitable Python found. Install Python 3.14 or 3.12 and retry.
exit /b 1
:torchfail
echo ERROR: CUDA PyTorch install failed. Check your connection and retry.
exit /b 1
:depsfail
echo ERROR: dependency install failed.
exit /b 1
:runtimefail
echo ERROR: runtime check failed - see the output above.
echo A CPU-only torch build is the usual cause. Re-run setup.bat.
exit /b 1
