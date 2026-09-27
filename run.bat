@echo off
REM Task launcher.
REM Usage: run.bat [app|check|test|ui|cameras|benchmark|smoke|preview]
setlocal
set PY=.venv\Scripts\python.exe

if not exist %PY% (
    echo No virtual environment found. Run setup.bat first.
    exit /b 1
)

if "%1"=="" goto app
if /i "%1"=="app"       goto app
if /i "%1"=="check"     goto check
if /i "%1"=="test"      goto test
if /i "%1"=="ui"        goto uitest
if /i "%1"=="cameras"   goto cameras
if /i "%1"=="benchmark" goto benchmark
if /i "%1"=="smoke"     goto smoke
if /i "%1"=="preview"   goto preview
echo Unknown task "%1".
echo Valid: app check test ui cameras benchmark smoke preview
exit /b 1

:app
%PY% src\main.py %2 %3 %4
exit /b %errorlevel%

:check
%PY% src\runtime_check.py
exit /b %errorlevel%

:test
%PY% tools\check_config.py
if errorlevel 1 exit /b 1
%PY% tools\test_liveness.py
if errorlevel 1 exit /b 1
%PY% tools\test_detector.py
if errorlevel 1 exit /b 1
%PY% tools\test_session.py
if errorlevel 1 exit /b 1
%PY% tools\test_ui.py
exit /b %errorlevel%

:uitest
%PY% tools\test_ui.py
exit /b %errorlevel%

:cameras
%PY% src\camera.py
exit /b %errorlevel%

:benchmark
%PY% tools\benchmark.py %2 %3 %4 %5 %6
exit /b %errorlevel%

:smoke
%PY% tools\smoke.py %2 %3
exit /b %errorlevel%

:preview
%PY% tools\preview.py
exit /b %errorlevel%
