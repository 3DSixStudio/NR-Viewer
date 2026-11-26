@echo off
cd /d "%~dp0"
call _find_python.bat || exit /b 1
%PY% -m pip install -q -r requirements.txt || (pause & exit /b 1)
%PY% nr_viewer.py %*
if errorlevel 1 pause
