@echo off
REM Tim Python that (bo qua ban "gia" cua Microsoft Store)
set "PY="
py -3 -c "import sys" >nul 2>&1 && set "PY=py -3"
if not defined PY python -c "import sys" >nul 2>&1 && set "PY=python"
if not defined PY for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do if exist "%%D\python.exe" set "PY="%%D\python.exe""
if not defined PY for /d %%D in ("%ProgramFiles%\Python3*") do if exist "%%D\python.exe" set "PY="%%D\python.exe""
if not defined PY (
  echo.
  echo [LOI] May chua cai Python.
  echo  1. Tai Python 3.14 hoac 3.15 tai https://www.python.org/downloads/windows/
  echo  2. Khi cai, TICK o "Add python.exe to PATH"
  echo  3. Chay lai file .bat nay
  echo.
  pause
  exit /b 1
)
echo Dung Python: %PY%
