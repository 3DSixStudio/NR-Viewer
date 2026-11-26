@echo off
REM Chay bang Python co san cua Maya 2026 (khong can cai Python rieng)
cd /d "%~dp0"
set "MAYAPY=C:\Program Files\Autodesk\Maya2026\bin\mayapy.exe"
if not exist "%MAYAPY%" (
  echo Khong thay %MAYAPY%
  pause & exit /b 1
)
"%MAYAPY%" -c "import OpenGL" >nul 2>&1 || "%MAYAPY%" -m pip install --user PyOpenGL
"%MAYAPY%" nr_viewer.py %*
if errorlevel 1 pause
