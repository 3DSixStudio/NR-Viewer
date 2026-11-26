@echo off
REM Dong goi NRViewer.exe
REM   Mac dinh: 1 file duy nhat  -> dist\NRViewer.exe
REM   Muon dang thu muc (mo nhanh hon): bo chu --onefile ben duoi
cd /d "%~dp0"
call _find_python.bat || exit /b 1
%PY% -m pip install -r requirements.txt pyinstaller || goto :fail
%PY% -m PyInstaller --noconfirm --onefile --windowed --name NRViewer ^
  --icon icon.ico --add-data "icon.ico;." ^
  --hidden-import OpenGL.platform.win32 --collect-submodules OpenGL ^
  nr_viewer.py || goto :fail
if exist "dist\NRViewer.exe" (
  echo.
  echo ===== XONG: %~dp0dist\NRViewer.exe =====
  explorer /select,"dist\NRViewer.exe"
  pause & exit /b 0
)
if exist "dist\NRViewer\NRViewer.exe" (
  echo.
  echo ===== XONG: %~dp0dist\NRViewer\NRViewer.exe =====
  explorer "dist\NRViewer"
  pause & exit /b 0
)
:fail
echo.
echo [LOI] Build that bai - chup man hinh cua so nay gui lai.
pause
exit /b 1
