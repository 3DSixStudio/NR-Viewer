; Inno Setup script cho NR Viewer
; Build:  ISCC.exe /DAppVersion=1.0.3 installer.iss
; Cai vao thu muc nguoi dung (khong can quyen admin) de tinh nang tu cap nhat ghi de duoc.

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{B8E7B4B0-3C2A-4E5F-9A1D-NRVIEWER0001}
AppName=NR Viewer
AppVersion={#AppVersion}
AppPublisher=3DSixStudio
AppPublisherURL=https://github.com/3DSixStudio/NR-Viewer
DefaultDirName={localappdata}\Programs\NR Viewer
DefaultGroupName=NR Viewer
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=installer_out
OutputBaseFilename=NRViewer-Setup
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\NRViewer.exe
UninstallDisplayName=NR Viewer
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible

[Languages]
Name: "vi"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Tao loi tat ngoai man hinh (Desktop)"; GroupDescription: "Loi tat:"

[Files]
Source: "dist\NRViewer.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\NR Viewer"; Filename: "{app}\NRViewer.exe"
Name: "{group}\Go cai dat NR Viewer"; Filename: "{uninstallexe}"
Name: "{userdesktop}\NR Viewer"; Filename: "{app}\NRViewer.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\NRViewer.exe"; Description: "Mo NR Viewer ngay"; Flags: nowait postinstall skipifsilent
