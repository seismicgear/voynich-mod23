; Inno Setup script — builds VoynichWorkbench-Setup.exe.
; Compiled in CI on the Windows runner after PyInstaller produces
; dist\VoynichWorkbench.exe (run from the repository root):
;   iscc packaging\windows_installer.iss

[Setup]
AppName=Voynich Decipherment Workbench
AppVersion=3.2.0
AppPublisher=Montgomery Kuykendall
DefaultDirName={autopf}\VoynichWorkbench
DefaultGroupName=Voynich Workbench
OutputDir=..\dist
OutputBaseFilename=VoynichWorkbench-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "..\dist\VoynichWorkbench.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Voynich Workbench"; Filename: "{app}\VoynichWorkbench.exe"
Name: "{autodesktop}\Voynich Workbench"; Filename: "{app}\VoynichWorkbench.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\VoynichWorkbench.exe"; Description: "Launch the workbench"; Flags: nowait postinstall skipifsilent
