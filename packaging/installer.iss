; Inno Setup script for ClipShow Windows installer
#ifndef AppVer
  #define AppVer "0.0.0"
#endif

[Setup]
AppName=ClipShow
AppVersion={#AppVer}
AppPublisher=ClipShow Contributors
DefaultDirName={autopf}\ClipShow
DefaultGroupName=ClipShow
OutputBaseFilename=ClipShow-{#AppVer}-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
UninstallDisplayName=ClipShow
LicenseFile=..\LICENSE

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\ClipShow\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\ClipShow"; Filename: "{app}\ClipShow.exe"
Name: "{group}\{cm:UninstallProgram,ClipShow}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\ClipShow"; Filename: "{app}\ClipShow.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\ClipShow.exe"; Description: "{cm:LaunchProgram,ClipShow}"; Flags: nowait postinstall skipifsilent
