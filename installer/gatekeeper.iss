#define ProductName "Manufacturing Junction gateKeeper AI Vision"
#define ProductPublisher "gateKeeper contributors"
#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif
#ifndef SourceDir
  #error SourceDir must point to the staged release directory.
#endif

[Setup]
AppId={{A32F60C4-2DCE-4A34-A2F7-E733E6C36A20}
AppName={#ProductName}
AppVersion={#MyAppVersion}
AppPublisher={#ProductPublisher}
DefaultDirName={autopf}\Manufacturing Junction\gateKeeper AI Vision
DefaultGroupName={#ProductName}
DisableProgramGroupPage=yes
OutputBaseFilename=Manufacturing-Junction-gateKeeper-AI-Vision-Setup-v{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
ExtraDiskSpaceRequired=6442450944
LicenseFile={#SourceDir}\LICENSE
InfoBeforeFile={#SourceDir}\NOTICE
UninstallDisplayName={#ProductName}

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#ProductName}"; Filename: "{app}\Manufacturing Junction gateKeeper AI Vision.exe"
Name: "{autodesktop}\{#ProductName}"; Filename: "{app}\Manufacturing Junction gateKeeper AI Vision.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\Manufacturing Junction gateKeeper AI Vision.exe"; Description: "Launch {#ProductName}"; Flags: nowait postinstall skipifsilent
