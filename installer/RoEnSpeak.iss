; RoEnSpeak installer (Inno Setup 6.5+; built and tested with 6.7).
;
; Normally compiled by build\build.ps1, which passes the version from app/contracts.py:
;   ISCC.exe /DAppVersion=0.1.0 installer\RoEnSpeak.iss
; Optional defines:
;   /DBundleDir=<dir>    PyInstaller onedir output (default: ..\build\out\RoEnSpeak)
;   /DOutputDir=<dir>    where RoEnSpeak-Setup.exe goes (default: ..\dist)
;   /DSignToolName=<n>   sign Setup + uninstaller with the sign tool <n>, defined on the command line
;                        as /S<n>=<command> (build.ps1 does this when SIGNTOOL_CERT is set)
;
; Install modes: per-user by default ({localappdata}\Programs\RoEnSpeak, no admin/UAC needed);
; the "Install for all users" choice (dialog or /ALLUSERS) installs to Program Files and needs admin.
; Silent: RoEnSpeak-Setup.exe /VERYSILENT /SUPPRESSMSGBOXES /CURRENTUSER [/DIR=<dir>] [/TASKS="desktopicon"]
; Silent uninstall: unins000.exe /VERYSILENT /SUPPRESSMSGBOXES  (keeps the downloaded models)

#ifndef AppVersion
  #error AppVersion is not defined. Pass /DAppVersion=<APP_VERSION from app/contracts.py> (build\build.ps1 does).
#endif
#ifndef BundleDir
  #define BundleDir "..\build\out\RoEnSpeak"
#endif
#ifndef OutputDir
  #define OutputDir "..\dist"
#endif

#define AppName "RoEnSpeak"
#define AppExe "RoEnSpeak.exe"
#define AppPublisher "RoEnSpeak"
; app/contracts.py single_instance_name(): CreateMutexW("Local\RoEnSpeak.SingleInstance").
; "Local\" is the per-session namespace, i.e. the same object as the unprefixed name that Inno's
; AppMutex/CheckForMutexes open, so this detects the user's running instance in this session.
; (Isolated test instances started with ROENSPEAK_CONFIG_DIR use a suffixed name and are not detected;
; Restart Manager / CloseApplications still handles their locked files.)
#define AppMutexName "Local\RoEnSpeak.SingleInstance"

[Setup]
AppId={{6B0E2C1E-4F3A-4C55-9E7B-7C2D9A1E5F40}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppMutex={#AppMutexName}
VersionInfoVersion={#AppVersion}
VersionInfoProductName={#AppName}
VersionInfoDescription={#AppName} Setup
; Per-user install by default; the user may switch to an all-users install (dialog or /ALLUSERS).
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline
DefaultDirName={autopf}\{#AppName}
DisableProgramGroupPage=yes
DefaultGroupName={#AppName}
UsePreviousAppDir=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern
SetupIconFile=..\assets\icons\app.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
OutputDir={#OutputDir}
OutputBaseFilename={#AppName}-Setup
; ~1.3 GB of input, dominated by cuBLAS. Solid LZMA2 with block-parallel compression in a separate
; 64-bit process keeps the build to a few minutes; "max" vs "ultra64" gains <1% on these DLLs.
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=6
LZMADictionarySize=65536
; Restart Manager: close anything still holding files in {app} (e.g. a test instance).
CloseApplications=yes
RestartApplications=no
#ifdef SignToolName
SignTool={#SignToolName}
SignedUninstaller=yes
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; Upgrades replace the whole bundle so files dropped from a newer build don't linger.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#BundleDir}\{#AppExe}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BundleDir}\RoEnSpeak-cli.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BundleDir}\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
; The app always starts in the tray; --minimized is the same flag autostart uses.
Filename: "{app}\{#AppExe}"; Parameters: "--minimized"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: dirifempty; Name: "{app}"

[Code]
const
  RunKey = 'Software\Microsoft\Windows\CurrentVersion\Run';

function AppIsRunning(): Boolean;
begin
  Result := CheckForMutexes('{#AppMutexName}');
end;

function WaitAppGone(Tries: Integer): Boolean;
var
  I: Integer;
begin
  for I := 1 to Tries do
  begin
    if not AppIsRunning() then
    begin
      Result := True;
      exit;
    end;
    Sleep(200);
  end;
  Result := not AppIsRunning();
end;

{ Close a running RoEnSpeak (tray app) before files are replaced or removed. Asks first in
  interactive mode; with /SUPPRESSMSGBOXES the answer is OK (close it). Returns False to abort. }
function CloseRunningApp(const Action: String): Boolean;
var
  Rc: Integer;
  TaskKill: String;
begin
  Result := True;
  if not AppIsRunning() then
    exit;
  if SuppressibleMsgBox('{#AppName} is running. It will be closed so that ' + Action + ' can continue.',
                        mbInformation, MB_OKCANCEL, IDOK) <> IDOK then
  begin
    Result := False;
    exit;
  end;
  TaskKill := ExpandConstant('{sys}\taskkill.exe');
  Log('Closing running {#AppExe}');
  { Polite WM_CLOSE first, then force. The tray app holds no unsaved user data (config is saved on change). }
  Exec(TaskKill, '/IM {#AppExe}', '', SW_HIDE, ewWaitUntilTerminated, Rc);
  if WaitAppGone(15) then
    exit;
  Exec(TaskKill, '/F /T /IM {#AppExe}', '', SW_HIDE, ewWaitUntilTerminated, Rc);
  if WaitAppGone(50) then
    exit;
  Result := SuppressibleMsgBox('{#AppName} could not be closed. Close it from its tray icon (Quit), then click Retry.',
                               mbError, MB_RETRYCANCEL, IDCANCEL) = IDRETRY;
  if Result then
    Result := CloseRunningApp(Action);
end;

function InitializeSetup(): Boolean;
begin
  Result := CloseRunningApp('Setup');
end;

function InitializeUninstall(): Boolean;
begin
  Result := CloseRunningApp('uninstall');
end;

{ Total size of a directory tree in bytes (for the "delete models?" question). }
function DirSize(const Dir: String): Int64;
var
  FindRec: TFindRec;
begin
  Result := 0;
  if FindFirst(AddBackslash(Dir) + '*', FindRec) then
  begin
    try
      repeat
        if (FindRec.Name <> '.') and (FindRec.Name <> '..') then
        begin
          if FindRec.Attributes and FILE_ATTRIBUTE_DIRECTORY <> 0 then
            Result := Result + DirSize(AddBackslash(Dir) + FindRec.Name)
          else
            Result := Result + (Int64(FindRec.SizeHigh) shl 32) + FindRec.SizeLow;
        end;
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
end;

procedure RemoveAutostartValue();
var
  Cmd: String;
begin
  { "Start with Windows" (app.win32.autostart): HKCU Run value named RoEnSpeak = "<exe>" --minimized.
    Only remove it when it points at this installation (a dev checkout may own the same value name). }
  if RegQueryStringValue(HKCU, RunKey, '{#AppName}', Cmd) then
    if Pos(Lowercase(ExpandConstant('{app}')), Lowercase(Cmd)) > 0 then
    begin
      Log('Removing HKCU Run value: ' + Cmd);
      RegDeleteValue(HKCU, RunKey, '{#AppName}');
    end;
end;

// Downloaded models (several GB) and logs live in %LOCALAPPDATA%\RoEnSpeak, outside the app dir.
// Ask once whether to delete them (default No; /SUPPRESSMSGBOXES answers No).
// Settings in %APPDATA%\RoEnSpeak\config.json are always kept so a reinstall keeps them.
procedure MaybeDeleteUserData();
var
  DataDir, ModelsDir, LogsDir: String;
  SizeMB: Int64;
begin
  DataDir := ExpandConstant('{localappdata}\{#AppName}');
  ModelsDir := DataDir + '\models';
  LogsDir := DataDir + '\logs';
  if not DirExists(ModelsDir) and not DirExists(LogsDir) then
    exit;
  SizeMB := (DirSize(ModelsDir) + DirSize(LogsDir)) div (1024 * 1024);
  if SuppressibleMsgBox('Also delete the downloaded speech models and logs (' + IntToStr(SizeMB) + ' MB)?' + #13#10#13#10 +
                        DataDir + #13#10#13#10 +
                        'Choose No to keep them for a later reinstall. Your settings are kept either way.',
                        mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES then
  begin
    Log('Deleting ' + ModelsDir + ' and ' + LogsDir);
    DelTree(ModelsDir, True, True, True);
    DelTree(LogsDir, True, True, True);
    RemoveDir(DataDir);
  end
  else
    Log('Keeping ' + DataDir);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
  begin
    RemoveAutostartValue();
    MaybeDeleteUserData();
  end;
end;
