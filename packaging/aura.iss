; =====================================================================
;  RontoGPT — installeur Windows (Inno Setup 6)
;  Construit par CI :  ISCC /DVersion=0.3.0 /DRacine=<racine> packaging\aura.iss
;  Produit : dist\RontoGPT-Setup-0.3.0.exe
;
;  L'installeur installe le PAQUET, pas la première réponse. C'est
;  scripts\installer.ps1 qui construit le venv, installe les dépendances et
;  télécharge le cerveau si aucun modèle n'est présent. Cet .iss ne fait que
;  préparer les fichiers pour que ce script tourne sans terminal.
;
;  Écran de premier lancement : si `modeles/` est vide, le TUI affiche un
;  écran « Bienvenue », télécharge le modèle avec une barre de progression,
;  puis propose une première question prouvée (« 17 x 23 ? »).
; =====================================================================

#ifndef Version
  #define Version "0.3.0"
#endif
#ifndef Racine
  #define Racine ".."
#endif

[Setup]
AppId={{A7E6A1C4-3B5E-4D2F-9C11-0AURA1B2C3D4E}
AppName=RontoGPT
AppVersion={#Version}
AppPublisher=Simonc44
DefaultDirName={autopf}\RontoGPT
DefaultGroupName=RontoGPT
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=RontoGPT-Setup-{#Version}
Compression=lzma2/max
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
WizardStyle=modern

[Languages]
Name: "fr"; MessagesFile: "compiler:Languages\French.isl"

[Files]
Source: "{#Racine}\aura\*"; DestDir: "{app}\aura"; Flags: recursesubdirs ignoreversion
Source: "{#Racine}\scripts\*"; DestDir: "{app}\scripts"; Flags: recursesubdirs ignoreversion
Source: "{#Racine}\packaging\Lancer RontoGPT.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Racine}\pyproject.toml"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Racine}\README.md"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#Racine}\aura.toml"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#Racine}\skills.toml"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#Racine}\tui\package.json"; DestDir: "{app}\tui"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#Racine}\tui\tsconfig.json"; DestDir: "{app}\tui"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#Racine}\tui\src\*"; DestDir: "{app}\tui\src"; Flags: recursesubdirs ignoreversion skipifsourcedoesntexist
; le .aef chiffre s'il existe déjà (sinon téléchargement ultérieur)
Source: "{#Racine}\aura_system.aef.enc"; DestDir: "{app}"; Flags: skipifsourcedoesntexist
Source: "{#Racine}\cle_publique.pub"; DestDir: "{app}"; Flags: skipifsourcedoesntexist

[Icons]
Name: "{autoprograms}\RontoGPT"; Filename: "{app}\Lancer RontoGPT.bat"
Name: "{autodesktop}\RontoGPT"; Filename: "{app}\Lancer RontoGPT.bat"
Name: "{autoprograms}\Desinstaurer RontoGPT"; Filename: "{uninstallexe}"

[Run]
; 1) venv + dépendances + raccourci, sans écran.
Filename: "powershell.exe"; \
  Parameters: "-ExecutionPolicy Bypass -File \""{app}\scripts\installer.ps1\"" -Silencieux -Destination \""{app}\"\""; \
  StatusMsg: "Installation du cerveau et des dépendances (2-4 min)..."; \
  Flags: runascurrentuser postinstall skipifsilent

; 2) premier lancement : écran TUI ; le .bat accepte d'être re-lancé sans re-installer.
Filename: "Lancer RontoGPT.bat"; Parameters: ""; StatusMsg: "Lancement de RontoGPT"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\modeles"
Type: filesandordirs; Name: "{app}\.cache_aef"
Type: filesandordirs; Name: "{app}\serveur"
