# =====================================================================
#  RontoGPT — Installeur Windows
#  Installe l'IA, ses dependances et la commande `aura` dans PowerShell.
#
#  Usage :
#    powershell -ExecutionPolicy Bypass -File installer.ps1           # interactif
#    powershell -ExecutionPolicy Bypass -File installer.ps1 -Silencieux
#
#  Apres installation (nouvelle fenetre PowerShell) :
#    aura "quel est le carre de 7"        # -> 49
#    aura                                 # mode conversationnel
#    aura --verifier "question"           # re-verification cryptographique
# =====================================================================
param(
    [switch]$Silencieux,
    [string]$Destination = "$env:LOCALAPPDATA\RontoGPT"
)

$ErrorActionPreference = "Stop"

$RacineProjet = Split-Path -Parent $PSScriptRoot   # scripts/ -> racine

Write-Host ""
Write-Host "=== Installeur RontoGPT ===" -ForegroundColor Cyan
Write-Host "Destination : $Destination"

# ---------------------------------------------------------------
# 1. Verifier Python 3.10+ (py launcher en priorite)
# ---------------------------------------------------------------
function Trouver-Python {
    foreach ($c in @("py -3", "python3", "python")) {
        try {
            $v = & $c.Split(" ")[0] $c.Split(" ")[1..99] -c "import sys; print(sys.version_info[0]*100+sys.version_info[1])" 2>$null
            if ($LASTEXITCODE -eq 0 -and [int]$v -ge 310) { return $c }
        } catch { }
    }
    return $null
}

$PyCmd = Trouver-Python
if (-not $PyCmd) {
    Write-Host "[X] Python 3.10+ introuvable." -ForegroundColor Red
    Write-Host "    Installe-le depuis https://www.python.org/downloads/ puis relance." 
    if (-not $Silencieux) { Read-Host "Entree pour quitter" }
    exit 1
}
Write-Host "[OK] Python trouve via '$PyCmd'"

# ---------------------------------------------------------------
# 2. Copier le paquet (aura/, scripts/, cles publiques) vers Destination
# ---------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $Destination | Out-Null

$elements = @("aura", "scripts\telecharger_llama.py")
# Garde : si Destination == RacineProjet (installeur Inno qui appelle ce
# script depuis {app}), la copie source->destination serait un copie sur soi.
if ($RacineProjet -ne (Resolve-Path $Destination).Path) {
foreach ($e in $elements) {
    $src = Join-Path $RacineProjet $e
    $dst = Join-Path $Destination $e
    if (Test-Path $src -PathType Container) {
        Copy-Item -Recurse -Force $src (Split-Path -Parent $dst)
    } elseif (Test-Path $src) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
        Copy-Item -Force $src $dst
    }
}
}

# Fichiers facultatifs : presents s'ils existent (exe, .aef, .sig, cerveau)
foreach ($f in @("dist\RontoGPT.exe", "aura_system.aef.enc", "aura_system.aef",
                 "aura_system.aef.sig", "cle_publique.pub",
                 ".cache_aef\cerveau.gguf")) {
    $src = Join-Path $RacineProjet $f
    if (Test-Path $src) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent (Join-Path $Destination $f)) | Out-Null
        Copy-Item -Force $src (Join-Path $Destination $f)
        Write-Host "  + $f"
    }
}

# Le secret de dechiffrement, si fourni a l'installeur (env)
if ($env:AURA_SECRET_INSTALL) {
    # .NET WriteAllText = UTF-8 sans BOM (compatible PS 5.1 et 7+)
    [System.IO.File]::WriteAllText((Join-Path $Destination "secret_aef.txt"),
                                   $env:AURA_SECRET_INSTALL)
    Write-Host "  + secret_aef.txt (env AURA_SECRET_INSTALL)"
}

# ---------------------------------------------------------------
# 3. Environnement virtuel + dependances (SANS llama-cpp ici :
#    telecharger_llama.py l'installe avec la bonne roue)
# ---------------------------------------------------------------
Write-Host "[..] Creation du venv..."
& ($PyCmd.Split(" ")[0]) $PyCmd.Split(" ")[1..99] -m venv (Join-Path $Destination ".venv")
if ($LASTEXITCODE -ne 0) { throw "echec de creation du venv" }

$VenvPy = Join-Path $Destination ".venv\Scripts\python.exe"
Write-Host "[..] Installation des dependances (2-4 min)..."
& $VenvPy -m pip install --quiet --upgrade pip
& $VenvPy -m pip install --quiet numpy scikit-learn joblib gplearn ddgs requests cryptography
if ($LASTEXITCODE -ne 0) { throw "echec d'installation des dependances" }
Write-Host "[OK] Dependances installees"

# GGUF : reutiliser le cache kernel s'il a ete copie, sinon telecharger
$GgufCache = Join-Path $Destination ".cache_aef\cerveau.gguf"
$GgufModeles = Join-Path $Destination "modeles\Llama-3.2-1B-Instruct-Q4_K_M.gguf"
if (-not (Test-Path $GgufCache) -and -not (Test-Path $GgufModeles)) {
    Write-Host "[..] Telechargement du cerveau (807 Mo, 1re fois seulement)..."
    & $VenvPy (Join-Path $Destination "scripts\telecharger_llama.py")
}

# ---------------------------------------------------------------
# 4. Enregistrer les commandes `rontogpt` et `aura` (compatibilite)
# ---------------------------------------------------------------
$FonctionRontoGPT = @'

# === RontoGPT (IA locale) - ajoute par installer.ps1 ===
function rontogpt {
    $d = "__AURA_DEST__"
    $fichier = Join-Path $d "aura_system.aef.enc"     # chiffre prioritaire
    if (-not (Test-Path $fichier)) { $fichier = Join-Path $d "aura_system.aef" }
    if (-not (Test-Path $fichier)) {
        Write-Host "[RontoGPT] Aucun fichier .aef dans $d" -ForegroundColor Red
        return
    }
    Push-Location $d                       # -m aura.kernel exige le paquet
    try {                                  # dans le chemin courant
        $sec = Join-Path $d "secret_aef.txt"
        if (Test-Path $sec) { $env:AURA_SECRET = (Get-Content $sec -Raw).Trim() }
        & "$d\.venv\Scripts\python.exe" -m aura.kernel $fichier @args
    } finally {
        Pop-Location
    }
}
function aura { rontogpt @args }
'@
# le here-string simple-quote protege $args/$env de l'interpolation :
# seule la destination est injectee, a l'installation
$FonctionRontoGPT = $FonctionRontoGPT.Replace("__AURA_DEST__", $Destination)

$Profil = $PROFILE.CurrentUserAllHosts   # vaut pour toutes les consoles PS
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Profil) | Out-Null
$ContenuProfil = if (Test-Path $Profil) { Get-Content $Profil -Raw } else { "" }
if ($ContenuProfil -notmatch [regex]::Escape("# === RontoGPT")) {
    Add-Content -Path $Profil -Value $FonctionRontoGPT -Encoding utf8
    Write-Host "[OK] Commande 'rontogpt' ajoutee au profil PowerShell"
} else {
    Write-Host "[=] Commande 'rontogpt' deja presente dans le profil"
}

# ---------------------------------------------------------------
# 4 bis. Dependances lourdes manquantes + raccourci + config
# ---------------------------------------------------------------
# Le bloc 3 installait la base (numpy...) : ici on ajoute ce qui est
# indispensable pour que `python -m aura` (interface terminal) et le cerveau
# fonctionnent reels — roue CPU officielle pour llama-cpp-python.
Write-Host "[..] Cerveau (llama-cpp-python)..."
& $VenvPy -m pip install --quiet "huggingface-hub>=0.25" `
    "llama-cpp-python==0.3.2" `
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu/
if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] llama-cpp-python non installe (voir plus haut)." -ForegroundColor Yellow
} else {
    Write-Host "[OK] cerveau + interface installes"
}

# Configuration centralisee (profils rapide/equilibre/qualite)
$TomlSrc = Join-Path $RacineProjet "aura.toml"
if (Test-Path $TomlSrc) { Copy-Item -Force $TomlSrc $Destination }

# Lancement de l'interface : un .bat + un raccourci sur le Bureau
$Bat = Join-Path $Destination "Lancer RontoGPT.bat"
@"
@echo off
cd /d "$Destination"
".venv\Scripts\python.exe" -m aura %*
pause
"@ | Set-Content -Path $Bat -Encoding ascii
try {
    $Bureau = [Environment]::GetFolderPath("Desktop")
    $Shell = New-Object -ComObject WScript.Shell
    $Lnk = $Shell.CreateShortcut((Join-Path $Bureau "RontoGPT.lnk"))
    $Lnk.TargetPath = $Bat
    $Lnk.WorkingDirectory = $Destination
    $Lnk.Description = "RontoGPT - IA locale 100% offline"
    $Lnk.Save()
    Write-Host "[OK] Raccourci 'RontoGPT' cree sur le Bureau"
} catch {
    Write-Host "[!] Raccourci Bureau impossible : $_" -ForegroundColor Yellow
}

# ---------------------------------------------------------------
# 5. Test final : boot du .aef + reponse
# ---------------------------------------------------------------
Write-Host "[..] Test final..."
$FichierSecret = Join-Path $Destination "secret_aef.txt"
if (Test-Path $FichierSecret) {
    $env:AURA_SECRET = (Get-Content $FichierSecret -Raw).Trim()
}
$Reponse = & $VenvPy -m aura.kernel (Join-Path $Destination "aura_system.aef.enc") "quel est le carre de 7" 2>$null
if ($Reponse -match "49") {
    Write-Host "[OK] Test reussi : 7^2 = $Reponse" -ForegroundColor Green
} else {
    Write-Host "[!] Le test n'a pas repondu 49 (normal si le .aef.enc est absent)." -ForegroundColor Yellow
    Write-Host "    Le mode dev reste disponible : $RacineProjet"
}

Write-Host ""
Write-Host "=== Installation terminee ===" -ForegroundColor Cyan
Write-Host "Ouvre une NOUVELLE fenetre PowerShell puis :"
Write-Host '  aura "quel est le carre de 7"'
Write-Host "Ou double-clique sur le raccourci 'RontoGPT' du Bureau (interface terminal)."
Write-Host "Pour voir si une mise a jour existe :"
Write-Host "  & `$Destination\.venv\Scripts\python.exe -m aura.maj"
if (-not $Silencieux) { Read-Host "Entree pour fermer" }
