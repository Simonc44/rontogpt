@echo off
rem RontoGPT — interface terminal (TUI).
rem %~dp0 = dossier de ce fichier : marche quelle que soit l'installation.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [RontoGPT] Cerveau non installe — lancement de l'installation...
    powershell -ExecutionPolicy Bypass -File "%~dp0scripts\installer.ps1" -Silencieux -Destination "%~dp0."
    if not exist ".venv\Scripts\python.exe" (
        echo [RontoGPT] Installation impossible. Voir les messages ci-dessus.
        pause
        exit /b 1
    )
)

set "MODELE=modeles\Llama-3.2-1B-Instruct-Q4_K_M.gguf"
set "MIN_ESPACE=3_500_000_000"
for /f "tokens=3" %%A in ('"%ProgramFiles%\Git\usr\bin\df.exe" --output=avail "%~dp0" 2^>nul ^
    ^|^| powershell -NoProfile -Command "[math]::Round((Get-PSDrive -Name $(Split-Path "%~dp0").Split(':')[0]).Free / 1MB)"') do set /a "DISP=%%A"
if defined DISP (
    set /a "MIN_MB=%MIN_ESPACE% / 1024 / 1024 / 1024" 2>nul || set "MIN_MB=3"
    powershell -NoProfile -Command "if ((Get-PSDrive ($(Split-Path '%~dp0').Split(':')[0])).Free -lt (%MIN_ESPACE%)) { exit 1 }" 2>nul || (
        echo [RontoGPT] Place insuffisante : %MIN_MB% Go recommandes au minimum.
        pause
        exit /b 1
    )
)

if not exist "modeles" mkdir "modeles"
if not exist "%MODELE%" (
    echo [RontoGPT] Modele absent — telechargement prealable...
    start "/wait" "%ProgramFiles%\Git\usr\bin\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\installer.ps1" -Module "%MODELE%" -MinEspace %MIN_ESPACE% -Seal "%MODELE%"
    if not exist "%MODELE%" (
        echo [RontoGPT] Telechargement echoue.
        pause
        exit /b 1
    )
)

set "API_URL=http://127.0.0.1:8760/v1"
"tui\node_modules\bun\bin\bun.exe" "tui\src\index.ts"
if errorlevel 1 pause
