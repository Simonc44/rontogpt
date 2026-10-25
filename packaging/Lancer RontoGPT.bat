@echo off
rem RontoGPT - interface terminal (TUI).
rem %~dp0 = dossier de ce fichier : marche quelle que soit l'installation.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [RontoGPT] Cerveau non installe - lancement de l'installation...
    powershell -ExecutionPolicy Bypass -File "%~dp0scripts\installer.ps1" -Destination "%~dp0."
    if not exist ".venv\Scripts\python.exe" (
        echo [RontoGPT] Installation impossible. Voir les messages ci-dessus.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" -m aura %*
pause
