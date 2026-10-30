@echo off
chcp 65001 >nul
set PYTHONUTF8=1
rem Abre o menu de treino local usando o ambiente isolado (.venv-train).
cd /d "%~dp0"
if not exist ".venv-train\Scripts\python.exe" (
  echo Ambiente de treino nao encontrado. Crie com:
  echo   python -m venv .venv-train ^&^& .venv-train\Scripts\pip install -r requirements-train.txt
  pause
  exit /b 1
)
".venv-train\Scripts\python.exe" scripts\train_local.py %*
pause
