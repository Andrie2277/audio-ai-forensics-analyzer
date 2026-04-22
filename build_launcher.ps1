$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$iconPath = Join-Path $projectRoot "assets\audio_analyzer.ico"

& $pythonExe -m PyInstaller `
  --noconfirm `
  --clean `
  --windowed `
  --onefile `
  --name "AudioAIForensicsLauncher" `
  --icon $iconPath `
  --distpath $projectRoot `
  --workpath (Join-Path $projectRoot "build") `
  --specpath $projectRoot `
  (Join-Path $projectRoot "launcher.py")

Write-Host "Launcher selesai dibuat di: $projectRoot\AudioAIForensicsLauncher.exe"
