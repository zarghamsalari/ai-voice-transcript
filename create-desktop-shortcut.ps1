# Creates a desktop shortcut that launches the AI Voice Transcript GUI.
# Run once after cloning. Uses Windows' built-in WScript.Shell COM object,
# so no extra dependencies are needed.
#
# Usage:
#   cd <project folder>
#   .\create-desktop-shortcut.ps1

$ProjectDir   = $PSScriptRoot
$TargetBat    = Join-Path $ProjectDir 'Launch App.bat'
$DesktopDir   = [Environment]::GetFolderPath('Desktop')
$ShortcutPath = Join-Path $DesktopDir 'AI Voice Transcript.lnk'

if (-not (Test-Path $TargetBat)) {
    Write-Error "Cannot find 'Launch App.bat' in $ProjectDir. Run this script from inside the project folder."
    exit 1
}

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath       = $TargetBat
$Shortcut.WorkingDirectory = $ProjectDir
$Shortcut.Description      = 'AI Voice Transcript - local Whisper transcription (desktop GUI)'
# Use a built-in Windows icon (microphone/audio glyph from imageres.dll).
# If you have your own .ico file, change this to its full path.
$Shortcut.IconLocation     = "$env:SystemRoot\System32\imageres.dll,-1024"
$Shortcut.Save()

Write-Host "Created shortcut at: $ShortcutPath" -ForegroundColor Green
Write-Host "Double-click it from your desktop to launch the app." -ForegroundColor Green
