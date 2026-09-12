# Builds the windowless launchers (classic tkinter GUI + native Qt GUI).
# Requires the .NET Framework compiler (ships with Windows).
$csc = "C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (-not (Test-Path $csc)) {
    Write-Error "csc.exe not found at $csc"
    exit 1
}
& $csc /nologo /target:winexe /win32icon:"$PSScriptRoot\assets\app-icon.ico" /out:"$PSScriptRoot\Transcribir.exe" `
    /reference:System.Windows.Forms.dll "$PSScriptRoot\launcher.cs"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $csc /nologo /target:winexe /win32icon:"$PSScriptRoot\assets\app-icon.ico" /out:"$PSScriptRoot\Transcribir-qt.exe" `
    /reference:System.Windows.Forms.dll "$PSScriptRoot\launcher-qt.cs"
if ($LASTEXITCODE -eq 0) {
    Write-Host "Transcribir.exe + Transcribir-qt.exe built."
} else {
    exit $LASTEXITCODE
}
