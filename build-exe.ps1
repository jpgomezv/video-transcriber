# Builds Transcribir.exe (a small launcher that opens the GUI without a console window).
# Requires the .NET Framework compiler (ships with Windows).
$csc = "C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (-not (Test-Path $csc)) {
    Write-Error "csc.exe not found at $csc"
    exit 1
}
& $csc /nologo /target:winexe /out:"$PSScriptRoot\Transcribir.exe" `
    /reference:System.Windows.Forms.dll "$PSScriptRoot\launcher.cs"
if ($LASTEXITCODE -eq 0) {
    Write-Host "Transcribir.exe built."
} else {
    exit $LASTEXITCODE
}
