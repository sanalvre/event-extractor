# Stop anything listening on the default Event Timeline Extractor web ports (localhost).
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ports = 8765, 8766, 8767, 8768
foreach ($port in $ports) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        $pid = $c.OwningProcess
        Write-Host "Stopping PID $pid on port $port"
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
    }
}
Write-Host "Done. Ports 8765-8768 should be free."
