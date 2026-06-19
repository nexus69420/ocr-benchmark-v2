# Sync OCR benchmark v2 golden set + run to H100, then run Chandra + Qwen.
# Run from laptop PowerShell (password prompts).
#
#   .\evals\ocr_benchmark_v2\scripts\sync_v2_to_h100.ps1
#   .\evals\ocr_benchmark_v2\scripts\sync_v2_to_h100.ps1 -RunId 20260618T103056Z

param(
    [string]$Jump = "amul-vm3-uintele",
    [string]$Gpu = "aicloud@10.185.25.197",
    [string]$RunId = "20260618T103056Z",
    [string]$Root = "c:\amul-oan-api"
)

$RemoteRepo = "~/amul-oan-api"

Write-Host "=== Sync v2 code + golden set to H100 ==="
ssh -o ProxyJump=$Jump $Gpu @"
mkdir -p $RemoteRepo/ocr_benchmark_v2/golden_set/assets/images
mkdir -p $RemoteRepo/runs/$RunId
"@

scp -o ProxyJump=$Jump "$Root\evals\__init__.py" "${Gpu}:${RemoteRepo}/evals/"
scp -o ProxyJump=$Jump -r "$Root\evals\ocr_benchmark_v2\*" "${Gpu}:${RemoteRepo}/ocr_benchmark_v2/"

if (Test-Path "$Root\eval_outputs\ocr_benchmark_v2\runs\$RunId") {
    scp -o ProxyJump=$Jump -r "$Root\eval_outputs\ocr_benchmark_v2\runs\$RunId\*" `
        "${Gpu}:${RemoteRepo}/runs/$RunId/"
}

Write-Host ""
Write-Host "=== SSH and run (models on /amulpfsdata) ==="
Write-Host "  ssh -o ProxyJump=$Jump $Gpu"
Write-Host "  cd ~/amul-oan-api"
Write-Host "  export CUDA_VISIBLE_DEVICES=6"
Write-Host "  bash ocr_benchmark_v2/scripts/run_h100_pending.sh $RunId"
Write-Host ""
Write-Host "=== Pull results back ==="
Write-Host "  scp -o ProxyJump=$Jump -r ${Gpu}:${RemoteRepo}/runs/$RunId $Root\eval_outputs\ocr_benchmark_v2\runs\"
