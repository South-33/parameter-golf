param(
    [string]$Checkpoint = "final_model.pt",
    [int]$SeqLen = 1024,
    [int]$ValMaxTokens = 131072,
    [int]$ValBatchSize = 16384,
    [int]$EvalStride = 64,
    [int]$EvalDocIsolated = 1,
    [string]$PythonExe = "D:\venvs\parameter-golf\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Invoke-Probe([string]$Label, [hashtable]$EnvVars) {
    $saved = @{}
    foreach ($k in $EnvVars.Keys) {
        $saved[$k] = [Environment]::GetEnvironmentVariable($k)
        Set-Item -Path "Env:$k" -Value ([string]$EnvVars[$k])
    }
    try {
        $output = & $PythonExe experiments\exporter_exact_probe.py `
            --checkpoint $Checkpoint `
            --seq-len $SeqLen `
            --val-max-tokens $ValMaxTokens `
            --val-batch-size $ValBatchSize `
            --eval-stride $EvalStride `
            --eval-doc-isolated $EvalDocIsolated `
            --device cuda
        if ($LASTEXITCODE -ne 0) {
            throw "probe failed for $Label"
        }
        $exact = $output | Select-String -Pattern '^exact bytes=(\d+) val_loss=([0-9.]+) val_bpb=([0-9.]+)'
        if (-not $exact) {
            throw "could not parse exact line for $Label"
        }
        [pscustomobject]@{
            label = $Label
            bytes = [int64]$exact.Matches[0].Groups[1].Value
            val_loss = [double]$exact.Matches[0].Groups[2].Value
            val_bpb = [double]$exact.Matches[0].Groups[3].Value
        }
    }
    finally {
        foreach ($k in $EnvVars.Keys) {
            if ([string]::IsNullOrEmpty($saved[$k])) {
                Remove-Item "Env:$k" -ErrorAction SilentlyContinue
            } else {
                Set-Item -Path "Env:$k" -Value $saved[$k]
            }
        }
    }
}

$variants = @(
    @{ label = "plain"; env = @{ INT8_SCALE_REPARAM_KIND = "none"; INT8_CENTER_ROWS = "0"; INT8_ACTIVATION_REPARAM_KIND = "none"; INT8_ACTIVATION_REPARAM_CALIB_BATCHES = "0"; INT8_GPTQ_TARGET = "none"; INT8_GPTQ_CALIB_BATCHES = "0" } },
    @{ label = "vproj"; env = @{ INT8_SCALE_REPARAM_KIND = "vproj"; INT8_CENTER_ROWS = "0"; INT8_ACTIVATION_REPARAM_KIND = "none"; INT8_ACTIVATION_REPARAM_CALIB_BATCHES = "0"; INT8_GPTQ_TARGET = "none"; INT8_GPTQ_CALIB_BATCHES = "0" } },
    @{ label = "activation_vproj"; env = @{ INT8_SCALE_REPARAM_KIND = "none"; INT8_CENTER_ROWS = "0"; INT8_ACTIVATION_REPARAM_KIND = "vproj"; INT8_ACTIVATION_REPARAM_ALPHA = "0.5"; INT8_ACTIVATION_REPARAM_CALIB_BATCHES = "2"; INT8_GPTQ_TARGET = "none"; INT8_GPTQ_CALIB_BATCHES = "0" } },
    @{ label = "center_rows"; env = @{ INT8_SCALE_REPARAM_KIND = "none"; INT8_CENTER_ROWS = "1"; INT8_ACTIVATION_REPARAM_KIND = "none"; INT8_ACTIVATION_REPARAM_CALIB_BATCHES = "0"; INT8_GPTQ_TARGET = "none"; INT8_GPTQ_CALIB_BATCHES = "0" } },
    @{ label = "gptq_attn_proj"; env = @{ INT8_SCALE_REPARAM_KIND = "none"; INT8_CENTER_ROWS = "0"; INT8_ACTIVATION_REPARAM_KIND = "none"; INT8_ACTIVATION_REPARAM_CALIB_BATCHES = "0"; INT8_GPTQ_TARGET = "attn_proj"; INT8_GPTQ_CALIB_BATCHES = "1"; INT8_GPTQ_DAMP = "0.01" } },
    @{ label = "gptq_attn_vproj"; env = @{ INT8_SCALE_REPARAM_KIND = "none"; INT8_CENTER_ROWS = "0"; INT8_ACTIVATION_REPARAM_KIND = "none"; INT8_ACTIVATION_REPARAM_CALIB_BATCHES = "0"; INT8_GPTQ_TARGET = "attn_vproj"; INT8_GPTQ_CALIB_BATCHES = "1"; INT8_GPTQ_DAMP = "0.01" } }
)

$results = foreach ($variant in $variants) {
    Invoke-Probe -Label $variant.label -EnvVars $variant.env
}

$baseline = $results | Where-Object { $_.label -eq "plain" } | Select-Object -First 1
if (-not $baseline) {
    throw "baseline result missing"
}

$results |
    Sort-Object val_bpb |
    Select-Object label, bytes, val_loss, val_bpb,
        @{ Name = "delta_bpb"; Expression = { [math]::Round($_.val_bpb - $baseline.val_bpb, 8) } },
        @{ Name = "delta_bytes"; Expression = { $_.bytes - $baseline.bytes } } |
    Format-Table -AutoSize
