param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$IterationsList = @("4", "8", "12"),
    [string]$RunPrefix = "exporter_depth_sweep",
    [string]$PythonExe = "D:\venvs\parameter-golf\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$IterationsList = @(
    $IterationsList |
        ForEach-Object { $_ -split '[,\s]+' } |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
)

$logsDir = Join-Path $repoRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$summaryPath = Join-Path $logsDir "$RunPrefix.summary.txt"
Remove-Item -Force -ErrorAction SilentlyContinue $summaryPath

function Set-Or-ClearEnv([string]$Name, [string]$Value) {
    if ([string]::IsNullOrEmpty($Value)) {
        Remove-Item "Env:$Name" -ErrorAction SilentlyContinue
    } else {
        Set-Item -Path "Env:$Name" -Value $Value
    }
}

function Parse-ProbeTableRow([string]$Line) {
    $parts = ($Line -replace '^\s+|\s+$', '') -split '\s+'
    if ($parts.Length -lt 6) {
        throw "Could not parse sweep row: $Line"
    }
    [pscustomobject]@{
        label = $parts[0]
        bytes = [int64]$parts[1]
        val_loss = [double]$parts[2]
        val_bpb = [double]$parts[3]
        delta_bpb = [double]$parts[4]
        delta_bytes = [int64]$parts[5]
    }
}

$results = @()

foreach ($iter in $IterationsList) {
    $runId = "$RunPrefix.iter$iter"
    $warmdown = if ([int]$iter -le 4) { "2" } else { [string]([math]::Max(4, [int]([math]::Floor([int]$iter / 2)))) }

    Set-Or-ClearEnv "ITERATIONS" $iter
    Set-Or-ClearEnv "WARMUP_STEPS" "2"
    Set-Or-ClearEnv "WARMDOWN_ITERS" $warmdown
    Set-Or-ClearEnv "MAX_WALLCLOCK_SECONDS" "240"
    Set-Or-ClearEnv "MUON_WEIGHT_DECAY" ""
    Set-Or-ClearEnv "TIED_EMB_FP32_MASTER" ""
    Set-Or-ClearEnv "OVERTONE_EMBED_INIT" ""
    Set-Or-ClearEnv "RESID_MIX_PHASE_INIT" ""
    Set-Or-ClearEnv "INT8_SCALE_REPARAM_KIND" ""
    Set-Or-ClearEnv "INT8_ACTIVATION_REPARAM_KIND" ""
    Set-Or-ClearEnv "INT8_CENTER_ROWS" ""
    Set-Or-ClearEnv "INT8_GPTQ_TARGET" ""

    powershell -ExecutionPolicy Bypass -File .\experiments\local_tiny_probe.ps1 -RunId $runId | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "local_tiny_probe failed for iterations=$iter"
    }

    $trainStatus = Get-Content (Join-Path $logsDir "$runId.status")
    $trainExactLine = ($trainStatus | Select-String -Pattern '^exact=' | Select-Object -First 1).Line
    $trainExactMatch = [regex]::Match($trainExactLine, 'val_bpb:([0-9.]+)$')
    if (-not $trainExactMatch.Success) {
        throw "Could not parse train exact line for iterations=$iter"
    }
    $trainExactBpb = [double]$trainExactMatch.Groups[1].Value

    $sweepOutput = powershell -ExecutionPolicy Bypass -File .\experiments\exporter_sweep.ps1
    if ($LASTEXITCODE -ne 0) {
        throw "exporter_sweep failed for iterations=$iter"
    }
    $sweepLog = Join-Path $logsDir "$runId.exporter_sweep.txt"
    $sweepOutput | Set-Content -Path $sweepLog

    $tableLines = $sweepOutput | Where-Object {
        $_ -match '^\s*(plain|vproj|activation_vproj|gptq_attn_proj|gptq_attn_vproj|center_rows)\s+'
    }
    if (-not $tableLines) {
        throw "Could not find exporter sweep table for iterations=$iter"
    }

    foreach ($line in $tableLines) {
        $row = Parse-ProbeTableRow $line
        $results += [pscustomobject]@{
            iterations = [int]$iter
            train_exact_bpb = $trainExactBpb
            label = $row.label
            bytes = $row.bytes
            val_loss = $row.val_loss
            val_bpb = $row.val_bpb
            delta_bpb = $row.delta_bpb
            delta_bytes = $row.delta_bytes
        }
    }
}

$table = $results |
    Sort-Object iterations, val_bpb |
    Select-Object iterations, train_exact_bpb, label, bytes, val_loss, val_bpb, delta_bpb, delta_bytes

$tableText = $table | Format-Table -AutoSize | Out-String -Width 240
$tableText | Tee-Object -FilePath $summaryPath
