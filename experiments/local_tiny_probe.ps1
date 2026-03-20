param(
    [string]$RunId = $(if ($env:RUN_ID) { $env:RUN_ID } else { "local_tiny_probe" }),
    [string]$PythonExe = "D:\venvs\parameter-golf\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$defaults = @{
    DATA_PATH           = "./data/datasets/fineweb10B_sp1024"
    TOKENIZER_PATH      = "./data/tokenizers/fineweb_1024_bpe.model"
    VOCAB_SIZE          = "1024"
    NUM_LAYERS          = "9"
    NUM_SHARED_BLOCKS   = "3"
    NUM_SHARED_MLPS     = "3"
    MODEL_DIM           = "896"
    MLP_HIDDEN          = "2304"
    TRAIN_SEQ_LEN       = "1024"
    TRAIN_BATCH_TOKENS  = "16384"
    ITERATIONS          = "4"
    WARMUP_STEPS        = "2"
    WARMDOWN_ITERS      = "2"
    MAX_WALLCLOCK_SECONDS = "120"
    EVAL_STRIDE_TOKENS  = "64"
    EVAL_DOC_ISOLATED   = "1"
    VAL_MAX_TOKENS      = "131072"
    TRAIN_LOG_EVERY     = "1"
    DISABLE_COMPILE     = "1"
}

foreach ($entry in $defaults.GetEnumerator()) {
    $existing = [Environment]::GetEnvironmentVariable($entry.Key)
    if ([string]::IsNullOrEmpty($existing)) {
        Set-Item -Path "Env:$($entry.Key)" -Value $entry.Value
    }
}

$logsDir = Join-Path $repoRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$driverLog = Join-Path $logsDir "$RunId.driver.log"
$trainerLog = Join-Path $logsDir "$RunId.txt"
$statusFile = Join-Path $logsDir "$RunId.status"
$doneFile = Join-Path $logsDir "$RunId.done"
$failedFile = Join-Path $logsDir "$RunId.failed"

Remove-Item -Force -ErrorAction SilentlyContinue $driverLog, $trainerLog, $statusFile, $doneFile, $failedFile

@(
    "run_id=$RunId"
    "started_at=$(Get-Date -Format o)"
    "cwd=$repoRoot"
    "python=$PythonExe"
    "status=RUNNING"
) | Tee-Object -FilePath $driverLog | Set-Content -Path $statusFile

& $PythonExe train_gpt.py 2>&1 | Tee-Object -FilePath $trainerLog
$cmdStatus = $LASTEXITCODE
if ($null -eq $cmdStatus) {
    $cmdStatus = 0
}

$exactLine = $null
if (Test-Path $trainerLog) {
    $exactLine = Select-String -Path $trainerLog -Pattern "final_int8_zlib_roundtrip_exact" | Select-Object -Last 1
}

if (($cmdStatus -eq 0) -and $exactLine) {
    @(
        "finished_at=$(Get-Date -Format o)"
        "status=DONE"
        "exact=$($exactLine.Line)"
    ) | Tee-Object -FilePath $driverLog -Append | Set-Content -Path $statusFile
    Set-Content -Path $doneFile -Value $exactLine.Line
    exit 0
}

$tail = @()
if (Test-Path $trainerLog) {
    $tail = Get-Content $trainerLog -Tail 40
}

@(
    "finished_at=$(Get-Date -Format o)"
    "status=FAILED"
    "cmd_status=$cmdStatus"
    if ($tail.Count -gt 0) { "trainer_tail_start" }
    $tail
    if ($tail.Count -gt 0) { "trainer_tail_end" }
) | Tee-Object -FilePath $driverLog -Append | Set-Content -Path $statusFile

Set-Content -Path $failedFile -Value "cmd_status=$cmdStatus"
exit $cmdStatus
