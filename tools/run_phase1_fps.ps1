# Phase 0/1 frame-robustness run.
# Everything runs on CUDA. Logs go to results/fps/logs. The script stops on the
# first failure and writes results/fps/DONE.txt only after all phases succeed.
# Use -Resume to keep completed score CSVs and skip the golden guard.

param([switch]$Resume)

$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
$ckpt = "results\aivad\scene01_gap5\AiVad\ShanghaiTech\v0\weights\lightning\model_robust.ckpt"
$logdir = Join-Path $root "results\fps\logs"
New-Item -ItemType Directory -Force -Path $logdir, (Join-Path $root "results\fps\golden") | Out-Null

$calPath = Join-Path $root "results\fps\calibration\phase1.json"
if (-not (Test-Path $calPath)) {
    throw "Missing calibration $calPath; run src/calibrate_from_csv.py on a normal-footage CSV first"
}
$cal = Get-Content $calPath -Raw | ConvertFrom-Json
function Format-Invariant([object]$Value) {
    return ([double]$Value).ToString([System.Globalization.CultureInfo]::InvariantCulture)
}
$script:imageThr = Format-Invariant $cal.image_threshold
$script:boxThr = Format-Invariant $cal.box_threshold
$script:motionThr = Format-Invariant $cal.motion_threshold
Write-Host "calibration: image=$($script:imageThr) box=$($script:boxThr) motion=$($script:motionThr)"

function Run-Py {
    param([string]$Name, [string[]]$Arguments)
    $log = Join-Path $logdir "$Name.log"
    Write-Host ""
    Write-Host "=== $Name ==="
    $output = & $py @Arguments 2>&1
    $code = $LASTEXITCODE
    $output | Tee-Object -FilePath $log
    if ($code -ne 0) {
        throw "$Name failed with exit code $code (see $log)"
    }
}

function Invoke-Eval {
    param(
        [string]$Name,
        [string]$Dataset,
        [string]$Clip,
        [string]$EmulateFps,
        [string]$Dt,
        [string]$Out,
        [string]$StartS = "",
        [string]$EndS = "",
        [switch]$NoDedupe
    )
    $a = @(
        "src\eval_aivad.py",
        "--dataset", $Dataset,
        "--clip", $Clip,
        "--checkpoint", $ckpt,
        "--emulate-fps", $EmulateFps,
        "--target-dt", $Dt,
        "--batch-size", "8",
        "--device", "cuda",
        "--image-threshold", $script:imageThr,
        "--box-threshold", $script:boxThr,
        "--motion-threshold", $script:motionThr,
        "--relative-threshold", "0.5",
        "--score-mode", "sum",
        "--cluster-radius", "90",
        "--focus-radius", "140",
        "--smooth-seconds", "1.5",
        "--focus-timeout-seconds", "3.0",
        "--cooldown-seconds", "3.0",
        "--track-window-seconds", "1.0",
        "--track-min-hits-seconds", "0.6",
        "--track-stop-hits-seconds", "1.0",
        "--output", $Out
    )
    if ($Dataset -eq "shanghaitech") {
        $a += @("--source-fps", "10")
    }
    if ($StartS -ne "") { $a += @("--start-s", $StartS) }
    if ($EndS -ne "") { $a += @("--end-s", $EndS) }
    if ($NoDedupe) { $a += "--no-dedupe" }
    if ($Resume -and (Test-Path $Out)) {
        Write-Host ""
        Write-Host "=== skip $Name (exists: $Out) ==="
        return
    }
    Run-Py $Name $a
}

Run-Py "preflight" @(
    "-c",
    "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
)

# --- Golden guard: the refactored renderer must reproduce the legacy CSV. ---
if (-not $Resume) {
Run-Py "golden_legacy" @(
    "src\_legacy_render_people_video.py",
    "--checkpoint", $ckpt,
    "--scene", "01",
    "--clip", "01_0026",
    "--limit", "50",
    "--score-only",
    "--output", "results\fps\golden\legacy.mp4"
)
Run-Py "golden_new" @(
    "src\render_people_video.py",
    "--checkpoint", $ckpt,
    "--scene", "01",
    "--clip", "01_0026",
    "--limit", "50",
    "--score-only",
    "--output", "results\fps\golden\new.mp4"
)
Run-Py "golden_compare" @(
    "src\compare_golden.py",
    "--legacy", "results\fps\golden\legacy.scores.csv",
    "--new", "results\fps\golden\new.scores.csv"
)
}

# --- E-FPS-01: invariance at fixed Δt = 0.2083 s (ShanghaiTech 10 fps source). ---
# Full clips at 5 and 10 fps.
foreach ($fps in @("5", "10")) {
    Invoke-Eval `
        -Name "sh_01_0026_fps${fps}" `
        -Dataset "shanghaitech" `
        -Clip "01_0026" `
        -EmulateFps $fps `
        -Dt "0.2083" `
        -Out "results\fps\shanghaitech\01_0026\fps${fps}_dt0p2083\scores.csv"
}
# 20/30 fps cadence check. Pair inference is cached, so a full-clip run costs the
# same GPU time as the 10 fps run while the temporal clock runs at 20/30 Hz.
foreach ($fps in @("20", "30")) {
    Invoke-Eval `
        -Name "sh_01_0026_fps${fps}" `
        -Dataset "shanghaitech" `
        -Clip "01_0026" `
        -EmulateFps $fps `
        -Dt "0.2083" `
        -Out "results\fps\shanghaitech\01_0026\fps${fps}_dt0p2083\scores.csv" `
        -NoDedupe
}
# Second clip: 5 vs 10 fps.
foreach ($fps in @("5", "10")) {
    Invoke-Eval `
        -Name "sh_03_0061_fps${fps}" `
        -Dataset "shanghaitech" `
        -Clip "03_0061" `
        -EmulateFps $fps `
        -Dt "0.2083" `
        -Out "results\fps\shanghaitech\03_0061\fps${fps}_dt0p2083\scores.csv"
}

# --- E-FPS-02: real 25 fps Avenue source, compared with a 10 fps emulation. ---
foreach ($fps in @("25", "10")) {
    Invoke-Eval `
        -Name "avenue_01_fps${fps}" `
        -Dataset "avenue" `
        -Clip "01" `
        -EmulateFps $fps `
        -Dt "0.2083" `
        -StartS "0" `
        -EndS "30" `
        -Out "results\fps\avenue\01\fps${fps}_dt0p2083\scores.csv"
}

# --- E-FPS-03: sub-frame Δt at 10 fps must quantise to one frame (0.1 s). ---
Invoke-Eval `
    -Name "sh_01_0026_fps10_dt0p05" `
    -Dataset "shanghaitech" `
    -Clip "01_0026" `
    -EmulateFps "10" `
    -Dt "0.05" `
    -Out "results\fps\shanghaitech\01_0026\fps10_dt0p05\scores.csv"

# --- Compare runs. ---
$gt26 = "data\shanghaitech\testing\test_frame_mask\01_0026.npy"
$gt61 = "data\shanghaitech\testing\test_frame_mask\03_0061.npy"
$base26 = "results\fps\shanghaitech\01_0026"
$base61 = "results\fps\shanghaitech\03_0061"

Run-Py "invariance_01_0026" @(
    "src\evaluate_fps.py",
    "--reference", "fps10",
    "--run", "fps10=$base26\fps10_dt0p2083\scores.csv",
    "--run", "fps5=$base26\fps5_dt0p2083\scores.csv",
    "--run", "fps20=$base26\fps20_dt0p2083\scores.csv",
    "--run", "fps30=$base26\fps30_dt0p2083\scores.csv",
    "--gt", $gt26,
    "--fps", "10",
    "--output", "results\fps\invariance_01_0026.csv"
)
Run-Py "invariance_03_0061" @(
    "src\evaluate_fps.py",
    "--reference", "fps10",
    "--run", "fps10=$base61\fps10_dt0p2083\scores.csv",
    "--run", "fps5=$base61\fps5_dt0p2083\scores.csv",
    "--gt", $gt61,
    "--fps", "10",
    "--output", "results\fps\invariance_03_0061.csv"
)
Run-Py "invariance_avenue_01" @(
    "src\evaluate_fps.py",
    "--reference", "fps25",
    "--run", "fps25=results\fps\avenue\01\fps25_dt0p2083\scores.csv",
    "--run", "fps10=results\fps\avenue\01\fps10_dt0p2083\scores.csv",
    "--fps", "25",
    "--output", "results\fps\invariance_avenue_01.csv"
)
Run-Py "quantisation_01_0026" @(
    "src\evaluate_fps.py",
    "--reference", "dt0p2083",
    "--run", "dt0p2083=$base26\fps10_dt0p2083\scores.csv",
    "--run", "dt0p05=$base26\fps10_dt0p05\scores.csv",
    "--gt", $gt26,
    "--fps", "10",
    "--output", "results\fps\quantisation_01_0026.csv"
)

Set-Content -Path "results\fps\DONE.txt" -Value "Phase 0/1 complete $(Get-Date -Format o)"
Write-Host ""
Write-Host "ALL DONE"
