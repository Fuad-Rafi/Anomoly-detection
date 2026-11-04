# Real 25 fps Avenue demo:
#   1. fit AI-VAD on Avenue normal videos at gap 5 (25 fps -> 0.2 s)
#   2. recalibrate normalization buffers
#   3. calibrate thresholds on held-out Avenue normal footage
#   4. render a 25 fps annotated test clip and evaluate it
#
# Use -Resume to keep completed artifacts and skip stages whose outputs exist.

param(
    [switch]$Resume,
    [string]$Clip = "05",
    [string]$ImageThreshold = "",
    [string]$BoxThreshold = ""
)

$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"

$fitRoot = "results\aivad\avenue_gap5"
$calCsv = "results\fps\avenue_calibration\01.csv"
$calJson = "results\fps\avenue_calibration\phase_avenue.json"
$demoCsv = "results\fps\avenue_demo\$Clip\scores.csv"
$demoVideo = "results\videos\avenue_${Clip}_25fps_annotated.mp4"
$logdir = Join-Path $root "results\fps\logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null

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

function Format-Invariant([object]$Value) {
    return ([double]$Value).ToString([System.Globalization.CultureInfo]::InvariantCulture)
}

Run-Py "avenue_preflight" @(
    "-c",
    "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
)

# --- 1/2. Fit and recalibrate -------------------------------------------------
$modelCkpt = $null
if (Test-Path $fitRoot) {
    $modelCkpt = Get-ChildItem $fitRoot -Recurse -Filter "model.ckpt" -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}
$robustCkpt = if ($modelCkpt) { Join-Path (Split-Path $modelCkpt) "model_robust.ckpt" } else { $null }

if (-not ($Resume -and $robustCkpt -and (Test-Path $robustCkpt))) {
    Run-Py "avenue_fit" @(
        "src\fit_aivad_gap.py",
        "--dataset", "avenue",
        "--gap", "5",
        "--stride", "25",
        "--batch-size", "2",
        "--num-workers", "0",
        "--output-root", $fitRoot
    )
    $modelCkpt = Get-ChildItem $fitRoot -Recurse -Filter "model.ckpt" |
        Select-Object -First 1 -ExpandProperty FullName
    if (-not $modelCkpt) { throw "Fit finished but no model.ckpt was found under $fitRoot" }
    $robustCkpt = Join-Path (Split-Path $modelCkpt) "model_robust.ckpt"

    Run-Py "avenue_recalibrate" @(
        "src\recalibrate_aivad_norm.py",
        "--dataset", "avenue",
        "--gap", "5",
        "--stride", "25",
        "--batches", "80",
        "--checkpoint", $modelCkpt,
        "--output", $robustCkpt
    )
} else {
    Write-Host "=== skip fit/recalibrate (exists: $robustCkpt) ==="
}

if (-not (Test-Path $robustCkpt)) { throw "Missing robust checkpoint: $robustCkpt" }

# --- 3. Calibrate on Avenue normal footage ------------------------------------
if (-not ($Resume -and (Test-Path $calCsv) -and (Test-Path $calJson))) {
    Run-Py "avenue_calibration_score" @(
        "src\eval_aivad.py",
        "--dataset", "avenue",
        "--clip", "01",
        "--video", "data\avenue\training_videos\01.avi",
        "--source-fps", "25",
        "--emulate-fps", "25",
        "--target-dt", "0.2083",
        "--start-s", "0",
        "--end-s", "20",
        "--checkpoint", $robustCkpt,
        "--score-mode", "sum",
        "--min-area", "400",
        "--batch-size", "8",
        "--device", "cuda",
        "--output", $calCsv
    )
    Run-Py "avenue_calibration_thresholds" @(
        "src\calibrate_from_csv.py",
        "--csv", $calCsv,
        "--quantile", "80",
        "--motion-threshold", "0",
        "--output", $calJson
    )
} else {
    Write-Host "=== skip calibration (exists: $calCsv) ==="
}

$cal = Get-Content $calJson -Raw | ConvertFrom-Json
$imageThr = if ($ImageThreshold -ne "") { $ImageThreshold } else { Format-Invariant $cal.image_threshold }
$boxThr = if ($BoxThreshold -ne "") { $BoxThreshold } else { Format-Invariant $cal.box_threshold }
Write-Host "Avenue calibration: image=$imageThr box=$boxThr motion=0"

# --- 4. Render and evaluate clip 05 ------------------------------------------
if (-not ($Resume -and (Test-Path $demoCsv) -and (Test-Path $demoVideo))) {
    Run-Py "avenue_demo_render_$Clip" @(
        "src\eval_aivad.py",
        "--dataset", "avenue",
        "--clip", $Clip,
        "--video", "data\avenue\testing_videos\$Clip.avi",
        "--source-fps", "25",
        "--emulate-fps", "25",
        "--target-dt", "0.2083",
        "--checkpoint", $robustCkpt,
        "--score-mode", "sum",
        "--min-area", "400",
        "--image-threshold", $imageThr,
        "--box-threshold", $boxThr,
        "--motion-threshold", "0",
        "--cluster-radius", "60",
        "--focus-radius", "90",
        "--smooth-seconds", "1.5",
        "--focus-timeout-seconds", "3.0",
        "--cooldown-seconds", "3.0",
        "--track-window-seconds", "1.0",
        "--track-min-hits-seconds", "0.6",
        "--track-stop-hits-seconds", "1.0",
        "--batch-size", "8",
        "--device", "cuda",
        "--render-output", $demoVideo,
        "--render-fps", "25",
        "--output", $demoCsv
    )
} else {
    Write-Host "=== skip demo render (exists: $demoVideo) ==="
}

Run-Py "avenue_labels" @(
    "src\convert_avenue_labels.py",
    "--input", "data\avenue\ground_truth_demo\testing_label_mask",
    "--output", "data\avenue\testing_labels_npy"
)
Run-Py "avenue_eval_$Clip" @(
    "src\evaluate_people_csv.py",
    "--csv", $demoCsv,
    "--gt", "data\avenue\testing_labels_npy\$Clip.npy"
)

Set-Content -Path "results\fps\avenue_demo\$Clip\DONE.txt" -Value "Avenue 25 fps demo complete $(Get-Date -Format o)"
Write-Host ""
Write-Host "ALL DONE: $demoVideo"
