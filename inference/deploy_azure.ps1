param(
    [Parameter(Mandatory = $true)]
    [string]$ResourceGroup,

    [string]$AppName = "shejian",
    [string]$SubscriptionId = "",
    [string]$AppRev = "2026-07-31-auth-1",

    # API 密钥 JSON，例如 '{"webapp-001":"sk-live-xxxx"}'
    [string]$ApiKeysJson = "",
    [string]$AuthEnabled = "true",
    [string]$RateLimitPerMinute = "60",
    [string]$TimestampWindow = "300",
    [string]$AllowedOrigins = ""
)

$ErrorActionPreference = "Stop"

Write-Host "[1/7] Check Azure CLI"
az version | Out-Null

if ($SubscriptionId -ne "") {
    Write-Host "[2/7] Switch subscription: $SubscriptionId"
    az account set --subscription $SubscriptionId
} else {
    Write-Host "[2/7] Use current subscription"
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Split-Path -Parent $scriptDir
$modelSource = Join-Path $workspaceRoot "best.onnx"

if (-not (Test-Path $modelSource)) {
    throw "Model file not found: $modelSource"
}

$publishDir = Join-Path $scriptDir "publish"
$zipPath = Join-Path $scriptDir "publish.zip"

Write-Host "[3/7] Build publish folder"
if (Test-Path $publishDir) { Remove-Item -Recurse -Force $publishDir }
if (Test-Path $zipPath) { Remove-Item -Force $zipPath }

New-Item -ItemType Directory -Path $publishDir | Out-Null
New-Item -ItemType Directory -Path (Join-Path $publishDir "models") | Out-Null

Copy-Item (Join-Path $scriptDir "app.py") (Join-Path $publishDir "app.py") -Force
Copy-Item (Join-Path $scriptDir "auth_utils.py") (Join-Path $publishDir "auth_utils.py") -Force
Copy-Item (Join-Path $scriptDir "requirements.txt") (Join-Path $publishDir "requirements.txt") -Force
Copy-Item (Join-Path $scriptDir "tongue_preprocess.py") (Join-Path $publishDir "tongue_preprocess.py") -Force
Copy-Item $modelSource (Join-Path (Join-Path $publishDir "models") "best.onnx") -Force

Write-Host "[4/7] Create zip package"
$pythonPack = @'
import os
import zipfile
import hashlib

publish_dir = os.environ["PUBLISH_DIR"]
zip_path = os.environ["ZIP_PATH"]
model_path = os.path.join(publish_dir, "models", "best.onnx")

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(publish_dir):
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, publish_dir)
            arc = rel.replace("\\", "/")
            zf.write(full, arc)

required = {"app.py", "auth_utils.py", "requirements.txt", "tongue_preprocess.py", "models/best.onnx"}
with zipfile.ZipFile(zip_path, "r") as zf:
    names = set(zf.namelist())
missing = sorted(required - names)
if missing:
    raise RuntimeError(f"ZIP missing required files: {missing}")

h = hashlib.sha256()
with open(model_path, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)

print(f"ZIP_OK {zip_path}")
print(f"ZIP_ENTRIES {len(names)}")
print(f"MODEL_SHA256 {h.hexdigest()}")
'@

$env:PUBLISH_DIR = $publishDir
$env:ZIP_PATH = $zipPath
$pythonPack | python -
Remove-Item Env:PUBLISH_DIR
Remove-Item Env:ZIP_PATH

Write-Host "[5/7] Configure app settings and startup command"
$appSettings = @(
    "SCM_DO_BUILD_DURING_DEPLOYMENT=true",
    "ENABLE_ORYX_BUILD=true",
    "MODEL_PATH=models/best.onnx",
    "INPUT_SIZE=640",
    "CONF_THRESHOLD=0.20",
    "NMS_THRESHOLD=0.50",
    "REPORT_CONF_THRESHOLD=0.30",
    "MAX_CONTENT_MB=6",
    "ENABLE_COLOR_CORRECTION=true",
    "APP_REV=$AppRev",
    "AUTH_ENABLED=$AuthEnabled",
    "RATE_LIMIT_PER_MINUTE=$RateLimitPerMinute",
    "TIMESTAMP_WINDOW=$TimestampWindow",
    "ALLOWED_ORIGINS=$AllowedOrigins"
)
if ($ApiKeysJson -ne "") {
    $appSettings += "API_KEYS_JSON=$ApiKeysJson"
}
az webapp config appsettings set `
    --resource-group $ResourceGroup `
    --name $AppName `
    --settings $appSettings | Out-Null

$startup = "gunicorn --bind=0.0.0.0 --workers 1 --timeout 180 --access-logfile - --error-logfile - --capture-output --log-level info app:app"

az webapp config set `
    --resource-group $ResourceGroup `
    --name $AppName `
    --startup-file $startup | Out-Null

Write-Host "[6/7] Deploy zip"
az webapp deploy --resource-group $ResourceGroup --name $AppName --src-path $zipPath --type zip

Write-Host "[7/7] Warmup and health check"
$hostName = az webapp show --resource-group $ResourceGroup --name $AppName --query defaultHostName -o tsv
if ([string]::IsNullOrWhiteSpace($hostName)) {
    $hostName = "$AppName.azurewebsites.net"
}
$healthUrl = "https://$hostName/health"
$ok = $false
for ($i = 1; $i -le 18; $i++) {
    Start-Sleep -Seconds 5
    try {
        $res = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 20
        if ($res.rev -eq $AppRev -and $res.model_loaded -eq $true) {
            $ok = $true
            Write-Host "Health check passed at attempt $i (rev=$($res.rev), model_loaded=$($res.model_loaded))"
            break
        }
        Write-Host "Health pending at attempt $i (rev=$($res.rev), model_loaded=$($res.model_loaded), model_error=$($res.model_error))"
    } catch {
        Write-Host "Health retry $i failed: $($_.Exception.Message)"
    }
}

if (-not $ok) {
    Write-Warning "Deploy finished but health check not fully ready. Check app logs for details."
}

Write-Host ""
Write-Host "Deploy finished"
Write-Host "Health URL: $healthUrl"
Write-Host "Predict URL: https://$hostName/predict"
