$ErrorActionPreference = 'Stop'
$Build = $false
$Gpu = $false
$Arguments = [System.Collections.Generic.List[string]]::new()
foreach ($Argument in $args) { $Arguments.Add([string]$Argument) }

while ($Arguments.Count -gt 0) {
    if ($Arguments[0] -in @('-Build', '--build')) {
        $Build = $true
        $Arguments.RemoveAt(0)
    } elseif ($Arguments[0] -in @('-Gpu', '--gpu')) {
        $Gpu = $true
        $Arguments.RemoveAt(0)
    } else {
        break
    }
}

if ($Arguments.Count -lt 1) {
    throw 'Kullanım: run.ps1 [-Build] [-Gpu] src/<modul>.py [argümanlar...]'
}

$ScriptPath = $Arguments[0]
$ScriptArgs = @()
if ($Arguments.Count -gt 1) {
    $ScriptArgs = $Arguments.GetRange(1, $Arguments.Count - 1).ToArray()
}

$ProjectDir = (Resolve-Path -LiteralPath $PSScriptRoot).ProviderPath
$Image = if ($env:PHD_NET_IMAGE) { $env:PHD_NET_IMAGE } else { 'localhost/phd-net-dev:latest' }

if (-not (Get-Command podman -ErrorAction SilentlyContinue)) {
    throw 'Podman bulunamadı. Podman Desktop kurulumunu ve Podman makinesinin çalıştığını kontrol edin.'
}

$ProjectDatasets = Join-Path $ProjectDir 'datasets'
$LegacyDatasets = Join-Path (Split-Path $ProjectDir -Parent) 'datasets'
if ($env:PHD_NET_DATASETS) {
    $DatasetsHost = $env:PHD_NET_DATASETS
} elseif (Test-Path -LiteralPath $ProjectDatasets -PathType Container) {
    $DatasetsHost = $ProjectDatasets
} elseif (Test-Path -LiteralPath $LegacyDatasets -PathType Container) {
    $DatasetsHost = $LegacyDatasets
} else {
    $DatasetsHost = $ProjectDatasets
}

if ($env:PHD_NET_DATASETS -and -not (Test-Path -LiteralPath $DatasetsHost -PathType Container)) {
    throw "PHD_NET_DATASETS dizini bulunamadı: $DatasetsHost"
}
if (Test-Path -LiteralPath $DatasetsHost -PathType Container) {
    $DatasetsHost = (Resolve-Path -LiteralPath $DatasetsHost).ProviderPath
}

$ImageExists = $false
& podman image exists $Image
if ($LASTEXITCODE -eq 0) { $ImageExists = $true }

if ($Build -or -not $ImageExists) {
    & podman build --file (Join-Path $ProjectDir 'Containerfile') --tag $Image $ProjectDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$PodmanArgs = @(
    'run', '--rm',
    '--volume', "${ProjectDir}:/workspace",
    '--workdir', '/workspace',
    '--env', 'PYTHONPATH=/workspace/src:/workspace'
)

if (Test-Path -LiteralPath $DatasetsHost -PathType Container) {
    $PodmanArgs += @('--volume', "${DatasetsHost}:/datasets:ro")
    $DatasetsContainer = '/datasets'
} else {
    $DatasetsContainer = '/workspace/datasets'
}
$PodmanArgs += @('--env', "PHD_NET_DATASETS=$DatasetsContainer")

if ($Gpu) {
    $PodmanArgs += @('--device', 'nvidia.com/gpu=all')
}

$ContainerScript = $ScriptPath.Replace('\', '/')
$PodmanArgs += @($Image, 'python3', $ContainerScript)
if ($ScriptArgs) { $PodmanArgs += $ScriptArgs }

& podman @PodmanArgs
exit $LASTEXITCODE
