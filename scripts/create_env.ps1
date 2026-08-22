param(
    [string]$CondaPath = "C:\ProgramData\anaconda3\Scripts\conda.exe",
    [string]$EnvironmentName = "singing-vocoders-refactor"
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$CondaLock = Join-Path $RepositoryRoot "conda-win-64.lock"
$PythonLock = Join-Path $RepositoryRoot "pylock.win-64.toml"

if (-not (Test-Path -LiteralPath $CondaPath -PathType Leaf)) {
    throw "Conda executable does not exist: $CondaPath"
}

$EnvironmentList = & $CondaPath env list --json | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect Conda environments"
}
$ExistingNames = $EnvironmentList.envs | ForEach-Object { Split-Path -Leaf $_ }
if ($ExistingNames -contains $EnvironmentName) {
    throw "Conda environment already exists and will not be modified: $EnvironmentName"
}

$EnvironmentCreated = $false
try {
    & $CondaPath create --name $EnvironmentName --file $CondaLock --yes
    if ($LASTEXITCODE -ne 0) {
        throw "Conda failed to create the locked environment"
    }
    $EnvironmentCreated = $true
    & $CondaPath run --name $EnvironmentName python `
        (Join-Path $RepositoryRoot "scripts\install_pylock.py") `
        $PythonLock `
        $RepositoryRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Python lock installation failed"
    }
}
catch {
    if ($EnvironmentCreated) {
        & $CondaPath env remove --name $EnvironmentName --yes
    }
    throw
}
