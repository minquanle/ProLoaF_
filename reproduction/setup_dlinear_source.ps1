$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceDirectory = Join-Path $projectRoot 'external\LTSF-Linear'
$sourceCommit = '0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6'
$createdCheckout = $false
if (-not (Test-Path -LiteralPath $sourceDirectory)) {
    git clone --depth 1 --filter=blob:none --sparse https://github.com/cure-lab/LTSF-Linear.git $sourceDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Could not clone DLinear source.' }
    $createdCheckout = $true
}
git -C $sourceDirectory cat-file -e "$sourceCommit^{commit}" 2>$null
if ($LASTEXITCODE -ne 0) {
    git -C $sourceDirectory fetch --depth 1 origin $sourceCommit
    if ($LASTEXITCODE -ne 0) { throw 'Could not fetch pinned DLinear commit.' }
}
$sourceHead = git -C $sourceDirectory rev-parse HEAD
if ($sourceHead -ne $sourceCommit) {
    if (-not $createdCheckout) {
        throw 'Checkout differs from pinned version. Inspect it manually; setup does not reset existing files.'
    }
    git -C $sourceDirectory checkout --detach $sourceCommit
    if ($LASTEXITCODE -ne 0) { throw 'Could not select pinned commit in new checkout.' }
}
git -C $sourceDirectory sparse-checkout set models exp utils
if ($LASTEXITCODE -ne 0) { throw 'Could not prepare sparse source.' }
Write-Output "DLinear source ready at $sourceDirectory ($sourceCommit)."
