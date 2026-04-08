param(
    [int[]]$Seeds = @(0, 1)
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$logRoot = Join-Path $root "log_building"

$runs = @(
    @{
        Name = "Guided-DiffFNO"
        Script = "main_building_fno_guided_bcfix_clean.py"
        BaseLogPrefix = "diffusion_fno_guided_bcfix_clean"
        Tag = "guided"
        ExtraArgs = @("--guidance-scale", "0.5")
    },
    @{
        Name = "DiffFNO w/o Guidance"
        Script = "main_building_fno_guided_bcfix_clean.py"
        BaseLogPrefix = "diffusion_fno_guided_bcfix_clean"
        Tag = "noguide"
        ExtraArgs = @()
    },
    @{
        Name = "DiffFNO w/o Residual"
        Script = "main_building_fno_guided_bcfix_clean_ablation.py"
        BaseLogPrefix = "diffusion_fno_guided_bcfix_clean_nores_guided"
        Tag = "nores_guided"
        ExtraArgs = @("--backbone-variant", "nores", "--guidance-scale", "0.5")
    },
    @{
        Name = "DiffFNO w/o Residual & Guidance"
        Script = "main_building_fno_guided_bcfix_clean_ablation.py"
        BaseLogPrefix = "diffusion_fno_guided_bcfix_clean_nores_noguide"
        Tag = "nores_noguide"
        ExtraArgs = @("--backbone-variant", "nores")
    },
    @{
        Name = "Diffusion-MLP"
        Script = "main_building_bcfix_clean.py"
        BaseLogPrefix = "diffusion_mlp_bcfix_clean"
        Tag = "mlp"
        ExtraArgs = @()
    },
    @{
        Name = "SAC"
        Script = "rl_baseline_bcfixclean.py"
        BaseLogPrefix = "sac_baseline_bcfixclean"
        Tag = "sac"
        ExtraArgs = @()
    },
    @{
        Name = "SAC+MPC"
        Script = "rl_baseline_mpc_bcfixclean.py"
        BaseLogPrefix = "sac_baseline_mpc_bcfixclean"
        Tag = "sac_mpc"
        ExtraArgs = @()
    }
)

function Get-MatchingLogDirs {
    param(
        [string]$BaseLogPrefix
    )

    if (-not (Test-Path $logRoot)) {
        return @()
    }

    return @(Get-ChildItem -Path $logRoot -Directory | Where-Object {
        $_.Name.StartsWith($BaseLogPrefix + "_OfficeSmall_Hot_Dry_")
    } | Select-Object -ExpandProperty Name)
}

foreach ($seed in $Seeds) {
    Write-Host ""
    Write-Host ("=" * 72)
    Write-Host "Running bcfixclean OfficeSmall multi-seed suite for seed=$seed"
    Write-Host ("=" * 72)

    foreach ($run in $runs) {
        $before = Get-MatchingLogDirs -BaseLogPrefix $run.BaseLogPrefix
        $args = @($run.Script, "--seed", "$seed") + $run.ExtraArgs
        Write-Host ""
        Write-Host ("[{0}] seed={1}" -f $run.Name, $seed)
        Write-Host ("python " + ($args -join " "))
        & python @args
        if ($LASTEXITCODE -ne 0) {
            throw ("Run failed: {0} seed={1}" -f $run.Name, $seed)
        }

        $after = Get-MatchingLogDirs -BaseLogPrefix $run.BaseLogPrefix
        $newDir = @($after | Where-Object { $_ -notin $before } | Select-Object -Last 1)
        if (-not $newDir) {
            $newDir = @(
                Get-ChildItem -Path $logRoot -Directory |
                Where-Object { $_.Name.StartsWith($run.BaseLogPrefix + "_OfficeSmall_Hot_Dry_") } |
                Sort-Object LastWriteTime |
                Select-Object -Last 1 -ExpandProperty Name
            )
        }
        if (-not $newDir) {
            throw ("Could not locate newly created log directory for {0} seed={1}" -f $run.Name, $seed)
        }

        $sourceName = [string]$newDir[0]
        $suffix = "__" + $run.Tag + "_seed" + $seed
        if (-not $sourceName.EndsWith($suffix)) {
            $targetName = $sourceName + $suffix
            Rename-Item -LiteralPath (Join-Path $logRoot $sourceName) -NewName $targetName
            Write-Host ("Renamed log dir -> {0}" -f $targetName)
        } else {
            Write-Host ("Log dir already tagged -> {0}" -f $sourceName)
        }
    }
}

Write-Host ""
Write-Host "All requested bcfixclean OfficeSmall multi-seed runs finished."
