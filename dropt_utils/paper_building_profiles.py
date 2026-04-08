from __future__ import annotations

import os
import pickle
from typing import Dict, Iterable, Optional


def default_out_dir(root_dir: str, profile: str) -> str:
    if profile == "bcfixclean_smalloffice":
        return os.path.join(root_dir, "paperfigure_bcfixclean_smalloffice")
    if profile == "bcfixclean_officemedium_partial":
        return os.path.join(root_dir, "paperfigure_bcfixclean_officemedium_partial")
    return os.path.join(root_dir, "paperfigure")


def resolve_bcfixclean_smalloffice_run_dirs(
    log_root: str,
    required_matchers: Iterable[str],
) -> Dict[str, str]:
    return resolve_bcfixclean_building_run_dirs(
        log_root=log_root,
        building_type="OfficeSmall",
        weather_type="Hot_Dry",
        required_matchers=required_matchers,
        allow_partial=False,
    )


def resolve_bcfixclean_officemedium_run_dirs(
    log_root: str,
    required_matchers: Iterable[str],
    allow_partial: bool = True,
) -> Dict[str, str]:
    return resolve_bcfixclean_building_run_dirs(
        log_root=log_root,
        building_type="OfficeMedium",
        weather_type="Hot_Dry",
        required_matchers=required_matchers,
        allow_partial=allow_partial,
    )


def resolve_bcfixclean_building_run_dirs(
    log_root: str,
    building_type: str,
    weather_type: str,
    required_matchers: Iterable[str],
    allow_partial: bool = False,
) -> Dict[str, str]:
    target_building_type = building_type
    target_weather_type = weather_type
    all_dirs = [
        name
        for name in os.listdir(log_root)
        if os.path.isdir(os.path.join(log_root, name))
    ]
    resolved: Dict[str, str] = {}
    matched_runs: Dict[str, list[str]] = {matcher: [] for matcher in required_matchers}

    default_mpc_runs = sorted(
        name
        for name in all_dirs
        if name.startswith("default_mpc")
        and target_building_type in name
        and target_weather_type in name
    )
    if default_mpc_runs:
        resolved["default_mpc_latest"] = default_mpc_runs[-1]

    for name in all_dirs:
        path = os.path.join(log_root, name)
        args = _load_run_args(path)
        if not args:
            continue

        current_building_type = str(args.get("building_type") or "")
        current_weather_type = str(args.get("weather_type") or "")
        if current_building_type != target_building_type or current_weather_type != target_weather_type:
            continue

        algorithm = str(args.get("algorithm") or "")
        log_prefix = str(args.get("log_prefix") or "")
        backbone_variant = str(args.get("backbone_variant") or "")
        guidance_scale = _safe_float(args.get("guidance_scale"))

        matcher: Optional[str] = None
        if algorithm == "diffusion_fno_guided_bcfix_clean":
            matcher = "fno_guided_full" if guidance_scale > 0 else "fno_guided_noguide_align"
        elif algorithm == "diffusion_fno_guided_bcfix_clean_nores_noguide":
            matcher = "fno_guided_nores_noguide_align"
        elif algorithm == "diffusion_fno_guided_bcfix_clean_nores_guided":
            matcher = "fno_guided_nores"
        elif backbone_variant == "nores":
            matcher = "fno_guided_nores" if guidance_scale > 0 else "fno_guided_nores_noguide_align"
        elif algorithm == "diffusion_mlp_bcfix_clean" or log_prefix == "diffusion_mlp_bcfix_clean":
            matcher = "MLP"
        elif log_prefix == "sac_baseline_bcfixclean":
            matcher = "sac_baseline"
        elif log_prefix == "sac_baseline_mpc_bcfixclean":
            matcher = "sac_baseline_mpc"

        if matcher and matcher in matched_runs:
            matched_runs[matcher].append(name)

    for matcher, names in matched_runs.items():
        if names:
            resolved[matcher] = sorted(names)[-1]

    missing = sorted(matcher for matcher in required_matchers if matcher not in resolved)
    if missing and not allow_partial:
        raise FileNotFoundError(
            f"Missing bcfixclean {target_building_type} runs for: {missing}"
        )
    return resolved


def _load_run_args(run_dir: str) -> Dict[str, object]:
    metadata_path = os.path.join(run_dir, "paper_data", "paper_metadata.pkl")
    if not os.path.exists(metadata_path):
        return {}
    try:
        with open(metadata_path, "rb") as fh:
            obj = pickle.load(fh)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    args = obj.get("args")
    if not isinstance(args, dict):
        return {}
    return args


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
