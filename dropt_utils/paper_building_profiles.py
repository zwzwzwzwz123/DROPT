from __future__ import annotations

import os
import pickle
from typing import Dict, Iterable, Optional, Sequence


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
    groups = resolve_bcfixclean_building_run_dir_groups(
        log_root=log_root,
        building_type=building_type,
        weather_type=weather_type,
        required_matchers=required_matchers,
        allow_partial=allow_partial,
    )
    resolved: Dict[str, str] = {}
    for matcher, names in groups.items():
        if not names:
            continue
        resolved[matcher] = sorted(names)[-1]
    return resolved


def resolve_bcfixclean_smalloffice_run_dir_groups(
    log_root: str,
    required_matchers: Iterable[str],
) -> Dict[str, list[str]]:
    return resolve_bcfixclean_building_run_dir_groups(
        log_root=log_root,
        building_type="OfficeSmall",
        weather_type="Hot_Dry",
        required_matchers=required_matchers,
        allow_partial=False,
    )


def resolve_bcfixclean_officemedium_run_dir_groups(
    log_root: str,
    required_matchers: Iterable[str],
    allow_partial: bool = True,
) -> Dict[str, list[str]]:
    return resolve_bcfixclean_building_run_dir_groups(
        log_root=log_root,
        building_type="OfficeMedium",
        weather_type="Hot_Dry",
        required_matchers=required_matchers,
        allow_partial=allow_partial,
    )


def resolve_bcfixclean_building_run_dir_groups(
    log_root: str,
    building_type: str,
    weather_type: str,
    required_matchers: Iterable[str],
    allow_partial: bool = False,
) -> Dict[str, list[str]]:
    target_building_type = building_type
    target_weather_type = weather_type
    required = list(required_matchers)
    all_dirs = [
        name
        for name in os.listdir(log_root)
        if os.path.isdir(os.path.join(log_root, name))
    ]
    resolved: Dict[str, str] = {}
    matched_runs: Dict[str, list[str]] = {matcher: [] for matcher in required}

    default_mpc_runs = sorted(
        name
        for name in all_dirs
        if name.startswith("default_mpc")
        and target_building_type in name
        and target_weather_type in name
    )
    if default_mpc_runs:
        matched_runs["default_mpc_latest"] = [default_mpc_runs[-1]]

    for name in all_dirs:
        path = os.path.join(log_root, name)
        args = _load_run_args(path)
        if not args:
            continue

        current_building_type = str(args.get("building_type") or "")
        current_weather_type = str(args.get("weather_type") or "")
        if current_building_type != target_building_type or current_weather_type != target_weather_type:
            continue

        matcher = _match_bcfixclean_run(args)

        if matcher and matcher in matched_runs:
            matched_runs[matcher].append(name)

    missing = sorted(matcher for matcher in required if not matched_runs.get(matcher))
    if missing and not allow_partial:
        raise FileNotFoundError(
            f"Missing bcfixclean {target_building_type} runs for: {missing}"
        )
    return {
        matcher: sorted(names)
        for matcher, names in matched_runs.items()
        if names
    }


def pick_run_dirs_by_seed_preference(
    log_root: str,
    run_groups: Dict[str, Sequence[str]],
    preferred_seeds: Sequence[int] = (42, 0, 1),
) -> Dict[str, str]:
    selected: Dict[str, str] = {}
    for matcher, names in run_groups.items():
        if not names:
            continue
        chosen: Optional[str] = None
        for seed in preferred_seeds:
            seeded = [
                name for name in names
                if _seed_for_run_dir(os.path.join(log_root, name)) == seed
            ]
            if seeded:
                chosen = sorted(seeded)[-1]
                break
        if chosen is None:
            chosen = sorted(names)[-1]
        selected[matcher] = chosen
    return selected


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


def _match_bcfixclean_run(args: Dict[str, object]) -> Optional[str]:
    algorithm = str(args.get("algorithm") or "")
    log_prefix = str(args.get("log_prefix") or "")
    backbone_variant = str(args.get("backbone_variant") or "")
    guidance_scale = _safe_float(args.get("guidance_scale"))

    if algorithm == "diffusion_fno_guided_bcfix_clean":
        return "fno_guided_full" if guidance_scale > 0 else "fno_guided_noguide_align"
    if algorithm == "diffusion_fno_guided_bcfix_clean_nores_noguide":
        return "fno_guided_nores_noguide_align"
    if algorithm == "diffusion_fno_guided_bcfix_clean_nores_guided":
        return "fno_guided_nores"
    if backbone_variant == "nores":
        return "fno_guided_nores" if guidance_scale > 0 else "fno_guided_nores_noguide_align"
    if algorithm == "diffusion_mlp_bcfix_clean" or log_prefix == "diffusion_mlp_bcfix_clean":
        return "MLP"
    if log_prefix == "sac_baseline_bcfixclean":
        return "sac_baseline"
    if log_prefix == "sac_baseline_mpc_bcfixclean":
        return "sac_baseline_mpc"
    return None


def _seed_for_run_dir(run_dir: str) -> Optional[int]:
    args = _load_run_args(run_dir)
    seed = args.get("seed")
    try:
        return int(seed) if seed is not None else None
    except (TypeError, ValueError):
        return None
