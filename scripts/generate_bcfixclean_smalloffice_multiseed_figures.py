#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate the OfficeSmall bcfix-clean paper figures with three-seed aggregation.

Statistical figures are aggregated across the available seeds (42, 0, 1).
Representative trajectory figures remain single-seed because averaging
heterogeneous time windows across seeds is not physically meaningful.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.join(ROOT_DIR, "scripts")
DEFAULT_OUT_DIR = os.path.join(ROOT_DIR, "paperfigure_bcfixclean_smalloffice_multiseed")


def _run(command: List[str]) -> None:
    proc = subprocess.run(command, cwd=ROOT_DIR)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _write_readme(out_dir: str) -> None:
    text = """# OfficeSmall bcfix-clean multiseed figures

This directory contains the refreshed paper figures for the OfficeSmall
bcfix-clean setting.

Three-seed aggregated figures:
- compare_energy.pdf/png
- compare_violations.pdf/png
- compare_energy_violations.pdf/png
- compare_reward_curves.pdf/png
- compare_action_smoothness.pdf/png
- action_psd_compare.pdf/png
- critic_q_mc_return.pdf/png
- ablation_summary_heatmap.pdf/png
- smalloffice_physical_psd_compare.pdf/png

Representative single-seed figures retained on a canonical seed:
- temperature_trajectories_paper.pdf/png
- control_sequence_paper.pdf/png
- multizone_action_coordination.pdf/png
- temperature_trajectories_all_models.pdf/png
- control_sequence_all_models.pdf/png

Rationale:
- The aggregated figures summarize method-level behavior and benefit from a
  multi-seed estimate.
- The representative trajectory figures visualize a concrete 72-hour window.
  Averaging trajectories across different seeds would destroy their physical
  interpretation, so they are intentionally kept as representative examples.
"""
    path = os.path.join(out_dir, "README_multiseed.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OfficeSmall bcfix-clean multiseed paper figures.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory for the generated figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    py = sys.executable
    common = ["--profile", "bcfixclean_smalloffice", "--out-dir", out_dir]

    _run([py, os.path.join(SCRIPT_DIR, "paper_compare_energy_violations.py"), *common, "--aggregate-seeds"])
    _run([py, os.path.join(SCRIPT_DIR, "paper_building_figure_suite.py"), *common, "--aggregate-seeds"])
    _run([py, os.path.join(SCRIPT_DIR, "paper_mechanism_figures.py"), *common, "--aggregate-seeds"])
    _run([py, os.path.join(SCRIPT_DIR, "paper_physical_psd_compare.py"), "--out-dir", out_dir, "--aggregate-seeds"])
    _write_readme(out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
