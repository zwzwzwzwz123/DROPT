# OfficeSmall bcfix-clean multiseed figures (no-guidance main)

This directory contains the conference-oriented multiseed figures where the
no-guidance residual DiffFNO model is treated as the main method.

Generated figures:
- compare_energy_violations_noguide_main.pdf/png
- ablation_summary_heatmap_noguide_main.pdf/png
- smalloffice_physical_psd_compare_noguide_main.pdf/png

Method selection logic:
- Pareto figure: DiffFNO, DiffFNO w/o Residual, Diffusion-MLP, MPC, SAC, SAC+MPC
- Heatmap: DiffFNO, DiffFNO w/o Residual, Diffusion-MLP
- Physical PSD: Physical MPC, DiffFNO, Diffusion-MLP

Rationale:
- Guidance-based variants are intentionally excluded from the conference
  figure set so the narrative stays centered on the unguided DiffFNO method.
- The heatmap isolates the residual-branch trade-off inside the unguided
  family, which aligns with the conference claim.
