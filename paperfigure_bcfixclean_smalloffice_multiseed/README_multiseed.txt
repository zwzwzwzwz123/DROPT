# OfficeSmall bcfix-clean multiseed figures

This directory contains the refreshed paper figures for the OfficeSmall
bcfix-clean setting.

Three-seed aggregated figures:
- compare_energy.pdf/png
- compare_violations.pdf/png
- compare_energy_violations.pdf/png
- compare_reward_curves.pdf/png
- compare_action_smoothness.pdf/png
- action_psd_compare.pdf/png
- ablation_summary_heatmap.pdf/png
- smalloffice_physical_psd_compare.pdf/png
- critic_q_mc_return_multiseed.pdf/png

Representative single-seed figures retained on a canonical seed:
- critic_q_mc_return.pdf/png
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
