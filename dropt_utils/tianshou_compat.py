"""
Compatibility shim for tianshou's training utilities.

tianshou<=1.x exposed a module level ``offpolicy_trainer`` function,
while 2.x switched to the ``OffPolicyTrainer`` class.  Importing through
this module keeps the rest of the codebase agnostic to the installed
version.
"""

from __future__ import annotations

try:  # tianshou <= 1.x
    from tianshou.trainer import offpolicy_trainer as offpolicy_trainer  # type: ignore
except ImportError:  # tianshou >= 2.0
    from tianshou.trainer import OffPolicyTrainer as _OffPolicyTrainer

    def offpolicy_trainer(**kwargs):
        trainer = _OffPolicyTrainer(**kwargs)
        trainer.run()
        return trainer.results

