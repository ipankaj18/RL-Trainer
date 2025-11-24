"""
Helper script to re-validate the Phase 2 action mask wiring.

This simply proxies to test_phase2_mask_fix.run_all_checks() so you can run
`python fix_phase2_masks.py` as a quick smoke test after modifying related code.
"""

from test_phase2_mask_fix import run_all_checks


if __name__ == "__main__":
    run_all_checks()
