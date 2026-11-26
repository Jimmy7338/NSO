"""Explicit external execution time profiles; independent of research policy."""
import math

STANDARD = 'standard_600s_v1'
LONG = 'long_budget_1200s_v1'
PROFILES = (STANDARD, LONG)


def validate_duration(duration_s, profile=STANDARD, mission_mode=None):
    if profile not in PROFILES:
        raise ValueError('unknown duration profile')
    if profile == LONG and mission_mode not in ('nso', 'joint'):
        raise ValueError('long duration profile requires explicit nso/joint mode')
    maximum = 1200. if profile == LONG else 600.
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)) or not math.isfinite(duration_s):
        raise ValueError('finite numeric duration required')
    if not 1. <= duration_s <= maximum:
        raise ValueError('duration outside selected profile bounds')
    return dict(profile=profile, duration_s=duration_s, maximum_duration_s=maximum,
                mission_mode=mission_mode, adapter_deadline_starts_at_arm=True,
                supervisor_deadline_uses_run_start=True,
                historical_40_action_equivalence=False if profile == LONG else None)
