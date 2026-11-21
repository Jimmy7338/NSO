"""Eligibility for existing native SDK zero-only CAN-mode initialization.

This is not motion admission: original supervisor checks and adapter arm remain
mandatory after CAN mode is established. No ROS, CAN socket or driver import.
"""
import math


def standby_native_eligibility(snapshot, now, *, execute):
    """Fresh normal STANDBY and >=1s actual zero in an unused execute session."""
    if execute is not True:
        return False, 'execute_session_required'
    if snapshot.get('ever_armed') is not False or snapshot.get('fault') is not None:
        return False, 'unused_fault_free_session_required'
    receipt=snapshot.get('receipt')
    if not isinstance(now,(int,float)) or not math.isfinite(now):
        return False, 'invalid_clock'
    def fresh(source, ttl):
        return isinstance(source,(int,float)) and math.isfinite(source) and 0<=now-source<=ttl
    if not fresh(receipt,.5):
        return False, 'adapter_telemetry_stale'
    status,velocity=snapshot.get('can_status'),snapshot.get('can_velocity')
    if not isinstance(status,dict) or not fresh(status.get('source'),.5):
        return False, 'chassis_status_stale'
    if (status.get('state'),status.get('mode'),status.get('error'))!=(0,0,0):
        return False, 'normal_standby_required'
    if not isinstance(velocity,dict) or not fresh(velocity.get('source'),.15):
        return False, 'chassis_velocity_stale'
    v,w=velocity.get('v'),velocity.get('w')
    if any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in (v,w)) or max(abs(v),abs(w))>.005:
        return False, 'actual_zero_velocity_required'
    zero=snapshot.get('can_zero_since')
    if not isinstance(zero,(int,float)) or not math.isfinite(zero) or now-zero<1.:
        return False, 'one_second_actual_zero_required'
    return True, 'fresh_standby_zero_only_native_handshake'


class NativeStartOnce:
    """A single existing native launch, early for eligible standby or normally."""
    def __init__(self, launch):
        self.launch=launch
        self.started=False
        self.context=None

    def start(self, context):
        if self.started:
            return False
        self.launch()
        self.started=True
        self.context=context
        return True

    def consider_standby(self, snapshot, now, *, execute):
        if self.started:
            return False
        eligible,reason=standby_native_eligibility(snapshot,now,execute=execute)
        return self.start(reason) if eligible else False
