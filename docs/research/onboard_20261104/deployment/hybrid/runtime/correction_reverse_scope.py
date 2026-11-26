"""Opt-in execution-only same-pending nominal correction phase.

Pure callback context manager: no ROS imports, parameters or devices on import.
"""
from contextlib import contextmanager

PROFILE='nso_nominal_correction_reverse_v1'
REQUESTED=dict(min_vel_x=-.05,forward_point_distance=0.)


@contextmanager
def correction_reverse_scope(configuration, phase, record, endpoint):
    """Enable authorized adapter phase, scope DWA config, then always restore.

    phase(bool) must raise unless the session-bound adapter acknowledges. No
    reverse goal can be sent before both phase and controller config succeed.
    Failures in either operation attempt disable AND parameter restoration.
    """
    receipt=dict(profile=PROFILE,action_id=endpoint['action_id'],action=endpoint['action'],
                 pending_pose=list(endpoint['pose']),target_odom_pose=list(endpoint['target_odom_pose']),
                 original_method_equivalence=False,requested=dict(REQUESTED),
                 original=None,applied=None,disabled=False,restored=False,errors=[])
    original=None
    try:
        current=configuration.get_configuration(timeout=3.)
        original={name:current[name] for name in REQUESTED}
        receipt['original']=dict(original)
        # At this point the controller is still at its ordinary min_vel_x.
        phase(True)
        applied=configuration.update_configuration(dict(REQUESTED))
        receipt['applied']={name:applied.get(name) for name in REQUESTED}
        if any(applied.get(k)!=v for k,v in REQUESTED.items()):
            raise RuntimeError('correction_reverse_controller_apply_mismatch')
        record('NSO_correction_reverse_phase_enabled',receipt=dict(receipt))
        yield receipt
    except BaseException as exc:
        receipt['errors'].append(type(exc).__name__+': '+str(exc))
        raise
    finally:
        # Clamp the cached negative proposal BEFORE restoring controller params.
        # The adapter also disables automatically on any fault/stop.
        try:
            phase(False)
            receipt['disabled']=True
        except Exception as exc:
            receipt['errors'].append('disable: '+type(exc).__name__+': '+str(exc))
        if original is not None:
            try:
                restored=configuration.update_configuration(dict(original))
                receipt['restore_response']={name:restored.get(name) for name in original}
                receipt['restored']=all(restored.get(k)==v for k,v in original.items())
                if not receipt['restored']:
                    receipt['errors'].append('controller_restore_mismatch')
            except Exception as exc:
                receipt['errors'].append('restore: '+type(exc).__name__+': '+str(exc))
        record('NSO_correction_reverse_phase_finished',receipt=receipt)
        if not receipt['disabled'] or (original is not None and not receipt['restored']):
            # Propagate to mission finally; never continue after incomplete cleanup.
            raise RuntimeError('correction_reverse_phase_cleanup_failed: '+str(receipt['errors']))
