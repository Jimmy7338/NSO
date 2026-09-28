"""Explicit observed-label encoding for the two artificial asset generators.

No simulator, world identity, hidden structure, reward or trained model access.
Encoding metadata is provenance, never an additional model feature.
"""
from copy import deepcopy
import math
from numbers import Real
import numpy as np

CANONICAL_SCHEMA='artificial_asset_binary/open_negative_closed_positive/1'
SOURCE_SIGNS={'competition_v9':1.,'inspection_v4':-1.}
COMMON_DIM=18
TAIL_DIM=8


def source_sign(source_schema):
    if source_schema not in SOURCE_SIGNS:
        raise ValueError('explicit supported source taxonomy required; never infer it from rewards')
    return SOURCE_SIGNS[source_schema]


def canonical_vote(raw_vote,source_schema):
    sign=source_sign(source_schema)
    if isinstance(raw_vote,(bool,np.bool_)) or not isinstance(raw_vote,Real):
        raise ValueError('numeric observed class vote required')
    raw_vote=float(raw_vote)
    if not math.isfinite(raw_vote) or abs(raw_vote)>1:
        raise ValueError('observed vote outside [-1,1]')
    return 0. if raw_vote==0 else sign*raw_vote


def canonical_observed_records(records,source_schema):
    source_sign(source_schema)
    result=[]
    for record in records:
        if 'semantic_encoding' in record or 'raw_class_vote' in record:
            raise ValueError('record already annotated; double encoding prohibited')
        row=deepcopy(record)
        row['class_vote']=canonical_vote(record['class_vote'],source_schema)
        row['raw_class_vote']=float(record['class_vote'])
        row['semantic_encoding']=dict(source_schema=source_schema,canonical_schema=CANONICAL_SCHEMA)
        result.append(row)
    return result


def canonical_feature_record(record,source_schema):
    """Encode the eight signed V14 moments, preserving common/geometry inputs.

    Flipping the source vote flips its confidence-weighted moments. No capacity,
    confidence, candidate, instance ordering or geometric quantity is changed.
    """
    sign=source_sign(source_schema)
    if 'semantic_encoding' in record:
        raise ValueError('feature already annotated; double encoding prohibited')
    geometry=np.asarray(record['geometry'],dtype=float)
    semantic=np.asarray(record['semantic'],dtype=float)
    if (geometry.shape!=(COMMON_DIM+TAIL_DIM,) or semantic.shape!=geometry.shape
            or not np.isfinite(geometry).all() or not np.isfinite(semantic).all()):
        raise ValueError('finite 26-dimensional V14 feature record required')
    if not np.array_equal(geometry[:COMMON_DIM],semantic[:COMMON_DIM]):
        raise ValueError('semantic and geometry common inputs disagree')
    confidence=record['residual_confidence']
    if isinstance(confidence,(bool,np.bool_)) or not isinstance(confidence,Real) or not 0<=confidence<=1:
        raise ValueError('invalid residual confidence')
    if confidence==0 and np.any(semantic[COMMON_DIM:]!=0):
        raise ValueError('nonzero semantic moments with zero confidence')
    row=deepcopy(record)
    tail=sign*semantic[COMMON_DIM:]
    # Missing evidence must not gain a source-dependent JSON hash via -0.0.
    tail[tail==0]=0.
    row['semantic']=np.r_[semantic[:COMMON_DIM],tail].tolist()
    row['semantic_encoding']=dict(source_schema=source_schema,canonical_schema=CANONICAL_SCHEMA,
        feature_schema='route_instance_v14/26',provenance_is_model_input=False)
    return row
