#!/usr/bin/env python3
"""Check the predeclared same-observation/different-hidden-depth fixture."""
import argparse,json
from pathlib import Path
import numpy as np
from evaluate_calibration import sha,read,resource,load_resource,prediction_bank,write
from distribution_metrics import SIGMA_GRID_M,joint_posterior

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--predictions',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    index=read(args.predictions);assert index['status'] in ('completed_isolated_banks','completed_with_failures')
    items=index['counterfactual'];assert len(items)==2
    result=dict(schema='B8_independent_counterfactual_posterior_check_v1',inference_index=resource(args.predictions),
        generator_truth_loaded=False,extra_calibration_object_units=0,sigma_grid_m=list(SIGMA_GRID_M),code_sha256=sha(__file__))
    try:
        pred=[load_resource(item['prediction']) for item in items]
        banks=[prediction_bank(p) for p in pred]
        a,da,qa,ea=banks[0];b,db,qb,eb=banks[1]
        for va,vb in [(da,db),(qa,qb),(ea,eb),(a['depth_axis_world'],b['depth_axis_world'])]:np.testing.assert_array_equal(va,vb)
        for key in ('particles','fit_input_points_sha256','fit_input_normals_sha256','fit_input_labels_sha256','fixed_ray_audit'):
            assert a[key]==b[key],key
        comparisons=[]
        for sigma in SIGMA_GRID_M:
            wa=joint_posterior(ea,qa,sigma);wb=joint_posterior(eb,qb,sigma);np.testing.assert_array_equal(wa,wb)
            comparisons.append(dict(sigma_m=sigma,max_joint_probability_difference=float(np.max(np.abs(wa-wb)))))
        result.update(status='passed',same_bank_likelihood_and_all_sigma_posteriors=True,comparisons=comparisons,
            prediction_resources=[item['prediction']for item in items],depth_axis_world=a['depth_axis_world'])
    except Exception as exc:
        result.update(status='failed',same_bank_likelihood_and_all_sigma_posteriors=False,error=repr(exc))
    write(args.output,result);print(json.dumps(dict(status=result['status'],output=resource(args.output))))
    if result['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
