# Three semantic development views

These are development setup moves and single-observation offline scores, not policy rollouts, controlled efficacy comparisons or physical counterfactuals. Each view has its own captured map and sensor observation. The actual robot setup goal and stop records are retained; C/G/S/X scores are shadows on the same state. Observation-stage and post-plan semantic diagnostics are separate.

Raw RGB-D and large input packets are excluded; source receipts record paths and hashes. Small semantic cache NPZ and its receipt are included. Consequently this is not a standalone detector or full sensor replay. Gzip is lossless and hashes are in MANIFEST.json.

Minus20 and central have zero G/S candidate differences. Plus20 has nonzero S−G candidate differences, while G and S retain the same top choice and a lower-ranked pair reverses order. One active asset makes S equal to a uniform 0.4135188632 scaling of the generic prior in this state; this is not evidence of semantic efficacy. See readonly_science_audit.json.
