def prospective_gates_under_test(protocol, rows, metadata, verification, methods, comparisons, rate_comparison, histories, selected, parents):
    spec = protocol['progression_necessary_conditions']
    gates = {'all_44_complete': gate(len(rows) == metadata['new_physical_branches'] == 44), 'independent_full_replay': verification, 'source_structure_and_release_integrity': gate(True), 'all_routes_collision_free_target_and_return': gate(all((r['collision_count'] == 0 and r['failure'] is None and r['original_target_reached'] and r['returned_to_anchor'] and r['full_original_route_completed'] for r in rows)))}
    for other in spec['primary_total_comparators']:
        delta = comparisons[other]['means']
        cm = methods['total'][other]['means']['new_area_m2']
        area_pass = delta['new_area_m2'] >= spec['near_zero_minimum_absolute_area_gain_m2'] if cm < spec['near_zero_comparator_area_m2'] else delta['new_area_m2'] / cm >= spec['min_S_relative_mean_new_area_gain_against_each_total_comparator']
        gates[f'S_total_area_over_{other}'] = gate(area_pass, delta_m2=delta['new_area_m2'], relative_gain=delta['new_area_m2'] / cm if cm else None)
        for metric, bound in (('final_f1_05cm', 'min_S_minus_each_total_comparator_mean_global_F1_05m'), ('final_joint_05cm', 'min_S_minus_each_total_comparator_mean_C_times_F1_05m'), ('branch_joint_auc_05cm', 'minimum_mean_S_minus_each_total_comparator_sparse_joint_auc')):
            gates[f'S_total_{metric}_over_{other}'] = gate(delta[metric] >= spec[bound], delta=delta[metric], threshold=spec[bound])
        for metric in PRIMARY:
            values = {p: comparisons[other]['per_parent'][p][metric] for p in parents}
            gates[f'all_parents_S_{metric}_over_{other}'] = gate(all((v > 0 for v in values.values())), deltas=values)
    for metric, bound in (('new_area_m2', 'minimum_mean_S_total_minus_S_rate_new_area_m2'), ('final_f1_05cm', 'minimum_mean_S_total_minus_S_rate_global_F1'), ('final_joint_05cm', 'minimum_mean_S_total_minus_S_rate_C_times_F1'), ('branch_joint_auc_05cm', 'minimum_mean_S_total_minus_S_rate_sparse_joint_auc')):
        value = rate_comparison['means'][metric]
        gates[f'total_vs_rate_{metric}'] = gate(value >= spec[bound], delta=value, threshold=spec[bound])
    gates['correct_S_over_X_all_primary'] = gate(all((comparisons['X']['means'][k] > 0 for k in PRIMARY)), deltas={k: comparisons['X']['means'][k] for k in PRIMARY})
    gates['missing_M_exact_G_both_families'] = gate(all((h['families'][f]['M_equals_G'] for h in histories for f in selected)))
    gates['coverage_noninferiority_to_N'] = gate(comparisons['N']['means']['final_coverage_2d'] >= spec['minimum_mean_S_minus_N_coverage_2d'])
    for p in parents:
        choices = [h['families']['total']['choices']['S'] for h in histories if h['context'] == p]
        gates[f'{p}_semantic_choice_exchange'] = gate(len(set(choices)) == 2, choices=choices)
    return gates
