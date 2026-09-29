from pathlib import Path

root = Path('/root/NSO')
source = root / 'docs/thesis/ARTICLE_SUBMISSION_EN_20260928.md'
target = root / 'docs/thesis/ARTICLE_SUBMISSION_EN_REVISED_20260929.md'
text = source.read_text()

def replace(old, new):
    global text
    assert text.count(old) == 1, (text.count(old), old[:100])
    text = text.replace(old, new)

abstract = '''Robots operating in reconfigured industrial workspaces must recover both spatial coverage and equipment surface geometry under limited observation budgets. This paper presents a category-conditioned active observation method for selecting useful viewing directions before geometry resolves a facility's configuration. The method combines category priors, bounded geometric feedback, diagnostic lookahead, and return-constrained planning in a hierarchical decision–execution loop. A finite-state planner evaluates observation sequences using public configuration templates, while measured depth alone contributes to volumetric reconstruction. We analyze when category information can reduce the cost of geometric diagnosis and demonstrate how contrary measurements correct an initially misleading prior. CPU simulations compare matched geometric and category-conditioned planners using identical sensing, reconstruction, and action budgets. On two reference layouts, category conditioning improves mean coverage–surface quality by 6.16% at unchanged planar coverage. With the policy fixed, two additional background layouts yield a 2.98% mean improvement, concentrated in a condition where the category cue changes the selected observation direction. An incorrect-prior comparison gives a 2.62% gain for the complete correction policy. Budget and geometry variations show positive mean gains at the intermediate budget and a small reversal at the larger budget. Recorded trajectories and reconstructed surfaces support budget-sensitive benefits of early category information for facility observation.'''
start = text.index('## Abstract\n\n') + len('## Abstract\n\n')
end = text.index('\n\n**Keywords:**', start)
text = text[:start] + abstract + text[end:]
replace('Submission discussion draft; not submitted. The companion evidence manuscript retains complete version-specific experiments and implementation details.',
        'Revised main manuscript, 29 September 2026; not submitted. This revision reorganizes the method and presentation of existing results. The earlier manuscript and experimental records remain available unchanged.')

replace('The contribution is a task-specific observation method and its mechanism evidence. Bayesian updating and semantic inspection provide established foundations. A separate multi-facility implementation examines the limits of sharing class-model evidence.',
        'The study focuses on category-informed observation allocation for facility documentation. Its central question is when an early category cue changes a useful viewing decision, and how that decision survives correction by acquired geometry. Bayesian estimation and informative planning provide the foundations; the contribution lies in their task-specific integration and the connection between recorded decisions and measured reconstruction.')

replace('## 2 Related Work\n\nActive Neural SLAM provides an early learned hierarchical exploration architecture [1]; this project retains the general decision/execution separation while replacing the mechanism and task. TARE coordinates local and global exploration through representations at different resolutions [5]. The active-SLAM survey by Placed et al. connects motion selection, map quality, and estimation [10]. Here, exact poses and a common safe graph isolate observation allocation rather than localization accuracy.',
        '## 2 Related Work\n\n### 2.1 Active mapping and hierarchical exploration\n\nActive mapping couples motion selection with map quality and state estimation [10]. TARE coordinates local and global exploration through representations at different resolutions [5]. Learned hierarchical exploration, including Active Neural SLAM [1], also separates strategic decisions from execution. The present method uses this general separation for budgeted facility observation. Exact poses and a common safe graph allow the experiments to isolate the observation-planning contribution.\n\n### 2.2 Reconstruction-driven view planning')
replace('Semantic inspection already connects target meaning with observation requirements.',
        '### 2.3 Semantic information for observation planning\n\nSemantic inspection connects target meaning with observation requirements.')
replace('## 3 Task, Information, and Quality', '## 3 Problem Formulation')

# Move numerical evaluation details to experimental setup; keep the task objective in Section 3.
begin = text.index('The evaluator derives the start-reachable safe-cell set')
end = text.index('\n\nThe desired policy', begin)
evaluation = text[begin:end]
text = text[:begin] + text[end+2:]
replace('### 5.1 Protocol and comparisons', '### 5.1 Experimental setup and matched comparisons')
anchor = 'The experiments isolate planning under controlled observations rather than reproduce physical sensor errors.'
replace(anchor, anchor + '\n\n' + evaluation)
replace('## 5 Experiments and Results\n\n',
        '## 5 Experiments and Results\n\nThe experiments address four questions: whether category information improves surface recovery at matched coverage; whether acquired geometry corrects a misleading cue; how observation allocation compares with other planning mechanisms; and how the benefit changes with budget, geometry, and background layout.\n\n')
replace('### 5.2 Category information changes observed surfaces', '### 5.2 Surface reconstruction at matched coverage')
replace('Table 2 retains all four confirmation pairs. Mean \\(J_5\\) increases from 0.702576 to 0.745881, a relative improvement of **6.1638%**, with two wins and two ties.',
        'Across the four confirmation conditions in Table 2, mean \\(J_5\\) increases from 0.702576 to 0.745881, a relative improvement of **6.1638%**.')
replace('The gain is therefore a surface-quality improvement at matched coverage, not a localization improvement.',
        'This decomposition attributes the observed joint-score gain to surface reconstruction at matched planar coverage.')
replace('**Figure 3.** Complete paired outcomes, including both ties, and the coverage/F1 decomposition.',
        '**Figure 3.** Paired outcomes for all four conditions and the coverage/F1 decomposition.')
replace('### 5.3 Measured correction and reconstruction progress', '### 5.3 Geometric correction and reconstruction progress')
replace('### 5.4 CPU mechanism comparisons', '### 5.4 Comparison with common-CPU planning mechanisms')
replace('### 5.5 Fixed budget, noise, and geometry extension', '### 5.5 Sensitivity to budget, noise, and facility geometry')
replace('At budget 42, both original and variant comparisons yield four S/G wins and four ties. At 54, there are four ties and four losses.',
        'At budget 42, improvements occur in the h1 configurations of both the original and variant layouts, while the h0 routes remain unchanged. At budget 54, the mean difference reverses to −0.6995%, with lower scores in four conditions and unchanged scores in the other four.')

# The different multi-facility controller and metric are discussed as a separate scope test.
# Its complete table remains in the unchanged companion evidence manuscript.
begin = text.index('### 5.6 Separate multi-facility extension')
end = text.index('### 5.7 Frozen-policy validation in new background layouts', begin)
text = text[:begin] + text[end:]
replace('### 5.7 Frozen-policy validation in new background layouts', '### 5.6 Transfer to changed background layouts')
replace('Table 6 reports every paired endpoint.', 'Table 5 reports every paired endpoint.')
replace('Mean joint quality increases from 0.705188 to 0.726215, or **2.9818%**, with one win and three ties.',
        'Mean joint quality increases from 0.705188 to 0.726215, or **2.9818%**. The gain is concentrated in the rear-partition h1 condition; the other three conditions retain the same trajectories and endpoint quality.')
replace('**Table 6. All frozen-policy layout pairs at 5 cm. Coverage is equal within each pair; qualification is 8/8.**',
        '**Table 5. Changed-background results at 5 cm for all four paired conditions. Coverage is equal within each pair; all eight tasks qualify.**')
replace("Figure 8 shows all saved paths. In the side-column layout, both initial observation directions satisfy the return constraint. Category information changes candidate values but does not change their ordering: both methods choose left and remain identical in actual actions. In the rear-partition h1 condition, G's initial left/right values tie within the frozen numerical tolerance; its original ordering selects left, whereas S selects right. Their nonsemantic histories and geometric log odds are identical at step 18; category weights alone differ. The first executed divergence is action 19, followed by greater recovered surface recall at unchanged coverage. This supports the complete category-conditioned route's effect, not a separate quality gain attributable to a single frame.",
        "The trajectories in Figure 8 explain the effect's dependence on the available observation routes. In the side-column layout, both initial directions are return-feasible. Category information changes their predicted values but preserves their ordering, so both planners select the same route. In the rear-partition h1 condition, G assigns nearly equal values to the two directions and selects left through its fixed tie rule, whereas S selects right. Up to step 18 their acquired nonsemantic histories and geometric support are identical. The category-conditioned decision changes action 19 and improves endpoint surface recall from 0.595795 to 0.724945. In this condition, \\(J_5\\) rises from 0.642160 to 0.726269, a relative improvement of **13.10%**, at unchanged coverage and action count. This is a complete-route effect: the subsequent measurements differ, so the endpoint gain is not attributed to a single frame.")
replace('Independent review checks all 344 saved frames, priors, measured residuals, selected actions, return constraints, source hashes and endpoint arithmetic. No parameters are retuned after these results, and no reserved additional tasks are used.',
        'The controller and all evaluation settings remain fixed throughout this background-layout comparison.')

begin = text.index('## 6 Discussion and Conclusion')
end = text.index('## References', begin)
discussion = '''## 6 Discussion

### 6.1 When category information has decision value

The results connect three events: an early category cue changes a feasible observation direction, the resulting route acquires different surfaces, and those measurements improve reconstructed surface recall at matched coverage. This chain is visible in both reference h1 configurations and in the rear-partition h1 condition. When category information changes scores without changing their ordering, as in the side-column layout, the resulting trajectory and quality remain unchanged. These cases explain why semantic availability alone is insufficient: the cue must alter an executable decision with a useful downstream consequence.

Budget determines the opportunity for that consequence. At 42 actions, the original and same-family geometry groups show positive mean gains. At 30 actions, the planner cannot meet its predicted coverage and return constraints. At 54 actions, both policies reach complete planar coverage, while S has slightly lower surface precision and joint quality. The decision analysis assumes meaningful directional utilities; the implemented template-exposure surrogate does not model all depth-fusion errors. Its mismatch with measured surface quality is therefore a concrete target for improvement, without treating the observed reversal as evidence for a particular unmeasured sensor-error cause.

### 6.2 Scope of the evidence and extensions

The local method uses public configuration templates, a supplied safe graph, exact poses, and synthetic category cues. These assumptions isolate the planning value of category information but leave natural recognition and estimated-pose SLAM for subsequent validation. The background test adds two layouts within the P00 facility family. Configuration pairs, noise repeats, and reconstruction checkpoints do not constitute additional independent layout samples. The results are descriptive evidence of the tested mechanism, rather than a statistical claim of general superiority. SWAP-I and VISTA-I adapt selected mechanisms to a common CPU setting; the TARE runs demonstrate integration feasibility. These comparisons do not rank the corresponding complete original systems.

A separate multi-facility study explores a different question: whether evidence from one instance improves observation of another instance of the same category. Across 36 runs covering three development layouts and three controller versions, shared and fixed category priors produce identical actions and quality in all nine matched comparisons. Thirty-five runs complete the original evaluation pipeline; the remaining run completes navigation but requires a separately reported numerical evaluation correction. Because this study uses observation macros and a different joint metric, its scores are not pooled with the local results. The complete version-specific tables and the earlier six-parent sharing study, which also did not establish a sharing gain, remain in the companion evidence manuscript. Additional cross-instance semantic benefit is therefore an open extension, separate from the local directional mechanism demonstrated here.

The available ROS 1 robot provides a future implementation platform; the reported performance evidence is entirely simulated. Industrial relevance currently concerns observation after workspace reconfiguration with a static environment during acquisition. Continuous map maintenance and dynamic obstacle avoidance are outside this evaluation.

## 7 Conclusion

This paper presented a category-conditioned observation method combining geometric correction, diagnostic lookahead, and budget-aware execution for facility reconstruction. Category information improves mean joint coverage–surface quality by **6.16%** on the reference layouts and **2.98%** on two additional background layouts, with unchanged mean coverage in each comparison. Recorded trajectories show that the benefit arises when early category support changes the viewing direction; acquired geometry can also redirect a route after a misleading cue. Budget sensitivity identifies the regime in which this allocation is useful and the limitations of the current exposure surrogate. These findings support category-informed, budget-constrained facility observation and motivate closer alignment between predicted observation value and measured reconstruction quality.

## Reproducibility and Data

The [complete evidence manuscript](/root/NSO/docs/thesis/VIRTUAL_PAPER_EN_20260928.md) retains all local and scene-level results, including unchanged outcomes, planning failures, and adverse differences. The [supplementary methods](/root/NSO/docs/thesis/ARTICLE_SUPPLEMENTARY_METHODS_20260928.pdf) describe the separate multi-instance implementation. Archived code, inputs, sensor observations, trajectories, actual meshes, and plotting sources support reproduction. The 128-trial extension, 36 scene-level attempts, and eight background-layout tasks are retained as distinct experiment groups. A representative 43-frame saved replay reproduces controller outputs and reconstructed meshes exactly; intermediate reconstruction checkpoints reuse existing observations. No additional experiment or metric recomputation is introduced by this manuscript revision.

'''
text = text[:begin] + discussion + text[end:]
target.write_text(text)
print(target)
print('Abstract words:', len(abstract.split()))
