# Cover letter draft — author approval required; not sent

Date: ____________________

To the Editors, *Journal of Intelligent & Robotic Systems*

**Proposed manuscript:** “Category Priors and Geometric Feedback for Budget-Constrained Active Observation of Facilities”

**Proposed article type:** Original research article (confirm the submission-system label).

Dear Editors,

Please consider the above manuscript for publication in the *Journal of Intelligent & Robotic Systems*. It investigates how category information can improve the choice of observation directions when a robot must document a static industrial facility under a finite action budget. The method connects category-conditioned configuration beliefs, diagnostic lookahead, geometric feedback, and return-constrained execution. Only acquired depth contributes to the reconstructed surfaces.

The contribution is a category-conditioned observation method with an explicit account of when semantic information changes the next observation decision. A matched geometric control retains diagnostic observations, revisiting, navigation, and reconstruction, so that the comparison isolates the additional category information. A two-configuration analysis relates directional utility to prior reliability and diagnostic cost. Recorded trajectories, actual truncated signed distance function meshes, and 40 reconstruction checkpoints connect planning decisions to surface recovery. Bayesian updating itself is not presented as a new algorithm.

Across four matched conditions in the original confirmation, the method improves the mean product of planar coverage and surface F1 by 6.16% while preserving mean coverage. A subsequent validation keeps the policy fixed and evaluates two additional background layouts, each with two facility configurations and both methods. Across all four matched conditions (eight executed tasks), the mean joint score improves by 2.98%. The benefit occurs when the category information changes the selected observation direction; where both policies select the same route, their measured outcomes coincide. This distinction connects the aggregate improvements to an identifiable planning mechanism rather than treating repeated conditions as independent layouts.

The revised manuscript develops the configuration belief, measured feedback, observation-value recursion, budget and return constraints, and execution algorithm in a self-contained method section. Budget and geometry variations test the regime in which early category information is useful. The fixed 128-trial extension includes 96 qualified completions and 32 low-budget feasibility failures; its larger-budget comparison gives a small negative difference. A separate 36-run multi-facility study does not establish an additional shared-prior benefit. The main discussion states these applicability limits, and the complete evidence manuscript retains all recorded groups and version-specific outcomes.

The study fits the journal's interests in robotic planning, simulation, and intelligent applications in industrial settings. The experiments use a processor-based, controlled simulation with known safe navigation and exact poses. The claims concern active observation and measured surface reconstruction, rather than localization accuracy or validated physical deployment. The accompanying materials provide complete evidence, editable tables, full-size vector figures, and reproducibility information.

Thank you for considering this manuscript.

Corresponding author: ____________________

Affiliation: ____________________

Email: ____________________

Signature and date, after author approval: ____________________

## Confirm before adopting or sending this draft

The statements below are **unconfirmed author declarations**, not assertions made by this draft. Complete them and incorporate the applicable wording into the final letter or submission form.

- [ ] Confirm originality, permitted reuse, and the relationship to the graduation thesis, preprints, repository releases, or other prior dissemination. List all relevant versions and links: ____________________.
- [ ] Confirm that the manuscript is not under consideration by another journal: ____________________.
- [ ] Confirm that every named author approves the manuscript, authorship order, contribution statement, and submission: ____________________.
- [ ] Confirm the accuracy of funding, competing interests, data/code availability, and AI-assistance disclosures already supplied as drafts in the manuscript: ____________________.
- [ ] Confirm permission to publish the supplied robot photograph and any other material requiring permission: ____________________.

This file is an internal submission draft. It does not sign on behalf of any author, contact the journal, or authorize submission or payment.
