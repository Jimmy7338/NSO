# First target divergence

After 19 completed actions (planning call 2), covered area is 34.92 m² (86.6071% of reachable space). Initial and all 19 action/pose/observation packet hashes match. The recorded planning inputs, complete candidate lists, candidate audit (including belief and safe-grid hashes), and score audits match. The first selected target differs here; the first executed action/pose differs at action 31.

G selects candidate 3 at [3, 9, 2]; S selects candidate 7 at [5, 11, 2]. Strict scores: G gives 16.791596 vs 16.747282 (margin 0.044313); S gives 21.964972 vs 21.674504 (margin 0.290468). The common proxy and the G/S prior components are recorded in mechanism_audit.json. Ranking reversal aligns exactly with the mode-dependent prior term; neither is a score tie. This is evidence from recorded synthetic-marker-conditioned scoring at a shared state, not a separate intervention or physical semantic validation.
