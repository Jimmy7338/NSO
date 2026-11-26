# Offline audit of same-goal route-pool capacity

The source is the eight sealed development episodes at commit `02e9c5dcc0da7df61f23d8683fec2dd92c145c1e`. The 25 audited decisions are the exact common states where the old selector admitted a non-native route. A replay issues only the recorded controls; the expanded rankings remain offline. Before expanding each pool, both old audits are recomputed with complete equality of candidates, scores, and winners.

The audit calls the original selector with an expanded limit that can hold all traces returned by its original generator. The original observed safety graph, coverage floor, action costs, return suffix and tie rule are retained. Generated scores describe hypothetical visibility at the recorded observation. Any later change to the observed map and its effect on execution are outside this audit.

- [Protocol](PROTOCOL.json) and [source freeze](SOURCE_FREEZE.json).
- [Per-decision CSV](verification/results.csv), [first divergence and verified totals](verification/results.json), [completion audit](completion_audit.json) and [evidence index](EVIDENCE_INDEX.json).
- [Manuscript-ready results](RESULTS_ZH.md).

The local study contains all per-route candidates and actions in `replay/runs/<layout>/states/`. The exact original runs, native mapper, initial sensors and fitted models remain at their frozen paths. Reproduce on the original host with Python 3.12 and the NSO science environment. Use a fresh copy of the run directories to preserve the published attempt.

```sh
python replay/supervise.py --freeze
python replay/supervise.py
python verification/verify_expansion.py --out /tmp/nso-pool-audit-verification
```

The verification scripts work on the sealed replay records without invoking a planner or simulator. `python audit.py` cross-checks the published local outputs and archived source hashes. The GitHub `evidence/<layout>/` files are gzip-compressed copies of the original per-state audit JSON and its sealed layout manifest.
