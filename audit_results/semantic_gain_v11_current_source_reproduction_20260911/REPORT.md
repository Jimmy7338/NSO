# V11 current-source reproduction receipt

The archived V11 training manifest records the pre-integration hash of
`nso/semantic_gain_v11.py`. The file was later extended with deserialization,
ensemble and runtime scoring interfaces. A clean training run with the current
source and the same frozen V9 inputs reproduced `summary.json`, `folds.json`
and `frozen_model.json` byte for byte. The runtime additions therefore did not
change the fitted development model or its reported metrics.

This receipt resolves source continuity for a new clone. The original manifest
remains unchanged as historical evidence; the independent numerical replay is
still the stronger implementation-independent check.
