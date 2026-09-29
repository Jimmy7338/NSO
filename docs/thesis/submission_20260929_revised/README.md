# JIRS formatting package — not submitted

Target: Journal of Intelligent & Robotic Systems. This package uses the unchanged
official Springer Nature December 2024 sn-jnl template and the journal guidelines
checked on 2026-09-29. It is a formatting draft, not a claim of acceptance and not
a commitment to publication fees. JIRS is fully open access; the author must
separately decide on APC funding or a verified waiver before submission.

Compile the single main.tex with pdfLaTeX three times, with shell escape disabled:

    pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex

Keep all supplied files in one directory. Tables and references are embedded in
main.tex; Table*.tex and references.tex/json are editable inspection copies, not
external inputs required by the main document. Fig*.pdf files preserve full-size
vector figures. Inspect these originals for dense panel details; no claim is made
that every small label in the scaled manuscript is at least 8 pt. ESM_*.pdf files
are the numbered Online Resources. The original template notices and CTAN source
archives retain their own licenses, not the research repository's license.

Before any submission the authors must fill author_fields.json and the commented
author/affiliation/email fields in main.tex, confirm funding, competing interests,
contributions, MSC classification, AI-use details, final public data/version links,
and Online Resource author metadata. They must review and approve the scientific
text and disclosures. Blank fields and draft declarations are intentional; this
package must not be represented as an author-approved submission.

build_receipt.json binds the actual manuscript, figures, supplementary PDFs,
template, references, compiler result and output file hashes. No experiment or
metric recomputation is part of this publication build.
