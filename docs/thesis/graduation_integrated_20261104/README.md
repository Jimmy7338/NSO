# Integrated graduation thesis reading draft

This is an integrated working draft, not a school-formatted or submitted thesis. The manuscript distinguishes controlled simulator comparisons, robot integration observations, and offline counterfactuals; it does not establish physical reconstruction accuracy or a general semantic advantage. The related Q0 arrangements are not independent scenes.

`thesis.md` is the current integrated source, `thesis.pdf` is the reading copy, and `integration_and_evidence.md` records its evidence boundaries. The referenced PNG figures and available matching vector PDFs are included. `validation.json`, `math_manifest.json`, and `math_render_errors.json` record the local build checks; generated HTML, intermediate Markdown and formula PNG caches are omitted.

Build from this directory with `bash build_pdf.sh`. The build requires Python 3 with Pillow, `pdflatex` with amsmath/amssymb/graphicx, Poppler `pdftocairo`, Pandoc, LibreOffice, and the Noto Sans CJK regular font at `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`. Pandoc can be selected with `NSO_THESIS_PANDOC` and its data directory with `NSO_THESIS_PANDOC_DATA_DIR`; `NSO_THESIS_PYTHON` selects Python. Layout and PDF bytes may vary by installed versions and fonts. The script renders formulas into a local cache and regenerates the reading PDF from the included `thesis.md`; it does not require the historical manuscripts.

`integrate.py` is retained for source provenance and rebuilding the integrated Markdown; it requires historical sibling manuscript sources in the repository layout and is not needed to render the included `thesis.md`. Those historical sources and experimental evidence must be checked before treating the integrated draft as a final manuscript.
