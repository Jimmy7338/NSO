"""Publication conversion tests. No planner, simulator, fusion or evaluator."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('journal_builder', ROOT / 'scripts/build_final_journal_20260929.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def source_text(body='## 1 Introduction\n\nA report [2], then [1,2].\n'):
    return ('# Test publication\n\n## Abstract\n\n' + 'Measured reconstruction. ' * 80
            + '\n\n**Keywords:** planning; mapping; geometry; observation\n\n'
            + body + '\n## References\n\n1. First, A. First study. (2020).\n'
            '2. Second, B. Second study. (2021).\n')


class PublicationConversionTest(unittest.TestCase):
    def test_citations_protect_math_code_and_links(self):
        s = r'Prior [2]. \(p\in[0,1]\), `a[3]`, [1](https://example.org), final [1–2].'
        result, order = builder.convert_citations(s, {1: 'one', 2: 'two'})
        self.assertEqual(order, [2, 1])
        self.assertIn(r'\(p\in[0,1]\)', result)
        self.assertIn('`a[3]`', result)
        self.assertIn('[1](https://example.org)', result)
        self.assertIn(r'\cite{ref1,ref2}', result)

    def test_rejects_unknown_or_unused_references(self):
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            builder.convert_citations('Text [3]', {1: 'one'})
        with self.assertRaisesRegex(ValueError, 'Uncited'):
            builder.convert_citations('Text [1]', {1: 'one', 2: 'two'})

    def test_journal_abstract_and_keyword_limits(self):
        self.assertEqual(len(builder.split_source(source_text())[2]), 4)
        with self.assertRaisesRegex(ValueError, 'abstract'):
            builder.split_source(source_text().replace('Measured reconstruction. ' * 80, 'Short.'))
        with self.assertRaisesRegex(ValueError, 'keywords'):
            builder.split_source(source_text().replace('planning; mapping; geometry; observation', 'mapping'))

    def test_flat_package_preserves_numeric_table_and_blank_authors(self):
        body = ('## 1 Introduction\n\nA report [2], then [1].\n\n'
                '### 1.1 Detail\n\n#### 1.1.1 Boundary\n\n'
                '**Table 1. Measured values.**\n\n'
                '| Method | Score |\n|---|---|\n| G | −0.6995% |\n| S | 6.1638%† |\n\n'
                '† Independently derived value; original failure retained.\n')
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as tmp:
            base = Path(tmp)
            source = base / 'source.md'; source.write_text(source_text(body))
            out = base / 'out'; out.mkdir()
            receipt = builder.prepare(source, out)
            tex = (out / 'main.tex').read_text()
            self.assertEqual(receipt['tables'][0]['rows'][1][1], '−0.6995%')
            self.assertIn(r'\subsection{Detail}', tex)
            self.assertIn(r'\subsubsection{Boundary}', tex)
            self.assertIn(r'\ensuremath{-}0.6995\%', tex)
            self.assertIn(r'6.1638\%', tex)
            self.assertIn(r'\begin{tablenotes}[flushleft]', tex)
            self.assertIn('original failure retained.', tex)
            self.assertIn(r'6.1638\%\textsuperscript{a}', tex)
            self.assertNotIn('/root/', tex)
            self.assertNotIn(r'\input{', tex)
            self.assertFalse(receipt['actual_pdflatex_validation'])
            self.assertEqual((out / 'sn-jnl.cls').read_bytes(), (builder.UPSTREAM / 'sn-jnl.cls').read_bytes())
            self.assertLess(tex.index(r'\end{thebibliography}'), tex.index(r'\section*{Statements and Declarations}'))
            source.write_text(source.read_text() + '\nchanged\n')
            with self.assertRaisesRegex(RuntimeError, 'Input changed'):
                builder.verify_sources(receipt['source_sha256'])

    def test_rejects_more_than_six_table_columns(self):
        row = '| a | b | c | d | e | f | g |\n'
        body = '## 1 Results\n\nText [1,2].\n\n**Table 1. Values.**\n\n' + row + '|---|---|---|---|---|---|---|\n' + row
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as tmp:
            base = Path(tmp); source = base / 'source.md'; source.write_text(source_text(body))
            out = base / 'out'; out.mkdir()
            with self.assertRaisesRegex(ValueError, '2–6'):
                builder.prepare(source, out)

    def test_rejects_outside_workspace_attachments(self):
        with self.assertRaisesRegex(ValueError, 'Unavailable'):
            builder.checked_local('/etc/passwd', ROOT / 'source.md')


if __name__ == '__main__':
    unittest.main()
