#!/usr/bin/env python3
"""Publication spacing for saved checkpoints; frozen reconstruction is unchanged.

Uses the original renderer, then adjusts text placement before vector/raster
export. This entry point never calls the reconstruction runner or evaluator.
"""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('article_checkpoint_renderer',
    ROOT/'scripts/replay_article_reconstruction_20260928.py')
renderer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
original_save=renderer.save_figure


def publication_save(figure,name):
    if name=='checkpoint_quality_components':
        for text in figure.texts:
            if text.get_text().startswith('P01/h1  cumulative distance'):
                x,y=text.get_position()
                text.set_position((x,y-.012))
    if name=='checkpoint_actual_meshes':
        for axis in figure.axes:
            for text in axis.texts:
                if text.get_text().startswith('$F_1$'):
                    text.set_bbox(dict(facecolor='white',edgecolor='none',alpha=.92,pad=.8))
    original_save(figure,name)


if __name__=='__main__':
    renderer.save_figure=publication_save
    renderer.plot()
    path=renderer.FIGURES/'manifest.json'
    manifest=json.loads(path.read_text())
    manifest.update(layout_source_sha256=renderer.sha(__file__),
        layout_source='scripts/plot_article_reconstruction_20260928.py',
        command='.venv/bin/python -B scripts/plot_article_reconstruction_20260928.py',
        layout_only_changes=['lower final distance annotation by 0.012 figure height',
            'white text backing behind mesh numerical labels'],
        frozen_reconstruction_script_modified=False)
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
