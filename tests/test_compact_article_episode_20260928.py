"""Synthetic filesystem compaction tests; no study artifact is deleted."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import archive_article_episode_20260928 as archive
from scripts import compact_article_episode_20260928 as compact
from tests.test_archive_article_episode_20260928 import fixture, write


def prepared(root):
    episode,review=fixture(root)
    raw=episode/'packets/000_rgbd.npz';raw.parent.mkdir();raw.write_bytes(b'opaque temporary byte fixture')
    manifest=json.loads((episode/'artifact_manifest.json').read_text())
    manifest['files']['packets/000_rgbd.npz']=dict(bytes=raw.stat().st_size,sha256=archive.sha_file(raw))
    write(episode/'artifact_manifest.json',manifest);pin=archive.sha_file(episode/'artifact_manifest.json')
    ledger=json.loads((root/'phase/start_ledger.json').read_text());ledger['entries'][0]['artifact_manifest_sha256']=pin
    write(root/'phase/start_ledger.json',ledger)
    report=json.loads((review/'review.json').read_text());report['input_manifest_sha256']=pin;write(review/'review.json',report)
    review_manifest=json.loads((review/'manifest.json').read_text())
    review_manifest['files']['review.json']=dict(bytes=(review/'review.json').stat().st_size,sha256=archive.sha_file(review/'review.json'))
    write(review/'manifest.json',review_manifest)
    packed=root/'fixture.tar.xz';archive.pack(episode,review,packed)
    planned=compact.plan_compaction(episode,review,packed,root/'plan')
    return episode,review,packed,planned


def rewrite_plan_for_negative_test(root,change):
    path=root/'plan/plan.json';value=json.loads(path.read_text());change(value);write(path,value)
    manifest=json.loads((root/'plan/manifest.json').read_text())
    manifest['plan_sha256']=archive.sha_file(path)
    manifest['files']['plan.json']=dict(bytes=path.stat().st_size,sha256=archive.sha_file(path))
    write(root/'plan/manifest.json',manifest)
    return manifest['plan_sha256']


class ArticleCompactionTests(unittest.TestCase):
    def test_plan_deletes_nothing_and_explicit_apply_keeps_every_metadata_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);episode,review,packed,planned=prepared(root)
            original={str(p.relative_to(episode)):p.read_bytes() for p in episode.rglob('*') if p.is_file()}
            self.assertEqual(planned['raw_files_deleted'],0)
            self.assertEqual(len(original),6)
            with self.assertRaisesRegex(ValueError,'explicit approval'):
                compact.apply_compaction(root/'plan',root/'operation','not the approved SHA')
            self.assertEqual({str(p.relative_to(episode)):p.read_bytes() for p in episode.rglob('*') if p.is_file()},original)
            result=compact.apply_compaction(root/'plan',root/'operation',planned['plan_sha256'])
            self.assertEqual(result['raw_files_removed'],2)
            self.assertEqual(result['original_metadata_files_edited'],0)
            for name,data in original.items():
                if compact.removable(name):self.assertFalse((episode/name).exists())
                else:self.assertEqual((episode/name).read_bytes(),data)
            restored=compact.materialize(root/'plan',root/'workspace')
            self.assertTrue(restored['strict_original_manifest_passed'])
            for name,data in original.items():self.assertEqual((Path(restored['episode'])/name).read_bytes(),data)
            self.assertEqual((Path(restored['episode']).parent.parent/'start_ledger.json').read_bytes(),
                             (root/'plan/phase_ledger_snapshot.json').read_bytes())

    def test_archive_corruption_refuses_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);episode,review,packed,planned=prepared(root)
            packed.write_bytes(packed.read_bytes()[:-8])
            with self.assertRaisesRegex(ValueError,'archive changed'):
                compact.apply_compaction(root/'plan',root/'operation',planned['plan_sha256'])
            self.assertTrue((episode/'packets/000_rgbd.npz').is_file())
            self.assertTrue((episode/'steps/000.json.gz').is_file())
            self.assertFalse((root/'operation').exists())

    def test_metadata_cannot_be_added_to_removal_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);episode,review,packed,planned=prepared(root)
            def corrupt(plan):
                row=next(x for x in plan['retain'] if x['path']=='result.json')
                plan['retain'].remove(row);plan['remove'].append(row)
            pin=rewrite_plan_for_negative_test(root,corrupt)
            with self.assertRaisesRegex(ValueError,'allowlist'):
                compact.apply_compaction(root/'plan',root/'operation',pin)
            self.assertTrue((episode/'result.json').exists())
            self.assertTrue((episode/'steps/000.json.gz').exists())

    def test_materialization_phase_cannot_escape_requested_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);episode,review,packed,planned=prepared(root)
            rewrite_plan_for_negative_test(root,lambda p:p.update(phase_name='../escape'))
            with self.assertRaisesRegex(ValueError,'bounded component'):
                compact.materialize(root/'plan',root/'workspace')
            self.assertFalse((root/'workspace').exists());self.assertFalse((root/'escape').exists())

    def test_interruption_records_partial_removal_and_archive_still_restores_all(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);episode,review,packed,planned=prepared(root)
            original_unlink=Path.unlink
            def fail_second(path,*args,**kwargs):
                if path==episode/'steps/000.json.gz':raise OSError('injected interruption before second deletion')
                return original_unlink(path,*args,**kwargs)
            with patch.object(Path,'unlink',fail_second):
                with self.assertRaises(OSError):
                    compact.apply_compaction(root/'plan',root/'operation',planned['plan_sha256'])
            partial=json.loads((root/'operation/incomplete.json').read_text())
            self.assertFalse(partial['automatic_resume'])
            self.assertEqual(partial['removed_paths'],['packets/000_rgbd.npz'])
            self.assertTrue((episode/'artifact_manifest.json').exists())
            restored=compact.materialize(root/'plan',root/'workspace')
            self.assertEqual(restored['verified_original_files'],6)
            self.assertTrue((Path(restored['episode'])/'packets/000_rgbd.npz').exists())

    def test_failed_episode_remains_ineligible_without_relabeling(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);episode,review,packed,planned=prepared(root)
            ledger_path=root/'phase/start_ledger.json';ledger=json.loads(ledger_path.read_text())
            ledger['entries'][0].update(status='attempt_failed',qualified=False);write(ledger_path,ledger)
            with self.assertRaisesRegex(ValueError,'completed qualified'):
                compact.apply_compaction(root/'plan',root/'operation',planned['plan_sha256'])
            self.assertTrue((episode/'steps/000.json.gz').exists())


if __name__=='__main__':unittest.main()
