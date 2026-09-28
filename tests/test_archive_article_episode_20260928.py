"""Temporary byte-level packaging fixtures; never experimental sensor runs."""
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts import archive_article_episode_20260928 as archive


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.canonical(value))


def fixture(root,*,gzip_mtime=0):
    episode=root/'phase/episodes/fixture';episode.mkdir(parents=True)
    logical=b'{"fixture":"'+b'analytic storage-only fixture;'*(100000)+b'"}\n'
    content={
        'steps/000.json.gz':gzip.compress(logical,compresslevel=6,mtime=gzip_mtime),
        'result.json':archive.canonical({'status':'controller_stop','fixture_not_scientific_data':True}),
        'encoding.json':archive.canonical({'steps':[dict(artifact='steps/000.json.gz',compression_level=6,
                                                       mtime=0,uncompressed_bytes=len(logical))]}),
        'payload.bin':bytes(range(256))*2,
    }
    for name,data in content.items():
        p=episode/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    files={name:dict(bytes=len(data),sha256=archive.sha_bytes(data)) for name,data in content.items()}
    write(episode/'artifact_manifest.json',dict(files=files))
    pin=archive.sha_file(episode/'artifact_manifest.json')
    write(root/'phase/start_ledger.json',dict(entries=[dict(run_id='fixture',status='controller_stop',
        qualified=True,artifact_manifest_sha256=pin,result_sha256=archive.sha_file(episode/'result.json'))]))
    review=root/'review';write(review/'review.json',dict(status='reviewed',all_checks_passed=True,
        qualified=True,run_id='fixture',input_manifest_sha256=pin))
    write(review/'manifest.json',dict(files={'review.json':dict(bytes=(review/'review.json').stat().st_size,
                                                              sha256=archive.sha_file(review/'review.json'))}))
    return episode,review


class ArticleArchiveTests(unittest.TestCase):
    def test_pack_streamed_gzip_restores_every_original_byte_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);episode,review=fixture(root);target=root/'fixture.tar.xz'
            original={str(p.relative_to(episode)):p.read_bytes() for p in episode.rglob('*') if p.is_file()}
            result=archive.pack(episode,review,target)
            self.assertEqual(result['verified_gzip_files'],1)
            self.assertEqual(result['verified_original_files'],5)
            self.assertFalse(result['original_episode_deleted'])
            restored=root/'restored';checked=archive.verify(target,restored)
            self.assertTrue(checked['strict_original_manifest_passed'])
            for name,data in original.items():
                self.assertEqual((episode/name).read_bytes(),data)
                self.assertEqual((restored/name).read_bytes(),data)
            with self.assertRaises(FileExistsError):archive.verify(target,restored)

    def test_failed_and_incomplete_episodes_are_not_packaged(self):
        for status in ('reserved','attempt_failed','wall_time_limit'):
            with self.subTest(status=status),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);episode,review=fixture(root)
                ledger=root/'phase/start_ledger.json';value=json.loads(ledger.read_text())
                value['entries'][0]['status']=status;write(ledger,value)
                with self.assertRaisesRegex(ValueError,'completed qualified'):
                    archive.pack(episode,review,root/'refused.tar.xz')
                self.assertFalse((root/'refused.tar.xz').exists())
                self.assertTrue(episode.is_dir())

    def test_noncanonical_gzip_cannot_claim_byte_exact_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);episode,review=fixture(root,gzip_mtime=10)
            with self.assertRaisesRegex(ValueError,'exactly reconstruct'):
                archive.pack(episode,review,root/'refused.tar.xz')

    def test_source_symlink_is_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);episode,review=fixture(root)
            (episode/'linked').symlink_to(episode/'result.json')
            with self.assertRaisesRegex(ValueError,'symlink'):
                archive.pack(episode,review,root/'refused.tar.xz')

    def test_traversal_in_packaging_mapping_rejected_before_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);episode,review=fixture(root)
            meta=archive.packaging_manifest(episode,archive.approved_episode(episode,review))
            meta=deepcopy(meta);meta['files'][0]['original_path']='../escape'
            target=root/'malicious.tar.xz';data=archive.canonical(meta)
            with tarfile.open(target,'w:xz') as tar:
                info=tarfile.TarInfo(archive.META);info.size=len(data);tar.addfile(info,io.BytesIO(data))
            with self.assertRaisesRegex(ValueError,'traversal'):
                archive.verify(target,root/'restore')
            self.assertFalse((root/'escape').exists());self.assertFalse((root/'restore').exists())

    def test_tar_symlink_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);episode,review=fixture(root)
            meta=archive.packaging_manifest(episode,archive.approved_episode(episode,review))
            target=root/'malicious.tar.xz';data=archive.canonical(meta)
            with tarfile.open(target,'w:xz') as tar:
                info=tarfile.TarInfo(archive.META);info.size=len(data);tar.addfile(info,io.BytesIO(data))
                link=tarfile.TarInfo(meta['files'][0]['stored_path']);link.type=tarfile.SYMTYPE
                link.linkname='../escape';tar.addfile(link)
            with self.assertRaisesRegex(ValueError,'links/nonregular'):
                archive.verify(target)
            self.assertFalse((root/'escape').exists())

    def test_truncated_xz_footer_is_not_accepted_after_tar_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);episode,review=fixture(root);target=root/'fixture.tar.xz'
            archive.pack(episode,review,target)
            broken=root/'truncated.tar.xz';broken.write_bytes(target.read_bytes()[:-8])
            with self.assertRaisesRegex(ValueError,'truncated XZ'):
                archive.verify(broken)


if __name__=='__main__':unittest.main()
