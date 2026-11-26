"""Fetch only the licensed example navmesh from a pinned official tar.

The reference derives from the scan and checks consistency with the source
simulator. It does not change the staged scene or the primary collision model.
"""

import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request


OUT = Path(__file__).resolve().parent
REPO = "matterport/habitat-matterport-3dresearch"
REVISION = "9b74d5aa58e371c9bc16b2ccb64748cddf9bf1c6"
TAR_PATH = "example/hm3d-example-habitat-v0.2.tar"
HOST = f"https://media.githubusercontent.com/media/{REPO}/{REVISION}/{TAR_PATH}"
RAW = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{TAR_PATH}"
LICENSE = "https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data"
TARGET_SOURCE = "00770-NBg5UqG3di3"


def http(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={"User-Agent": "NSO-academic-navmesh-check", **(headers or {})}), timeout=20)


def span(start, n, total):
    end = start + n - 1
    with http(HOST, {"Range": f"bytes={start}-{end}"}) as stream:
        if stream.status != 206 or stream.headers["Content-Range"] != f"bytes {start}-{end}/{total}":
            raise RuntimeError(f"Expected partial range, got {stream.status}: {stream.headers.get('Content-Range')}")
        result = stream.read()
    if len(result) != n:
        raise ValueError("Source returned incomplete range")
    return result


def main():
    with http(RAW) as stream:
        pointer = stream.read(1024).decode()
    source = dict(line.split(" ", 1) for line in pointer.splitlines() if " " in line)
    total = int(source["size"])
    if not source["oid"].startswith("sha256:"):
        raise ValueError("Unrecognized official LFS pointer")
    # The pinned archive uses ordinary short POSIX names in 512-byte tar headers.
    offset, found, members = 0, [], []
    while offset < total - 512:
        header = span(offset, 512, total)
        if header == b"\x00" * 512:
            break
        member = tarfile.TarInfo.frombuf(header, encoding="utf-8", errors="strict")
        members.append(dict(name=member.name, bytes=member.size, type=member.type.decode(), header_offset=offset))
        if member.isfile() and member.name.startswith(TARGET_SOURCE + "/") and member.name.endswith(".navmesh"):
            blob = span(offset + 512, member.size, total)
            path = OUT / Path(member.name).name
            with path.open("xb") as stream:
                stream.write(blob)
            found.append(dict(member=member.name, path=path.name, bytes=member.size,
                              sha256=hashlib.sha256(blob).hexdigest()))
        offset += 512 + ((member.size + 511) // 512) * 512
    if not found:
        raise RuntimeError("Official requested scan navmesh not present")
    receipt = {"official_archive": HOST, "revision": REVISION, "archive_size": total,
               "pinned_lfs_archive_sha256": source["oid"].removeprefix("sha256:"),
               "archive_hash_recomputed": False, "download_scope": "POSIX tar headers and matching navmesh only",
               "academic_use_license": LICENSE, "source": TARGET_SOURCE, "matches": found, "members": members}
    with (OUT / "hm3d_navmesh_source_receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(found))


if __name__ == "__main__":
    main()
