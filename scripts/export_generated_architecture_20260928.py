#!/usr/bin/env python3
"""Wrap the selected, unmodified generated PNG in a lossless single-page PDF.

This is format conversion, not raster retouching or a vector reconstruction.
Image generation and every visual edit were performed with built-in imagegen.
"""
from pathlib import Path
import hashlib
import json
import zlib

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'docs/thesis/figures/system_architecture_20260928'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = FIG / 'system_architecture.png'
    with Image.open(source) as im:
        if im.mode != 'RGB':
            raise ValueError('Expected RGB PNG; do not silently alter source pixels')
        width, height = im.size
        pixels = im.tobytes()
    page_width = 504.0
    page_height = page_width * height / width
    stream = zlib.compress(pixels, 9)
    drawing = f'q {page_width:.6f} 0 0 {page_height:.6f} 0 0 cm /Im0 Do Q\n'.encode()
    objects = [
        b'<< /Type /Catalog /Pages 2 0 R >>',
        b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
        (f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width:.6f} {page_height:.6f}] '
         '/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>').encode(),
        (f'<< /Type /XObject /Subtype /Image /Width {width} /Height {height} '
         f'/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length {len(stream)} >>\nstream\n').encode()
        + stream + b'\nendstream',
        f'<< /Length {len(drawing)} >>\nstream\n'.encode() + drawing + b'endstream',
    ]
    pdf = bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f'{number} 0 obj\n'.encode() + body + b'\nendobj\n')
    start = len(pdf)
    pdf.extend(f'xref\n0 {len(objects) + 1}\n0000000000 65535 f \n'.encode())
    for offset in offsets[1:]:
        pdf.extend(f'{offset:010d} 00000 n \n'.encode())
    pdf.extend((f'trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n'
                f'startxref\n{start}\n%%EOF\n').encode())
    target = FIG / 'system_architecture.pdf'
    target.write_bytes(pdf)
    assert zlib.decompress(stream) == pixels
    receipt = {
        'source_png': str(source.relative_to(ROOT)),
        'source_sha256': digest(source),
        'pdf_sha256': digest(target),
        'pixels': [width, height],
        'rgb_pixels_sha256': hashlib.sha256(pixels).hexdigest(),
        'pdf_type': 'lossless embedded raster, not vector',
        'resampling': False,
        'image_edits_by_exporter': False,
        'script_sha256': digest(Path(__file__)),
    }
    (FIG / 'export_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
