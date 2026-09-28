#!/usr/bin/env python3
"""Restore exact original V1 PLY paths from their independently verified archive."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
sys.dont_write_bytecode=True
import argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.archive_inspection_meshes_v24 import DEFAULT_ARCHIVE,restore

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=DEFAULT_ARCHIVE)
    p.add_argument('--only',help='Restore exactly this inventoried project-relative path; omit for all 464')
    a=p.parse_args();restore(a.archive.absolute(),a.only)

if __name__=='__main__':main()
