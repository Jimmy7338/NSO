"""Install the sensor-only actor's private-resource/network audit boundary."""
import os
from pathlib import Path
import sys

def install_boundary(forbidden):
    roots=tuple(Path(p).resolve()for p in forbidden)
    blocked=[];read_files=set()
    def audit(event,args):
        if event=='open':
            value=args[0]
            if isinstance(value,(str,bytes,os.PathLike)):
                path=Path(os.fsdecode(value)).resolve()
                if any(path==r or r in path.parents for r in roots):
                    blocked.append(str(path))
                    raise PermissionError('Sensor-only actor cannot open private geometry or generation specification')
                read_files.add(str(path))
        if event in ('socket.connect','socket.bind'):
            raise PermissionError('Sensor-only actor has no network interface')
    sys.addaudithook(audit)
    return blocked,read_files
