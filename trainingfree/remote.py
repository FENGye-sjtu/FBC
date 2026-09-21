"""Private stdin SSH session; credentials are never saved or logged.

First stdin line: password. Subsequent lines: JSON requests, one per line.
Requests: {"cmd": "..."} or {"upload": "local path", "target": "remote path"}.
"""
import json
from pathlib import Path, PurePosixPath
import sys

import paramiko

ROOTS = (PurePosixPath('/data_cinema/zqc/CVPR'), PurePosixPath('/data_wyn/zqc/CVPR'))


def main():
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('202.120.36.100', port=2034, username='zqc',
                   password=sys.stdin.readline().rstrip('\r\n'), timeout=30)
    print('CONNECTED', flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if 'upload' in request:
                target = PurePosixPath(request['target'])
                if '..' in target.parts or not any(r in target.parents for r in ROOTS):
                    raise ValueError('Upload outside the authorized CVPR directories')
                with client.open_sftp() as sftp:
                    # Do not follow existing directory symlinks out of the workspace.
                    parent = target.parent
                    missing = []
                    while True:
                        try:
                            resolved = PurePosixPath(sftp.normalize(str(parent)))
                            sftp.stat(str(parent))
                            if not any(resolved == r or r in resolved.parents for r in ROOTS):
                                raise ValueError('Remote parent resolves outside CVPR')
                            break
                        except FileNotFoundError:
                            missing.append(parent)
                            parent = parent.parent
                    for item in reversed(missing):
                        sftp.mkdir(str(item))
                    sftp.put(str(Path(request['upload'])), str(target))
                print(json.dumps({'uploaded': str(target)}), flush=True)
            elif 'download' in request:
                source = PurePosixPath(request['download'])
                if '..' in source.parts or not any(r in source.parents for r in ROOTS):
                    raise ValueError('Download outside authorized CVPR directories')
                target = Path(request['target']).resolve()
                target.parent.mkdir(parents=True,exist_ok=True)
                with client.open_sftp() as sftp:
                    sftp.get(str(source),str(target))
                print(json.dumps({'downloaded':str(target)}),flush=True)
            else:
                _, stdout, stderr = client.exec_command(request['cmd'])
                print(stdout.read().decode('utf-8', errors='replace'), flush=True)
                print(stderr.read().decode('utf-8', errors='replace'), flush=True)
                print('EXIT', stdout.channel.recv_exit_status(), flush=True)
        except Exception as exc:
            print(type(exc).__name__, str(exc), flush=True)
    client.close()


if __name__ == '__main__':
    main()
