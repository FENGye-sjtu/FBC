"""Scoped SSH operations; password is read interactively and never persisted."""
import argparse
import getpass
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys

import paramiko

ROOT = "/data_cinema/zqc/VBC"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["deploy", "run", "upload", "publish"])
    parser.add_argument("--source")
    parser.add_argument("--destination", help="Path relative to remote VBC root")
    parser.add_argument("--script", help="Local shell script to execute on server")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--label", default="setup")
    args = parser.parse_args()
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect("202.120.36.100", port=2034, username="zqc", password=getpass.getpass("SSH password: "),
                   look_for_keys=False, allow_agent=False, timeout=20)
    sftp = client.open_sftp()

    def ensure(directory):
        if directory != ROOT and not directory.startswith(ROOT + "/"):
            raise ValueError("Remote path outside project root")
        # Resolve every existing component so a symlink cannot escape the root.
        try:
            resolved = sftp.normalize(directory)
            if resolved != ROOT and not resolved.startswith(ROOT + "/"):
                raise ValueError("Remote symlink escapes project root")
            sftp.stat(directory)
        except FileNotFoundError:
            ensure(str(PurePosixPath(directory).parent))
            sftp.mkdir(directory)

    if args.action == "deploy":
        repo = Path(__file__).resolve().parents[1]
        listing = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=repo, text=True)
        count = 0
        for name in sorted(set(listing.splitlines())):
            source = repo / name
            if name.startswith("third_party/") or not source.is_file():
                continue
            destination = ROOT + "/code/" + name.replace("\\", "/")
            ensure(str(PurePosixPath(destination).parent))
            sftp.put(str(source), destination)
            count += 1
        print(f"Deployed {count} source files to {ROOT}/code")
    elif args.action=="upload":
        if not args.source or not args.destination:
            parser.error("upload requires --source and --destination")
        relative=PurePosixPath(args.destination)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError("Destination must be project-relative")
        source=Path(args.source).resolve()
        if not source.is_dir():
            raise ValueError("Upload source must be a directory")
        count=0
        for path in source.rglob('*'):
            if not path.is_file() or path.is_symlink() or '.cache' in path.parts:
                continue
            target=ROOT+'/'+str(relative)+'/'+path.relative_to(source).as_posix()
            ensure(str(PurePosixPath(target).parent))
            sftp.put(str(path),target)
            count+=1
        print(f"Uploaded {count} files under {ROOT}/{relative}")
    elif args.action=="publish":
        from huggingface_hub import get_token
        if not args.source or not args.destination:
            parser.error('publish requires project-relative --source and dataset-relative --destination')
        relative=PurePosixPath(args.source)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Source must be project-relative')
        source=sftp.normalize(ROOT+'/'+str(relative))
        if not source.startswith(ROOT+'/'):
            raise ValueError('Source escapes the project')
        token=get_token()
        if not token:
            raise ValueError('Local Hugging Face login is required')
        # Credentials travel over the authenticated SSH channel, never in argv,
        # a shell script, a remote credential file or a printed request body.
        stdin,stdout,stderr=client.exec_command(ROOT+'/envs/efa/bin/python '+ROOT+'/code/scripts/publish_remote.py')
        stdin.write(json.dumps(dict(source=source,destination=args.destination,token=token)))
        stdin.channel.shutdown_write()
        for line in stdout:
            print(line,end='',flush=True)
        errors=stderr.read().decode()
        if errors:
            print(errors,file=sys.stderr)
        if stdout.channel.recv_exit_status():
            raise SystemExit('Remote data publication failed')
    else:
        if not args.script:
            parser.error("--script is required")
        body = Path(args.script).read_text(encoding="utf-8")
        if args.background:
            if not args.label.replace("_", "").replace("-", "").isalnum():
                raise ValueError("Invalid job label")
            dest = ROOT + "/tools/" + args.label + ".sh"
            ensure(ROOT + "/tools")
            with sftp.open(dest, "w") as stream:
                stream.write(body)
            cmd = (f"nohup bash {shlex.quote(dest)} > "
                   f"{ROOT}/logs/{args.label}.log 2>&1 < /dev/null & echo $!")
            _, stdout, stderr = client.exec_command(cmd)
            print(stdout.read().decode(), stderr.read().decode())
            print(f"Started {args.label}; log: {ROOT}/logs/{args.label}.log")
        else:
            stdin, stdout, stderr = client.exec_command("bash -s")
            stdin.write(body)
            stdin.channel.shutdown_write()
            for line in stdout:
                print(line, end="", flush=True)
            errors = stderr.read().decode()
            if errors:
                print(errors, file=sys.stderr)
            status = stdout.channel.recv_exit_status()
            if status:
                raise SystemExit(status)
    sftp.close()
    client.close()


if __name__ == "__main__":
    main()
