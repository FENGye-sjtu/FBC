"""Exact-object, fast-forward fallback when Git HTTPS transport is unavailable."""
import base64
import datetime
import json
import subprocess
import urllib.request


def git(*args, data=None):
    return subprocess.check_output(['git', *args], input=data)


def main():
    credential=git('credential','fill',data=b'protocol=https\nhost=github.com\n\n')
    fields=dict(line.split('=',1) for line in credential.decode().splitlines() if '=' in line)
    token=fields['password']
    base='https://api.github.com/repos/FENGye-sjtu/FBC/'

    def api(path,payload=None,method=None):
        request=urllib.request.Request(base+path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
                     'Content-Type':'application/json'},method=method)
        with urllib.request.urlopen(request,timeout=40) as response:
            return json.load(response)

    if git('status','--porcelain').strip():
        raise RuntimeError('Commit verified changes first')
    remote=api('git/ref/heads/main')['object']['sha']
    subprocess.run(['git','merge-base','--is-ancestor',remote,'HEAD'],check=True)
    commits=git('rev-list','--reverse',remote+'..HEAD').decode().splitlines()
    for commit in commits:
        raw=git('cat-file','commit',commit).decode()
        header,message=raw.split('\n\n',1)
        headers=[line.split(' ',1) for line in header.splitlines()]
        parents=[v for k,v in headers if k=='parent']
        if len(parents)!=1 or parents[0]!=remote:
            raise RuntimeError('Only a linear fast-forward is supported')
        meta=dict(headers)
        changed=git('diff-tree','--no-commit-id','--name-only','-r','-z',commit).split(b'\0')
        changed={p.decode() for p in changed if p}
        tree=[]
        for entry in git('ls-tree','-rz',commit).split(b'\0'):
            if not entry:continue
            info,path=entry.split(b'\t',1)
            mode,kind,sha=info.decode().split();path=path.decode()
            if path in changed and kind=='blob':
                content=git('cat-file','blob',sha)
                created=api('git/blobs',{'content':base64.b64encode(content).decode(),'encoding':'base64'})
                if created['sha']!=sha:raise RuntimeError('Blob hash mismatch')
            tree.append(dict(path=path,mode=mode,type=kind,sha=sha))
        created=api('git/trees',{'tree':tree})
        if created['sha']!=meta['tree']:raise RuntimeError('Tree hash mismatch')

        def identity(value):
            person,stamp,offset=value.rsplit(' ',2)
            name,email=person.rsplit(' <',1)
            minutes=(int(offset[1:3])*60+int(offset[3:]))*(1 if offset[0]=='+' else -1)
            zone=datetime.timezone(datetime.timedelta(minutes=minutes))
            date=datetime.datetime.fromtimestamp(int(stamp),zone).isoformat()
            return dict(name=name,email=email.rstrip('>'),date=date)

        created=api('git/commits',dict(message=message,tree=meta['tree'],parents=parents,
            author=identity(meta['author']),committer=identity(meta['committer'])))
        if created['sha']!=commit:
            raise RuntimeError('Commit hash mismatch; branch was not advanced')
        # A concurrent branch change requires reconciliation, never force.
        if api('git/ref/heads/main')['object']['sha']!=remote:
            raise RuntimeError('Remote changed concurrently; reconcile before retry')
        api('git/refs/heads/main',dict(sha=commit,force=False),method='PATCH')
        remote=commit
        print('Published exact commit '+commit,flush=True)
    if api('git/ref/heads/main')['object']['sha']!=git('rev-parse','HEAD').decode().strip():
        raise RuntimeError('Remote verification failed')
    git('update-ref','refs/remotes/origin/main',remote)
    print('Remote main matches local HEAD',flush=True)


if __name__=='__main__':
    main()
