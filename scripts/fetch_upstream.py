"""Fetch the pinned DeepEyesV2 source; no git-history transfer required."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

root=Path(__file__).resolve().parents[1]
lock=json.loads((root/"configs/upstream.json").read_text())
url=f"https://codeload.github.com/Visual-Agent/DeepEyesV2/zip/{lock['revision']}"
with urllib.request.urlopen(url,timeout=120) as r:
    raw=r.read()
destination=root/"third_party/DeepEyesV2"
destination.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    for info in archive.infolist():
        components=Path(info.filename).parts[1:]
        if not components or info.is_dir():
            continue
        target=destination.joinpath(*components).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise ValueError("Unsafe upstream archive member")
        content=archive.read(info)
        if target.exists() and target.read_bytes()!=content:
            raise RuntimeError(f"Refusing to overwrite modified upstream file: {target}")
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content)
(destination/"UPSTREAM_ARCHIVE.json").write_text(json.dumps(dict(lock,archive_sha256=hashlib.sha256(raw).hexdigest()),indent=2))
print(f"DeepEyesV2 {lock['revision']}: {len(raw)} bytes downloaded to {destination}")
