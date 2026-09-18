"""Bounded HTTP ranges for reading selected members of a large public ZIP."""
from collections import OrderedDict
import io
import re
import urllib.request


class RangeReader(io.RawIOBase):
    def __init__(self,url,max_bytes=512*1024**2,block_size=1024**2):
        self.url=url;self.max_bytes=max_bytes;self.block_size=block_size
        self.position=0;self.bytes_read=0;self.cache=OrderedDict();self.size=None
        self._request(0,0)

    def _request(self,start,end):
        if self.bytes_read+end-start+1>self.max_bytes:raise ValueError('Remote ZIP transfer budget exceeded')
        request=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(request,timeout=60) as response:
            match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
            if response.status!=206 or not match or tuple(map(int,match.groups()[:2]))!=(start,end):
                raise ValueError('Server did not honor the requested bounded byte range')
            size=int(match[3])
            if self.size is not None and self.size!=size:raise ValueError('Archive size changed')
            self.size=size
            # Resolve the public download redirect once. Its temporary signed URL
            # is memory-only and never logged or included in artifact provenance.
            self.url=response.geturl()
            data=response.read(end-start+2)
        if len(data)!=end-start+1:raise ValueError('Short or oversized archive range')
        self.bytes_read+=len(data);return data

    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        pos=offset if whence==0 else self.position+offset if whence==1 else self.size+offset if whence==2 else -1
        if pos<0:raise ValueError('Invalid archive seek')
        self.position=pos;return pos

    def read(self,n=-1):
        n=max(0,self.size-self.position) if n<0 else min(n,max(0,self.size-self.position))
        chunks=[]
        while n:
            block=self.position//self.block_size;start=block*self.block_size
            if block not in self.cache:
                self.cache[block]=self._request(start,min(start+self.block_size,self.size)-1)
                if len(self.cache)>8:self.cache.popitem(last=False)
            self.cache.move_to_end(block)
            data=self.cache[block];offset=self.position-start;chunk=data[offset:offset+n]
            chunks.append(chunk);self.position+=len(chunk);n-=len(chunk)
        return b''.join(chunks)

    def readinto(self,buffer):
        data=self.read(len(buffer));buffer[:len(data)]=data;return len(data)
