"""Seekable read-only HTTP ranges for selected members of a large source archive."""
import io
import urllib.request


class HTTPRangeReader(io.RawIOBase):
    def __init__(self,url,max_bytes=256*1024*1024):
        super().__init__()
        self.url=url;self.position=0;self.transferred=0;self.max_bytes=max_bytes
        req=urllib.request.Request(url,headers={'Range':'bytes=0-0','Accept-Encoding':'identity'})
        with urllib.request.urlopen(req,timeout=60) as response:
            if response.status!=206:raise ValueError('Archive server does not support byte ranges')
            self.size=int(response.headers['Content-Range'].split('/')[-1])
            response.read(1)

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position

    def seek(self,offset,whence=0):
        position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset if whence==2 else -1
        if position<0:raise ValueError('Invalid archive seek')
        self.position=position;return position

    def read(self,size=-1):
        count=max(0,self.size-self.position) if size<0 else max(0,min(size,self.size-self.position))
        if not count:return b''
        if self.transferred+count>self.max_bytes:raise ValueError('Archive transfer budget exceeded')
        end=self.position+count-1
        req=urllib.request.Request(self.url,headers={'Range':f'bytes={self.position}-{end}','Accept-Encoding':'identity'})
        with urllib.request.urlopen(req,timeout=90) as response:
            expected=f'bytes {self.position}-{end}/{self.size}'
            if response.status!=206 or response.headers.get('Content-Range')!=expected:
                raise ValueError('Archive range response mismatch')
            value=response.read(count+1)
        if len(value)!=count:raise ValueError('Incomplete archive range')
        self.position+=count;self.transferred+=count
        return value
