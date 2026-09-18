import io
import unittest
from unittest.mock import patch
import zipfile
from efa.remote_zip import RangeReader


class RemoteZipTests(unittest.TestCase):
    def test_zip_reads_across_small_cached_http_ranges(self):
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('selected.txt','original content '*200)
            archive.writestr('other.txt','untouched')
        data=buffer.getvalue();requests=[]
        def respond(request,timeout):
            start,end=map(int,request.headers['Range'].removeprefix('bytes=').split('-'))
            requests.append((start,end))
            response=io.BytesIO(data[start:end+1]);response.status=206
            response.headers={'Content-Range':f'bytes {start}-{end}/{len(data)}'}
            response.geturl=lambda:'https://example.invalid/public.zip'
            return response
        with patch('efa.remote_zip.urllib.request.urlopen',respond):
            with RangeReader('https://example.invalid/public.zip',block_size=64) as reader,zipfile.ZipFile(reader) as archive:
                self.assertEqual(archive.read('selected.txt'),b'original content '*200)
                self.assertLessEqual(max(end-start+1 for start,end in requests),64)

    def test_server_ignoring_ranges_is_not_downloaded(self):
        response=io.BytesIO(b'oversized body');response.status=200;response.headers={}
        with patch('efa.remote_zip.urllib.request.urlopen',return_value=response):
            with self.assertRaisesRegex(ValueError,'honor'):RangeReader('https://example.invalid/public.zip')
