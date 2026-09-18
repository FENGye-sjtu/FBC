"""Read the authorized teacher model catalog without persisting credentials."""
import getpass
import json
import urllib.request
import urllib.error


def main():
    key=getpass.getpass('Teacher API key: ')
    req=urllib.request.Request('https://dashscope.aliyuncs.com/compatible-mode/v1/models',
        headers={'Authorization':'Bearer '+key})
    try:
        with urllib.request.urlopen(req,timeout=45) as response:
            body=json.load(response)
        names=[x['id'] for x in body.get('data',[]) if 'vl' in x['id'].lower() or 'qwen3.5' in x['id'].lower()]
        print(json.dumps(dict(available_vision_models=names)))
    except urllib.error.HTTPError as error:
        print(json.dumps(dict(status=error.code,endpoint='dashscope.aliyuncs.com',reason=error.reason)))
        raise SystemExit(1)


if __name__=='__main__':
    main()
