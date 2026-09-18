"""Launch the approved DashScope pilot; do not persist the teacher credential."""
import getpass
import os
import sys


if __name__=='__main__':
    os.environ['EFA_TEACHER_BASE_URL']='https://dashscope.aliyuncs.com/compatible-mode/v1'
    os.environ.setdefault('EFA_TEACHER_MODEL','qwen3.5-flash-2026-02-23')
    os.environ['EFA_TEACHER_API_KEY']=getpass.getpass('Teacher API key: ')
    if '--review' in sys.argv:
        sys.argv.remove('--review')
        from efa.review import main
    else:
        from efa.annotate import main
    main()
