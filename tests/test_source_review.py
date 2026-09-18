import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from efa.review import review


class SourceReviewTests(unittest.TestCase):
    def test_full_context_cannot_approve_an_unsupported_target_crop(self):
        row=dict(image='image.png',answer='blue',steps=[dict(id=1,parents=[],atoms=[dict(
            subject='bag',predicate='color',value='blue',kind='visual',
            region=dict(boxes=[[0,0,10,10]],context_boxes=[[0,0,100,100]],insufficient=False))])])
        calls=[]
        def teacher(messages,cache,stage):
            payload=json.loads(messages[1]['content'][-1]['text']);calls.append(payload)
            return dict(output=dict(supported=stage=='source_review_expanded',insufficient=False))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);Image.new('RGB',(100,100),'blue').save(root/'image.png')
            with patch('efa.review.teacher_json',teacher):result=review(row,root,root)
        self.assertEqual(result['annotation_status'],'teacher_review_rejected')
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0]['original_coordinates_in_image_order'],[[0,0,10,10]])
        self.assertEqual([r['view'] for r in result['review']['checks']],['target','expanded'])
