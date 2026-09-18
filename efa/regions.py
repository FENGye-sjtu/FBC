"""Original-pixel geometry. No full-image encoding survives target deletion."""
from dataclasses import dataclass
from hashlib import sha256
import json


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    def valid(self, width, height):
        return 0 <= self.x1 < self.x2 <= width and 0 <= self.y1 < self.y2 <= height

    def overlaps(self, other):
        return (max(self.x1, other.x1) < min(self.x2, other.x2)
                and max(self.y1, other.y1) < min(self.y2, other.y2))

    def expand(self, width, height, ratio=0.25):
        dx, dy = round((self.x2-self.x1)*ratio), round((self.y2-self.y1)*ratio)
        return Box(max(0, self.x1-dx), max(0, self.y1-dy), min(width, self.x2+dx), min(height, self.y2+dy))

    def subtract(self, target):
        if not self.overlaps(target):
            return [self]
        ix1, iy1 = max(self.x1, target.x1), max(self.y1, target.y1)
        ix2, iy2 = min(self.x2, target.x2), min(self.y2, target.y2)
        pieces = [Box(self.x1,self.y1,self.x2,iy1), Box(self.x1,iy2,self.x2,self.y2),
                  Box(self.x1,iy1,ix1,iy2), Box(ix2,iy1,self.x2,iy2)]
        return [p for p in pieces if p.x1 < p.x2 and p.y1 < p.y2]


def complement(width, height, targets):
    pieces = [Box(0, 0, width, height)]
    for target in targets:
        pieces = [p for box in pieces for p in box.subtract(target)]
    return pieces


def source_version(image_digest, query, boxes, encoder):
    payload = [image_digest, query, [vars(b) for b in boxes], encoder]
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
