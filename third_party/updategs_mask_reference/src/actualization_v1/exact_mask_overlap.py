"""Exact packed-bit mask overlap backend; intentionally not wired into production."""
import numpy as np

# Keep the helper importable under the official Python 3.8 environments.
LUT = np.array([bin(int(i)).count("1") for i in range(256)], dtype=np.uint8)

class ExactMaskOverlap:
    def __init__(self, masks):
        a = np.asarray(masks, dtype=bool)
        if a.ndim != 3: raise ValueError("masks must be N,H,W")
        self.shape = a.shape
        self.masks = a
        self.packed = np.packbits(a.reshape(len(a), a.shape[1] * a.shape[2]), axis=1, bitorder="little")
        self.areas = a.reshape(len(a), a.shape[1] * a.shape[2]).sum(axis=1, dtype=np.int64)
        self.pairint_count = 0

    def intersection(self, i, j):
        self.pairint_count += 1
        return int(LUT[np.bitwise_and(self.packed[i], self.packed[j])].sum(dtype=np.int64))

    def iou(self, i, j):
        inter = self.intersection(i, j)
        return inter / max(int(self.areas[i] + self.areas[j] - inter), 1)

    def matrix(self):
        n = len(self.masks); out = np.zeros((n,n), np.float64)
        for i in range(n):
            for j in range(i,n): out[i,j] = out[j,i] = self.iou(i,j)
        return out


def exact_nms(masks, scores, iou_threshold=.7):
    """Byte-for-byte semantic copy of scar_mask_proposals.mask_nms with exact overlap."""
    ex = ExactMaskOverlap(masks); scores=np.asarray(scores,float); order=np.argsort(-scores); keep=[]
    for idx in order:
        if all(ex.iou(int(idx), j) < iou_threshold for j in keep): keep.append(int(idx))
    return np.asarray(keep, dtype=np.int64)
