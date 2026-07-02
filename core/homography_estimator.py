import cv2
import numpy as np


class InsufficientPointsError(Exception):
    pass


class HomographyEstimationError(Exception):
    pass


class HomographyEstimator:
    @staticmethod
    def compute(src_pts, dst_pts, method=cv2.RANSAC, ransac_thresh=3.0):
        if len(src_pts) < 4 or len(dst_pts) < 4:
            raise InsufficientPointsError(f"匹配点数量不足: {len(src_pts)} < 4")
        
        src_pts = np.float32(src_pts).reshape(-1, 1, 2)
        dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, method, ransac_thresh)
        
        if H is None:
            raise HomographyEstimationError("单应矩阵计算失败")
        
        mask = mask.ravel() if mask is not None else np.ones(len(src_pts), dtype=np.uint8)
        
        return H, mask
