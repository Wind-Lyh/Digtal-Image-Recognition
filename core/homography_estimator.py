import cv2
import numpy as np

import config


class InsufficientPointsError(Exception):
    """输入特征点数量不足时抛出的异常"""
    pass


class HomographyEstimationError(Exception):
    """单应矩阵计算失败时抛出的异常"""
    pass


class HomographyEstimator:
    """单应矩阵估计器"""
    
    @staticmethod
    def compute(src_pts: np.ndarray, dst_pts: np.ndarray, 
                method: int = cv2.RANSAC, 
                ransac_thresh: float = config.RANSAC_THRESH) -> tuple:
        """
        计算单应矩阵
        
        Args:
            src_pts: 源图像特征点，形状 (N, 2)
            dst_pts: 目标图像特征点，形状 (N, 2)
            method: 计算方法，默认使用RANSAC
            ransac_thresh: RANSAC重投影阈值
            
        Returns:
            tuple: (单应矩阵H, 掩码mask)
                H: 3x3单应矩阵
                mask: 形状 (N,) 的numpy数组，标识内点
                
        Raises:
            InsufficientPointsError: 当输入特征点数量小于4时
            HomographyEstimationError: 当单应矩阵计算失败时
        """
        if src_pts.shape[0] < 4 or dst_pts.shape[0] < 4:
            raise InsufficientPointsError(
                f"特征点数量不足，需要至少4个点，实际得到 {min(src_pts.shape[0], dst_pts.shape[0])} 个"
            )
        
        if src_pts.shape[0] != dst_pts.shape[0]:
            raise ValueError("源点和目标点数量必须相等")
        
        H, mask = cv2.findHomography(src_pts, dst_pts, method, ransac_thresh)
        
        if H is None:
            raise HomographyEstimationError("单应矩阵计算失败")
        
        return H, mask
