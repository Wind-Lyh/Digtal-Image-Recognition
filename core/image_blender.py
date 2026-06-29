import cv2
import numpy as np

import config


class ImageBlender:
    """图像融合器"""
    
    @staticmethod
    def blend(img1: np.ndarray, img2: np.ndarray, overlap_mask=None, mode: str = 'linear', levels: int = 5) -> np.ndarray:
        """
        统一融合接口，根据模式选择融合方式
        
        Args:
            img1: 第一张图像
            img2: 第二张图像
            overlap_mask: 重叠区域掩码（可选）
            mode: 融合模式，'linear' 或 'multiband'
            levels: 多频段金字塔层数（仅用于multiband模式）
            
        Returns:
            融合后的图像
        """
        if mode == 'linear':
            return ImageBlender.linear_blend(img1, img2, overlap_mask)
        elif mode == 'multiband':
            return ImageBlender.multi_band_blend(img1, img2, overlap_mask, levels)
        else:
            raise ValueError(f"不支持的融合模式: {mode}")
    
    @staticmethod
    def linear_blend(img1: np.ndarray, img2: np.ndarray, overlap_mask=None) -> np.ndarray:
        """
        加权平均融合
        
        Args:
            img1: 第一张图像
            img2: 第二张图像
            overlap_mask: 重叠区域掩码（可选），若未提供则自动生成
            
        Returns:
            融合后的图像
        """
        if img1.shape != img2.shape:
            raise ValueError("两张图像必须具有相同的尺寸")
        
        if overlap_mask is None:
            mask1 = np.any(img1 != 0, axis=2).astype(np.float32)
            mask2 = np.any(img2 != 0, axis=2).astype(np.float32)
            overlap_mask = mask1 * mask2
        
        mask1_only = (np.any(img1 != 0, axis=2) & (overlap_mask == 0)).astype(np.float32)
        mask2_only = (np.any(img2 != 0, axis=2) & (overlap_mask == 0)).astype(np.float32)
        
        rows, cols = overlap_mask.shape
        alpha = np.zeros((rows, cols), dtype=np.float32)
        
        for r in range(rows):
            cols_in_overlap = np.where(overlap_mask[r] > 0)[0]
            if len(cols_in_overlap) > 0:
                min_col = cols_in_overlap.min()
                max_col = cols_in_overlap.max()
                if max_col > min_col:
                    alpha[r, cols_in_overlap] = 1.0 - (cols_in_overlap - min_col) / (max_col - min_col)
                else:
                    alpha[r, cols_in_overlap] = 0.5
        
        alpha = np.stack([alpha, alpha, alpha], axis=2)
        
        result = np.zeros_like(img1, dtype=np.float32)
        
        img1_float = img1.astype(np.float32)
        img2_float = img2.astype(np.float32)
        
        result = alpha * img1_float + (1 - alpha) * img2_float
        
        mask1_only_3d = np.stack([mask1_only, mask1_only, mask1_only], axis=2)
        mask2_only_3d = np.stack([mask2_only, mask2_only, mask2_only], axis=2)
        
        result = result * (1 - mask1_only_3d) + img1_float * mask1_only_3d
        result = result * (1 - mask2_only_3d) + img2_float * mask2_only_3d
        
        return result.astype(np.uint8)
    
    @staticmethod
    def multi_band_blend(img1: np.ndarray, img2: np.ndarray, overlap_mask=None, levels: int = 5) -> np.ndarray:
        """
        多频段金字塔融合（基于拉普拉斯金字塔）
        
        Args:
            img1: 第一张图像
            img2: 第二张图像
            overlap_mask: 重叠区域掩码（可选）
            levels: 金字塔层数
            
        Returns:
            融合后的图像
        """
        if img1.shape != img2.shape:
            raise ValueError("两张图像必须具有相同的尺寸")
        
        if overlap_mask is None:
            mask1 = np.any(img1 != 0, axis=2).astype(np.float32)
            mask2 = np.any(img2 != 0, axis=2).astype(np.float32)
            overlap_mask = mask1 * mask2
        
        rows, cols = overlap_mask.shape
        alpha = np.zeros((rows, cols), dtype=np.float32)
        
        for r in range(rows):
            cols_in_overlap = np.where(overlap_mask[r] > 0)[0]
            if len(cols_in_overlap) > 0:
                min_col = cols_in_overlap.min()
                max_col = cols_in_overlap.max()
                if max_col > min_col:
                    alpha[r, cols_in_overlap] = 1.0 - (cols_in_overlap - min_col) / (max_col - min_col)
                else:
                    alpha[r, cols_in_overlap] = 0.5
        
        alpha = np.stack([alpha, alpha, alpha], axis=2)
        
        img1_float = img1.astype(np.float32)
        img2_float = img2.astype(np.float32)
        
        gaussian_pyr1 = [img1_float]
        gaussian_pyr2 = [img2_float]
        gaussian_pyr_alpha = [alpha]
        
        for _ in range(levels):
            img1_float = cv2.pyrDown(img1_float)
            img2_float = cv2.pyrDown(img2_float)
            alpha = cv2.pyrDown(alpha)
            
            gaussian_pyr1.append(img1_float)
            gaussian_pyr2.append(img2_float)
            gaussian_pyr_alpha.append(alpha)
        
        laplacian_pyr1 = []
        laplacian_pyr2 = []
        
        for i in range(levels):
            next_level = cv2.pyrUp(gaussian_pyr1[i + 1])
            if next_level.shape != gaussian_pyr1[i].shape:
                next_level = cv2.resize(next_level, (gaussian_pyr1[i].shape[1], gaussian_pyr1[i].shape[0]))
            laplacian_pyr1.append(gaussian_pyr1[i] - next_level)
            
            next_level = cv2.pyrUp(gaussian_pyr2[i + 1])
            if next_level.shape != gaussian_pyr2[i].shape:
                next_level = cv2.resize(next_level, (gaussian_pyr2[i].shape[1], gaussian_pyr2[i].shape[0]))
            laplacian_pyr2.append(gaussian_pyr2[i] - next_level)
        
        laplacian_pyr1.append(gaussian_pyr1[-1])
        laplacian_pyr2.append(gaussian_pyr2[-1])
        
        blended_pyr = []
        for i in range(levels + 1):
            alpha_level = gaussian_pyr_alpha[i]
            blended = alpha_level * laplacian_pyr1[i] + (1 - alpha_level) * laplacian_pyr2[i]
            blended_pyr.append(blended)
        
        result = blended_pyr[-1]
        for i in range(levels - 1, -1, -1):
            result = cv2.pyrUp(result)
            if result.shape != blended_pyr[i].shape:
                result = cv2.resize(result, (blended_pyr[i].shape[1], blended_pyr[i].shape[0]))
            result = result + blended_pyr[i]
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        mask1_only = (np.any(img1 != 0, axis=2) & (overlap_mask[:, :, 0] == 0)).astype(np.float32)
        mask2_only = (np.any(img2 != 0, axis=2) & (overlap_mask[:, :, 0] == 0)).astype(np.float32)
        
        mask1_only_3d = np.stack([mask1_only, mask1_only, mask1_only], axis=2)
        mask2_only_3d = np.stack([mask2_only, mask2_only, mask2_only], axis=2)
        
        result_float = result.astype(np.float32)
        result_float = result_float * (1 - mask1_only_3d) + img1.astype(np.float32) * mask1_only_3d
        result_float = result_float * (1 - mask2_only_3d) + img2.astype(np.float32) * mask2_only_3d
        
        return result_float.astype(np.uint8)
