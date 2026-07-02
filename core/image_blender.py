import cv2
import numpy as np


class ImageBlender:
    @staticmethod
    def linear_blend(img1, img2, overlap_mask=None):
        if overlap_mask is None:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            mask1 = gray1 > 3
            mask2 = gray2 > 3
            overlap_mask = mask1 & mask2
        
        h, w = img1.shape[:2]
        w1 = np.zeros((h, w), dtype=np.float32)
        w1[img1[..., 0] > 3] = 1.0
        
        if np.any(overlap_mask):
            cols_overlap = np.where(np.any(overlap_mask, axis=0))[0]
            for y in range(h):
                row_overlap = overlap_mask[y, :]
                if np.any(row_overlap):
                    cols = np.where(row_overlap)[0]
                    l, r = cols[0], cols[-1]
                    if r > l:
                        alpha = np.linspace(1.0, 0.0, r - l + 1)
                        w1[y, l:r + 1] = alpha
        
        w1_3ch = np.stack([w1, w1, w1], axis=-1)
        blended = (w1_3ch * img1.astype(np.float32) +
                   (1.0 - w1_3ch) * img2.astype(np.float32))
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        return blended
    
    @staticmethod
    def multi_band_blend(img1, img2, mask, levels=5):
        pass
    
    @staticmethod
    def blend(img1, img2, mode='linear'):
        if mode == 'multiband':
            return ImageBlender.linear_blend(img1, img2)
        else:
            return ImageBlender.linear_blend(img1, img2)
