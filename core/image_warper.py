from typing import List, Tuple

import cv2
import numpy as np


class ImageWarper:
    """图像透视变换器"""
    
    @staticmethod
    def get_canvas_size(imgs: List[np.ndarray], Hs: List[np.ndarray]) -> Tuple[int, int, int, int]:
        """
        计算变换后画布的尺寸和偏移量
        
        Args:
            imgs: 图像列表
            Hs: 对应的单应矩阵列表
            
        Returns:
            tuple: (canvas_width, canvas_height, x_offset, y_offset)
                canvas_width: 画布宽度
                canvas_height: 画布高度
                x_offset: x方向偏移量（非负）
                y_offset: y方向偏移量（非负）
        """
        all_corners = []
        
        for img, H in zip(imgs, Hs):
            h, w = img.shape[:2]
            corners = np.array([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            transformed_corners = cv2.perspectiveTransform(corners, H)
            all_corners.extend(transformed_corners.reshape(-1, 2))
        
        all_corners = np.array(all_corners)
        x_min = np.min(all_corners[:, 0])
        y_min = np.min(all_corners[:, 1])
        x_max = np.max(all_corners[:, 0])
        y_max = np.max(all_corners[:, 1])
        
        x_offset = max(0, -x_min)
        y_offset = max(0, -y_min)
        
        canvas_width = int(np.ceil(x_max - x_min))
        canvas_height = int(np.ceil(y_max - y_min))
        
        return canvas_width, canvas_height, int(x_offset), int(y_offset)
    
    @staticmethod
    def warp_image(img: np.ndarray, H: np.ndarray, 
                   canvas_width: int, canvas_height: int, 
                   x_offset: int, y_offset: int) -> np.ndarray:
        """
        对图像进行透视变换并放置到画布上
        
        Args:
            img: 输入图像
            H: 单应矩阵
            canvas_width: 画布宽度
            canvas_height: 画布高度
            x_offset: x方向偏移量
            y_offset: y方向偏移量
            
        Returns:
            变换后的图像
        """
        translation_matrix = np.array([
            [1, 0, x_offset],
            [0, 1, y_offset],
            [0, 0, 1]
        ], dtype=np.float64)
        
        H_with_offset = translation_matrix @ H
        
        warped = cv2.warpPerspective(
            img, 
            H_with_offset, 
            (canvas_width, canvas_height),
            borderMode=cv2.BORDER_TRANSPARENT
        )
        
        return warped
