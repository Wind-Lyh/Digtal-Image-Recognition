import cv2
import numpy as np
from typing import List


class ImageWarper:
    @staticmethod
    def get_canvas_size(imgs: List[np.ndarray], Hs: List[np.ndarray]) -> tuple:
        all_x = []
        all_y = []
        
        for i, (img, H) in enumerate(zip(imgs, Hs)):
            h, w = img.shape[:2]
            corners = np.array([
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1]
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            warped_corners = cv2.perspectiveTransform(corners, H)
            
            all_x.extend(warped_corners[:, 0, 0])
            all_y.extend(warped_corners[:, 0, 1])
        
        x_min = int(np.floor(np.min(all_x)))
        x_max = int(np.ceil(np.max(all_x)))
        y_min = int(np.floor(np.min(all_y)))
        y_max = int(np.ceil(np.max(all_y)))
        
        canvas_width = x_max - x_min + 1
        canvas_height = y_max - y_min + 1
        x_offset = -x_min
        y_offset = -y_min
        
        return canvas_width, canvas_height, x_offset, y_offset
    
    @staticmethod
    def warp_image(img: np.ndarray, H: np.ndarray, canvas_width, canvas_height, x_offset, y_offset) -> np.ndarray:
        M_offset = np.array([
            [1, 0, x_offset],
            [0, 1, y_offset],
            [0, 0, 1]
        ], dtype=np.float64)
        
        H_final = M_offset @ H
        
        warped = cv2.warpPerspective(
            img, H_final, (canvas_width, canvas_height),
            borderMode=cv2.BORDER_TRANSPARENT
        )
        
        return warped
