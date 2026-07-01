from typing import List, Tuple

import cv2
import numpy as np


class ImageWarper:
    """图像透视变换器"""
    
    @staticmethod
    def get_canvas_size(imgs: List[np.ndarray], Hs: List[np.ndarray]) -> Tuple[int, int, int, int]:
        """
        精确计算透视变换后四个顶点的极限坐标，算出画布尺寸和平移偏移量
        
        Args:
            imgs: 图像列表
            Hs: 对应的单应矩阵列表
            
        Returns:
            tuple: (canvas_width, canvas_height, dx, dy)
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
        
        # 精确计算最小负偏移量 dx, dy (确保将所有负坐标平移至非负数区间)
        dx = max(0, -x_min)
        dy = max(0, -y_min)
        
        canvas_width = int(np.ceil(x_max - x_min))
        canvas_height = int(np.ceil(y_max - y_min))
        
        return canvas_width, canvas_height, int(dx), int(dy)
    
    @staticmethod
    def warp_image(img: np.ndarray, H: np.ndarray, 
                   canvas_width: int, canvas_height: int, 
                   dx: int, dy: int) -> np.ndarray:
        """
        构造平移矩阵更新单应矩阵，防止画面重叠与超出画布
        
        Args:
            img: 输入图像
            H: 单应矩阵
            canvas_width: 画布宽度
            canvas_height: 画布高度
            dx: x方向极小负偏移量
            dy: y方向极小负偏移量
            
        Returns:
            绝对安全无截断的变换后图像
        """
        # 构造精确的平移矩阵 M_shift
        # M_shift = [ 1  0  dx ]
        #           [ 0  1  dy ]
        #           [ 0  0  1  ]
        M_shift = np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # 将单应矩阵更新为 H_new = M_shift * H
        # 确保图像在扭曲后绝对不会超出画布左侧或上侧边界
        H_new = M_shift @ H
        
        warped = cv2.warpPerspective(
            img, 
            H_new, 
            (canvas_width, canvas_height),
            borderMode=cv2.BORDER_TRANSPARENT
        )
        
        return warped
