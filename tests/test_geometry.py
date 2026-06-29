import numpy as np
import pytest

from core.homography_estimator import HomographyEstimator, InsufficientPointsError
from core.image_warper import ImageWarper
from utils.image_io import crop_black_border


class TestHomographyEstimator:
    """单应矩阵估计器测试类"""
    
    def test_insufficient_points(self):
        """测试输入点数小于4时抛出异常"""
        src_pts = np.array([[0, 0], [100, 0], [100, 100]], dtype=np.float64)
        dst_pts = np.array([[0, 0], [100, 0], [100, 100]], dtype=np.float64)
        
        with pytest.raises(InsufficientPointsError):
            HomographyEstimator.compute(src_pts, dst_pts)
    
    def test_compute_homography(self):
        """测试正常计算单应矩阵"""
        src_pts = np.array([
            [0, 0],
            [100, 0],
            [100, 100],
            [0, 100]
        ], dtype=np.float64)
        
        dst_pts = np.array([
            [50, 50],
            [150, 50],
            [150, 150],
            [50, 150]
        ], dtype=np.float64)
        
        H, mask = HomographyEstimator.compute(src_pts, dst_pts)
        
        assert H is not None
        assert H.shape == (3, 3)
        assert mask is not None
        assert mask.shape == (4,)


class TestImageWarper:
    """图像变换器测试类"""
    
    def test_get_canvas_size_single_image(self):
        """测试单张图返回正确的画布尺寸"""
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        H = np.eye(3)
        
        canvas_width, canvas_height, x_offset, y_offset = ImageWarper.get_canvas_size([img], [H])
        
        assert canvas_width == 600
        assert canvas_height == 400
        assert x_offset == 0
        assert y_offset == 0
    
    def test_get_canvas_size_with_transform(self):
        """测试带变换的画布尺寸计算"""
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        
        H = np.array([
            [1, 0, 100],
            [0, 1, 50],
            [0, 0, 1]
        ], dtype=np.float64)
        
        canvas_width, canvas_height, x_offset, y_offset = ImageWarper.get_canvas_size([img], [H])
        
        assert canvas_width >= 700  # 600 + 100
        assert canvas_height >= 450  # 400 + 50
        assert x_offset == 0
        assert y_offset == 0


class TestImageIO:
    """图像IO工具测试类"""
    
    def test_crop_black_border(self):
        """测试裁剪纯黑边"""
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        
        img[20:180, 30:270] = [255, 128, 64]
        
        cropped = crop_black_border(img)
        
        assert cropped.shape[0] == 160  # 180 - 20
        assert cropped.shape[1] == 240  # 270 - 30
    
    def test_crop_black_border_no_border(self):
        """测试没有黑边的情况"""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 128
        
        cropped = crop_black_border(img)
        
        assert cropped.shape == (200, 300, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
