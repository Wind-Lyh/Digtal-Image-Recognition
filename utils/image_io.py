import os
from typing import List

import cv2
import numpy as np

from config import BLACK_BORDER_THRESHOLD, MAX_IMAGE_SIZE


def read_images_from_folder(folder_path: str) -> List[np.ndarray]:
    """
    读取文件夹内所有常见格式图片，按文件名排序返回列表
    
    Args:
        folder_path: 文件夹路径
        
    Returns:
        图像列表，每个元素为numpy数组（BGR格式）
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_paths = []
    
    if not os.path.exists(folder_path):
        return []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            image_paths.append(os.path.join(folder_path, filename))
    
    image_paths.sort()
    
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    
    return images


def resize_image(img: np.ndarray, max_size: int = MAX_IMAGE_SIZE) -> np.ndarray:
    """
    等比例缩放图像，使长边不超过max_size
    
    Args:
        img: 输入图像
        max_size: 长边最大尺寸
        
    Returns:
        缩放后的图像
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img.copy()
    
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def crop_black_border(img: np.ndarray) -> np.ndarray:
    """
    自动裁剪纯黑边（基于阈值检测）
    
    Args:
        img: 输入图像
        
    Returns:
        裁剪后的图像
    """
    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    mask = gray > BLACK_BORDER_THRESHOLD
    
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img.copy()
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return img[y_min:y_max+1, x_min:x_max+1]


def save_panorama(img: np.ndarray, output_path: str) -> bool:
    """
    保存全景图像
    
    Args:
        img: 全景图像
        output_path: 输出路径
        
    Returns:
        True表示保存成功，False表示失败
    """
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        _, ext = os.path.splitext(output_path)
        if ext.lower() in ('.jpg', '.jpeg'):
            cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(output_path, img)
        
        return True
    except Exception as e:
        print(f"保存图像失败: {e}")
        return False
