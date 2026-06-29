import os
import cv2
import numpy as np


def create_checkerboard(width, height, square_size=50):
    """
    创建棋盘格图案
    
    Args:
        width: 图像宽度
        height: 图像高度
        square_size: 方块大小
        
    Returns:
        棋盘格图像
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            if (y // square_size + x // square_size) % 2 == 0:
                img[y:y+square_size, x:x+square_size] = [255, 255, 255]
            else:
                img[y:y+square_size, x:x+square_size] = [128, 128, 128]
    
    return img


def generate_test_images(output_dir="test_images"):
    """
    生成测试图片
    
    Args:
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    width, height = 800, 600
    
    # 组1：水平平移（两张图重叠50%）
    print("生成水平平移测试图片...")
    img1_group1 = create_checkerboard(width, height)
    img2_group1 = create_checkerboard(width, height)
    
    # 添加不同颜色的标记以便区分
    cv2.putText(img1_group1, "Image 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.putText(img2_group1, "Image 2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    cv2.imwrite(os.path.join(output_dir, "group1_1.jpg"), img1_group1)
    cv2.imwrite(os.path.join(output_dir, "group1_2.jpg"), img2_group1)
    
    # 组2：带15°旋转
    print("生成旋转测试图片...")
    img1_group2 = create_checkerboard(width, height)
    img2_group2 = create_checkerboard(width, height)
    
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
    img2_group2 = cv2.warpAffine(img2_group2, rotation_matrix, (width, height))
    
    cv2.putText(img1_group2, "Image 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.putText(img2_group2, "Image 2 (rot)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    cv2.imwrite(os.path.join(output_dir, "group2_1.jpg"), img1_group2)
    cv2.imwrite(os.path.join(output_dir, "group2_2.jpg"), img2_group2)
    
    # 组3：缩放比例不一致
    print("生成缩放测试图片...")
    img1_group3 = create_checkerboard(width, height)
    img2_scaled = create_checkerboard(int(width * 1.2), int(height * 1.2))
    img2_group3 = cv2.resize(img2_scaled, (width, height))
    
    cv2.putText(img1_group3, "Image 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.putText(img2_group3, "Image 2 (scaled)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    cv2.imwrite(os.path.join(output_dir, "group3_1.jpg"), img1_group3)
    cv2.imwrite(os.path.join(output_dir, "group3_2.jpg"), img2_group3)
    
    print(f"测试图片已生成到 {output_dir}/")


if __name__ == "__main__":
    generate_test_images()
