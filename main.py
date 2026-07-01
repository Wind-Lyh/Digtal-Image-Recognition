import argparse
import os
from typing import Dict, List

import cv2
import numpy as np

import config
from core.homography_estimator import HomographyEstimator, HomographyEstimationError, InsufficientPointsError
from core.image_warper import ImageWarper
from core.image_blender import ImageBlender
from utils.image_io import read_images_from_folder, crop_black_border, save_panorama


class StitchPipeline:
    """全景图像拼接管道"""
    
    def __init__(self, config: Dict = None):
        """
        初始化拼接管道
        
        Args:
            config: 配置字典，若为None则使用默认配置
        """
        self.config = config if config is not None else {}
    
    def run_simulation(self, img_paths=None) -> str:
        """
        用于自测的模拟流程，不依赖真实匹配数据
        
        Args:
            img_paths: 图像路径列表，若为None则生成模拟图像
            
        Returns:
            输出图像的保存路径
        """
        if img_paths is None or len(img_paths) < 2:
            print("未提供图像路径，生成模拟测试图像...")
            img1 = np.zeros((400, 600, 3), dtype=np.uint8)
            img1[:] = [255, 128, 64]
            cv2.rectangle(img1, (50, 50), (550, 350), (255, 255, 255), 5)
            
            img2 = np.zeros((400, 600, 3), dtype=np.uint8)
            img2[:] = [64, 128, 255]
            cv2.circle(img2, (300, 200), 150, (255, 255, 255), 5)
            
            imgs = [img1, img2]
        else:
            imgs = []
            for path in img_paths[:2]:
                img = cv2.imread(path)
                if img is not None:
                    imgs.append(img)
            
            if len(imgs) < 2:
                print("读取图像不足，生成模拟测试图像...")
                img1 = np.zeros((400, 600, 3), dtype=np.uint8)
                img1[:] = [255, 128, 64]
                cv2.rectangle(img1, (50, 50), (550, 350), (255, 255, 255), 5)
                
                img2 = np.zeros((400, 600, 3), dtype=np.uint8)
                img2[:] = [64, 128, 255]
                cv2.circle(img2, (300, 200), 150, (255, 255, 255), 5)
                
                imgs = [img1, img2]
        
        h1, w1 = imgs[0].shape[:2]
        h2, w2 = imgs[1].shape[:2]
        
        src_pts = np.array([
            [50, 50],
            [550, 50],
            [550, 350],
            [50, 350]
        ], dtype=np.float64)
        
        dst_pts = np.array([
            [100 + np.random.uniform(-10, 10), 50 + np.random.uniform(-10, 10)],
            [w2 - 50 + np.random.uniform(-10, 10), 50 + np.random.uniform(-10, 10)],
            [w2 - 50 + np.random.uniform(-10, 10), h2 - 50 + np.random.uniform(-10, 10)],
            [100 + np.random.uniform(-10, 10), h2 - 50 + np.random.uniform(-10, 10)]
        ], dtype=np.float64)
        
        try:
            H, mask = HomographyEstimator.compute(src_pts, dst_pts)
            print(f"单应矩阵计算成功，内点数量: {np.sum(mask)}")
        except (InsufficientPointsError, HomographyEstimationError) as e:
            print(f"单应矩阵计算失败: {e}")
            return ""
        
        Hs = [np.eye(3), H]
        
        canvas_width, canvas_height, x_offset, y_offset = ImageWarper.get_canvas_size(imgs, Hs)
        print(f"变换后画布尺寸: {canvas_width} x {canvas_height}")
        print(f"偏移量: ({x_offset}, {y_offset})")
        
        warped_imgs = []
        for i, (img, H) in enumerate(zip(imgs, Hs)):
            warped = ImageWarper.warp_image(img, H, canvas_width, canvas_height, x_offset, y_offset)
            warped_imgs.append(warped)
        
        blended = ImageBlender.blend(warped_imgs[0], warped_imgs[1], mode='linear')
        
        cropped = crop_black_border(blended)
        
        output_path = "output/simulated_panorama.jpg"
        save_panorama(cropped, output_path)
        
        return output_path
    
    def run_real(self, img_paths: List[str], blend_mode: str = 'linear') -> str:
        """
        使用真实图片进行拼接，从.npy文件加载预计算的匹配点
        
        Args:
            img_paths: 图像路径列表
            blend_mode: 融合模式，'linear' 或 'multiband'
            
        Returns:
            输出图像的保存路径
        """
        if len(img_paths) < 2:
            print("需要至少两张图片进行拼接")
            return ""
        
        imgs = []
        for path in img_paths:
            img = cv2.imread(path)
            if img is not None:
                imgs.append(img)
                print(f"读取图像: {path}, 尺寸: {img.shape}")
            else:
                print(f"无法读取图像: {path}")
        
        if len(imgs) < 2:
            print("有效图像不足")
            return ""
        
        img1 = imgs[0]
        img2 = imgs[1]
        
        print("从预计算文件加载匹配点...")
        src_pts = np.load("images/src_pts.npy")
        dst_pts = np.load("images/dst_pts.npy")
        print(f"加载匹配点数量: {src_pts.shape[0]}")
        
        try:
            H, mask = HomographyEstimator.compute(src_pts, dst_pts)
            print(f"单应矩阵计算成功，内点数量: {np.sum(mask)}")
        except (InsufficientPointsError, HomographyEstimationError) as e:
            print(f"单应矩阵计算失败: {e}")
            return ""
        
        canvas_width, canvas_height, x_offset, y_offset = ImageWarper.get_canvas_size([img1, img2], [np.eye(3), H])
        
        warped_img1 = ImageWarper.warp_image(img1, np.eye(3), canvas_width, canvas_height, x_offset, y_offset)
        warped_img2 = ImageWarper.warp_image(img2, H, canvas_width, canvas_height, x_offset, y_offset)
        
        result = ImageBlender.blend(warped_img1, warped_img2, mode=blend_mode)
        
        cropped = crop_black_border(result)
        
        output_path = "output/real_panorama.jpg"
        save_panorama(cropped, output_path)
        
        return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='全景图像拼接系统')
    parser.add_argument('--folder', type=str, help='图片文件夹路径')
    parser.add_argument('--blend', type=str, default='linear', choices=['linear', 'multiband'], 
                        help='融合方式: linear (线性融合) 或 multiband (多频段融合)')
    parser.add_argument('--output', type=str, default='output/panorama.jpg', help='输出路径')
    parser.add_argument('--simulate', action='store_true', help='运行模拟测试')
    
    args = parser.parse_args()
    
    pipeline = StitchPipeline()
    
    if args.simulate:
        print("运行模拟测试...")
        output_path = pipeline.run_simulation()
        if output_path:
            print(f"模拟测试完成，全景图像已保存到: {output_path}")
        else:
            print("模拟测试失败")
    elif args.folder:
        print(f"读取图片文件夹: {args.folder}")
        img_paths = [os.path.join(args.folder, f) for f in sorted(os.listdir(args.folder)) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not img_paths:
            print("文件夹中未找到图片")
        else:
            print(f"找到 {len(img_paths)} 张图片")
            output_path = pipeline.run_real(img_paths, blend_mode=args.blend)
            if output_path:
                print(f"拼接完成，全景图像已保存到: {output_path}")
            else:
                print("拼接失败")
    else:
        print("运行默认模拟测试...")
        output_path = pipeline.run_simulation()
        if output_path:
            print(f"全景图像已保存到: {output_path}")
        else:
            print("全景图像生成失败")
