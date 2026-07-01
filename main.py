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
    
    def _detect_direction(self, img1, img2, kp1, kp2, matches):
        """
        根据匹配点位置自动检测拼接方向
        
        Args:
            img1: 第一张图像
            img2: 第二张图像
            kp1: 第一张图像的关键点
            kp2: 第二张图像的关键点
            matches: 匹配结果
            
        Returns:
            检测到的拼接方向: 'left', 'right', 'top', 'bottom'
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        img1_x_coords = []
        img2_x_coords = []
        img1_y_coords = []
        img2_y_coords = []
        
        for m in matches[:min(50, len(matches))]:
            img1_x_coords.append(kp1[m.queryIdx].pt[0])
            img1_y_coords.append(kp1[m.queryIdx].pt[1])
            img2_x_coords.append(kp2[m.trainIdx].pt[0])
            img2_y_coords.append(kp2[m.trainIdx].pt[1])
        
        img1_x_mean = np.mean(img1_x_coords)
        img2_x_mean = np.mean(img2_x_coords)
        img1_y_mean = np.mean(img1_y_coords)
        img2_y_mean = np.mean(img2_y_coords)
        
        img1_x_ratio = img1_x_mean / w1
        img2_x_ratio = img2_x_mean / w2
        img1_y_ratio = img1_y_mean / h1
        img2_y_ratio = img2_y_mean / h2
        
        x_diff_ratio = abs(img1_x_ratio - img2_x_ratio)
        y_diff_ratio = abs(img1_y_ratio - img2_y_ratio)
        
        if x_diff_ratio > y_diff_ratio:
            if img1_x_ratio > img2_x_ratio:
                direction = 'right'
            else:
                direction = 'left'
        else:
            if img1_y_ratio > img2_y_ratio:
                direction = 'bottom'
            else:
                direction = 'top'
        
        print(f"自动检测拼接方向: {direction} (img1_x_ratio={img1_x_ratio:.2f}, img2_x_ratio={img2_x_ratio:.2f})")
        return direction
    
    def run_real(self, img_paths: List[str], blend_mode: str = 'linear', direction: str = 'auto') -> str:
        """
        使用真实图片进行拼接，从.npy文件加载预计算的匹配点
        
        Args:
            img_paths: 图像路径列表
            blend_mode: 融合模式，'linear' 或 'multiband'
            direction: 拼接方向，'auto', 'left', 'right', 'top', 'bottom'
            
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
        
        if len(imgs) == 2:
            img1 = imgs[0]
            img2 = imgs[1]
            
            orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)
            
            print(f"检测到特征点: 图1={len(kp1)}个, 图2={len(kp2)}个")
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)
            
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            good_matches = sorted(good_matches, key=lambda x: x.distance)
            good_matches = good_matches[:min(200, len(good_matches))]
            
            if len(good_matches) < MIN_MATCH_COUNT:
                print(f"匹配点数量不足: {len(good_matches)} < {MIN_MATCH_COUNT}")
                return ""
            
            if direction == 'auto':
                detected_direction = self._detect_direction(img1, img2, kp1, kp2, good_matches)
            else:
                detected_direction = direction
            
            print(f"拼接方向: {detected_direction}")
            
            if detected_direction in ['left', 'top']:
                src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                ref_img = img2
                warp_img = img1
                print(f"第二张图放在左边/上边")
            else:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                ref_img = img1
                warp_img = img2
                print(f"第一张图放在左边/上边")
            
            try:
                H, mask = HomographyEstimator.compute(src_pts.reshape(-1, 2), dst_pts.reshape(-1, 2))
                print(f"单应矩阵计算成功，内点数量: {np.sum(mask)}")
            except (InsufficientPointsError, HomographyEstimationError) as e:
                print(f"单应矩阵计算失败: {e}")
                return ""
            
            canvas_width, canvas_height, x_offset, y_offset = ImageWarper.get_canvas_size([ref_img, warp_img], [np.eye(3), H])
            
            warped_ref = ImageWarper.warp_image(ref_img, np.eye(3), canvas_width, canvas_height, x_offset, y_offset)
            warped_warp = ImageWarper.warp_image(warp_img, H, canvas_width, canvas_height, x_offset, y_offset)
            
            result = ImageBlender.blend(warped_ref, warped_warp, mode=blend_mode)
        else:
            result = imgs[0]
            
            for i in range(1, len(imgs)):
                print(f"正在拼接第 {i} 张图像...")
                
                img1 = result
                img2 = imgs[i]
                
                orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
                kp1, des1 = orb.detectAndCompute(img1, None)
                kp2, des2 = orb.detectAndCompute(img2, None)
                
                print(f"检测到特征点: 图1={len(kp1)}个, 图2={len(kp2)}个")
                
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = bf.knnMatch(des1, des2, k=2)
                
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                
                good_matches = sorted(good_matches, key=lambda x: x.distance)
                good_matches = good_matches[:min(200, len(good_matches))]
                
                if len(good_matches) < MIN_MATCH_COUNT:
                    print(f"匹配点数量不足: {len(good_matches)} < {MIN_MATCH_COUNT}")
                    continue
                
                if direction == 'auto':
                    detected_direction = self._detect_direction(img1, img2, kp1, kp2, good_matches)
                else:
                    detected_direction = direction
                
                if detected_direction in ['left', 'top']:
                    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    ref_img = img2
                    warp_img = img1
                else:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    ref_img = img1
                    warp_img = img2
                
                try:
                    H, mask = HomographyEstimator.compute(src_pts.reshape(-1, 2), dst_pts.reshape(-1, 2))
                    print(f"单应矩阵计算成功，内点数量: {np.sum(mask)}")
                except (InsufficientPointsError, HomographyEstimationError) as e:
                    print(f"单应矩阵计算失败: {e}")
                    continue
                
                canvas_width, canvas_height, x_offset, y_offset = ImageWarper.get_canvas_size([ref_img, warp_img], [np.eye(3), H])
                
                warped_ref = ImageWarper.warp_image(ref_img, np.eye(3), canvas_width, canvas_height, x_offset, y_offset)
                warped_warp = ImageWarper.warp_image(warp_img, H, canvas_width, canvas_height, x_offset, y_offset)
                
                result = ImageBlender.blend(warped_ref, warped_warp, mode=blend_mode)
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
    parser.add_argument('--direction', type=str, default='auto', 
                        choices=['auto', 'left', 'right', 'top', 'bottom'],
                        help='拼接方向: auto(自动检测), left(第二张在左), right(第二张在右), top(第二张在上), bottom(第二张在下)')
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
            output_path = pipeline.run_real(img_paths, blend_mode=args.blend, direction=args.direction)
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
