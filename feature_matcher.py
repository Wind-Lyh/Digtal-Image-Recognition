import cv2
import numpy as np

def load_and_preprocess(image_path):
    """读取图片并预处理"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"图片 {image_path} 没找到")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # 提升对比度
    return img, gray

class ORBFeatureExtractor:
    """ORB特征提取器"""
    def __init__(self, n_features=1500):
        self.orb = cv2.ORB_create(nfeatures=n_features)
    
    def detect_and_compute(self, gray_img):
        kp, des = self.orb.detectAndCompute(gray_img, None)
        return kp, des

def match_images_from_memory(img1: np.ndarray, img2: np.ndarray):
    """
    针对内存图像的匹配函数
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.equalizeHist(gray1)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.equalizeHist(gray2)
    
    # 2. 提取ORB特征
    extractor = ORBFeatureExtractor(n_features=1500)
    kp1, des1 = extractor.detect_and_compute(gray1)
    kp2, des2 = extractor.detect_and_compute(gray2)
    
    # 3. 特征匹配（BFMatcher + KNN + Lowe's ratio test）
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # 4. 筛选优质匹配点（ratio test < 0.75）
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    # 5. 提取坐标数组
    if len(good_matches) < 4:
        return None, None, None, None, None
        
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    
    return src_pts, dst_pts, kp1, kp2, good_matches

def match_two_images(img1_path, img2_path):
    """
    开发者A的核心匹配函数
    输入：两张图片路径
    输出：src_pts（图1特征点坐标）, dst_pts（图2特征点坐标）, 以及用于可视化的关键点和匹配结果
    """
    # 1. 读取预处理
    img1, gray1 = load_and_preprocess(img1_path)
    img2, gray2 = load_and_preprocess(img2_path)
    print(f"图1尺寸: {gray1.shape}, 图2尺寸: {gray2.shape}")
    
    src_pts, dst_pts, kp1, kp2, good_matches = match_images_from_memory(img1, img2)
    
    if good_matches is not None:
        print(f"图1特征点: {len(kp1)}, 图2特征点: {len(kp2)}")
        print(f"[OK] 优质匹配点数量: {len(good_matches)}")
    else:
        print("[WARN] 匹配点太少，无法计算单应矩阵！")
        
    return src_pts, dst_pts, kp1, kp2, good_matches

if __name__ == "__main__":
    # 匹配两张测试图
    src_pts, dst_pts, kp1, kp2, good_matches = match_two_images("images/test1.jpg", "images/test2.jpg")
    
    if good_matches is not None:
        # 1. 绘制匹配结果（可视化）
        img1, _ = load_and_preprocess("images/test1.jpg")
        img2, _ = load_and_preprocess("images/test2.jpg")
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # 保存匹配图（非图形界面环境）
        cv2.imwrite("images/match_result.jpg", match_img)
        print("[INFO] 匹配结果图已保存到 images/match_result.jpg")
        
        # 2. 【核心交付】保存坐标数据为 .npy 文件（给开发者B用）
        np.save("images/src_pts.npy", src_pts)
        np.save("images/dst_pts.npy", dst_pts)
        
        print("\n[INFO] 数据已打包！")
        print(f"src_pts 形状: {src_pts.shape} 已保存到 images/src_pts.npy")
        print(f"dst_pts 形状: {dst_pts.shape} 已保存到 images/dst_pts.npy")
        print("[OK] 请将这两个 .npy 文件发给开发者B！")
    else:
        print("匹配失败，请换一组重叠度更高的图片试试。")