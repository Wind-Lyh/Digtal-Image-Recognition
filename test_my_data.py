import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
OUT_DIR = os.path.join(BASE_DIR, "output")


def compute_canvas_size(img1_shape, img2_shape, H_2to1):
    h2, w2 = img2_shape[:2]
    corners_img2 = np.array([
        [0, 0],
        [w2 - 1, 0],
        [w2 - 1, h2 - 1],
        [0, h2 - 1]
    ], dtype=np.float32).reshape(-1, 1, 2)

    corners_img2_warped = cv2.perspectiveTransform(corners_img2, H_2to1)

    h1, w1 = img1_shape[:2]
    all_x = np.concatenate([[0, w1 - 1], corners_img2_warped[:, 0, 0]])
    all_y = np.concatenate([[0, h1 - 1], corners_img2_warped[:, 0, 1]])

    x_min = int(np.floor(np.min(all_x)))
    x_max = int(np.ceil(np.max(all_x)))
    y_min = int(np.floor(np.min(all_y)))
    y_max = int(np.ceil(np.max(all_y)))

    canvas_w = x_max - x_min + 1
    canvas_h = y_max - y_min + 1
    offset_x = -x_min
    offset_y = -y_min

    return canvas_w, canvas_h, offset_x, offset_y


def make_blend_mask(img1_mask, img2_mask):
    h, w = img1_mask.shape
    overlap = img1_mask & img2_mask

    w1 = np.zeros((h, w), dtype=np.float32)
    w1[img1_mask] = 1.0

    if not np.any(overlap):
        return w1

    cols_overlap = np.where(np.any(overlap, axis=0))[0]
    overlap_left = cols_overlap[0]
    overlap_right = cols_overlap[-1]

    for y in range(h):
        row_overlap = overlap[y, :]
        if not np.any(row_overlap):
            continue
        cols = np.where(row_overlap)[0]
        l = cols[0]
        r = cols[-1]
        if r <= l:
            continue
        alpha = np.linspace(1.0, 0.0, r - l + 1)
        w1[y, l:r + 1] = alpha

    w1[~img1_mask & img2_mask] = 0.0

    return w1


def auto_crop_black(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_mask = cv2.threshold(gray, 3, 255, cv2.THRESH_BINARY)

    rows = np.any(bin_mask, axis=1)
    cols = np.any(bin_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return img

    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]

    margin = 3
    top = max(0, top - margin)
    bottom = min(img.shape[0] - 1, bottom + margin)
    left = max(0, left - margin)
    right = min(img.shape[1] - 1, right + margin)

    return img[top:bottom + 1, left:right + 1]


def multi_band_blend(img1: np.ndarray, img2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray, levels: int = 5) -> np.ndarray:
    """
    多频段金字塔融合（基于拉普拉斯金字塔），从 core 移植的优质版本
    """
    mask1_bool = mask1 > 0
    mask2_bool = mask2 > 0
    overlap_mask = mask1_bool & mask2_bool
        
    rows, cols = overlap_mask.shape
    alpha = np.zeros((rows, cols), dtype=np.float32)
    
    x1_center = np.mean(np.where(mask1_bool)[1]) if np.any(mask1_bool) else 0
    x2_center = np.mean(np.where(mask2_bool)[1]) if np.any(mask2_bool) else 0
    is_img1_left = x1_center < x2_center
    
    for r in range(rows):
        row_cols = np.where(overlap_mask[r])[0]
        if len(row_cols) > 0:
            c_min, c_max = row_cols.min(), row_cols.max()
            if c_max > c_min:
                if is_img1_left:
                    alpha[r, c_min:c_max+1] = np.linspace(1.0, 0.0, c_max - c_min + 1)
                else:
                    alpha[r, c_min:c_max+1] = np.linspace(0.0, 1.0, c_max - c_min + 1)
            else:
                alpha[r, c_min] = 0.5
                
    alpha[mask1_bool & ~mask2_bool] = 1.0
    alpha[mask2_bool & ~mask1_bool] = 0.0
    
    alpha = np.stack([alpha, alpha, alpha], axis=2)
    
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)
    
    gaussian_pyr1 = [img1_float]
    gaussian_pyr2 = [img2_float]
    gaussian_pyr_alpha = [alpha]
    
    for _ in range(levels):
        img1_float = cv2.pyrDown(img1_float)
        img2_float = cv2.pyrDown(img2_float)
        alpha = cv2.pyrDown(alpha)
        
        gaussian_pyr1.append(img1_float)
        gaussian_pyr2.append(img2_float)
        gaussian_pyr_alpha.append(alpha)
    
    laplacian_pyr1 = []
    laplacian_pyr2 = []
    
    for i in range(levels):
        next_level = cv2.pyrUp(gaussian_pyr1[i + 1])
        if next_level.shape != gaussian_pyr1[i].shape:
            next_level = cv2.resize(next_level, (gaussian_pyr1[i].shape[1], gaussian_pyr1[i].shape[0]))
        laplacian_pyr1.append(gaussian_pyr1[i] - next_level)
        
        next_level = cv2.pyrUp(gaussian_pyr2[i + 1])
        if next_level.shape != gaussian_pyr2[i].shape:
            next_level = cv2.resize(next_level, (gaussian_pyr2[i].shape[1], gaussian_pyr2[i].shape[0]))
        laplacian_pyr2.append(gaussian_pyr2[i] - next_level)
    
    laplacian_pyr1.append(gaussian_pyr1[-1])
    laplacian_pyr2.append(gaussian_pyr2[-1])
    
    blended_pyr = []
    for i in range(levels + 1):
        alpha_level = gaussian_pyr_alpha[i]
        blended = alpha_level * laplacian_pyr1[i] + (1 - alpha_level) * laplacian_pyr2[i]
        blended_pyr.append(blended)
    
    result = blended_pyr[-1]
    for i in range(levels - 1, -1, -1):
        result = cv2.pyrUp(result)
        if result.shape != blended_pyr[i].shape:
            result = cv2.resize(result, (blended_pyr[i].shape[1], blended_pyr[i].shape[0]))
        result = result + blended_pyr[i]
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def run_stitching_pipeline(img1, img2, src_pts, dst_pts, blend_mode="linear"):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    print(f"[A Engine] Image1: {w1}x{h1}, Image2: {w2}x{h2}, Mode: {blend_mode}")

    H_1to2, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    inlier_count = int(np.sum(mask))
    print(f"[A Engine] Homography inliers: {inlier_count}/{len(src_pts)}")
    if inlier_count < 10:
        raise ValueError(f"配准内点数量过少 ({inlier_count})，计算单应矩阵可能存在错误！")

    H_2to1 = np.linalg.inv(H_1to2)
    print("[A Engine] Computing canvas size...")
    canvas_w, canvas_h, offset_x, offset_y = compute_canvas_size(
        img1.shape, img2.shape, H_2to1
    )
    print(f"           Canvas: {canvas_w} x {canvas_h}, offset: ({offset_x}, {offset_y})")

    M_offset = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float64)

    H1_final = M_offset @ np.eye(3)
    H2_final = M_offset @ H_2to1

    print("[A Engine] Warping images and explicit masks to canvas...")
    warped1 = cv2.warpPerspective(img1, H1_final, (canvas_w, canvas_h), borderMode=cv2.BORDER_TRANSPARENT)
    warped2 = cv2.warpPerspective(img2, H2_final, (canvas_w, canvas_h), borderMode=cv2.BORDER_TRANSPARENT)
    
    src_mask1 = np.full(img1.shape[:2], 255, dtype=np.uint8)
    src_mask2 = np.full(img2.shape[:2], 255, dtype=np.uint8)
    warped_mask1 = cv2.warpPerspective(src_mask1, H1_final, (canvas_w, canvas_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_mask2 = cv2.warpPerspective(src_mask2, H2_final, (canvas_w, canvas_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    print(f"[A Engine] Blending images using {blend_mode}...")
    if blend_mode == "multiband":
        blended = multi_band_blend(warped1, warped2, warped_mask1, warped_mask2, levels=5)
    else:
        w1_map = make_blend_mask(warped_mask1 > 0, warped_mask2 > 0)
        w1_3ch = np.stack([w1_map, w1_map, w1_map], axis=-1)
        blended = (w1_3ch * warped1.astype(np.float32) +
                   (1.0 - w1_3ch) * warped2.astype(np.float32))
        blended = np.clip(blended, 0, 255).astype(np.uint8)

    final = auto_crop_black(blended)
    print(f"[A Engine] Auto-cropped final size: {final.shape[1]} x {final.shape[0]}")
    
    return final

def smart_topology_probe(img1: np.ndarray, img2: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, exposure_thresh: int = 25):
    """
    智能拓扑探针：
    1. 计算平均平移量来判断方向
    2. 计算重叠区域灰度差来判断是否强制使用多频段融合
    """
    if len(src_pts) == 0 or len(dst_pts) == 0:
        return "未知方向", False, 0.0

    # 1. 计算平均几何平移向量
    dx = np.mean(src_pts[:, 0] - dst_pts[:, 0])
    dy = np.mean(src_pts[:, 1] - dst_pts[:, 1])
    
    if abs(dx) > abs(dy):
        if dx > 0:
            direction = "右侧拼接"
            matrix_desc = "横向"
        else:
            direction = "左侧拼接"
            matrix_desc = "横向"
    else:
        if dy > 0:
            direction = "下方拼接"
            matrix_desc = "纵向"
        else:
            direction = "上方拼接"
            matrix_desc = "纵向"
            
    direction_msg = f"图 B 位于图 A 的【{direction}】，转换为{matrix_desc}矩阵。"
    
    # 2. 曝光差异度计算 (提取匹配点局部灰度)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    h1, w1 = gray1.shape
    h2, w2 = gray2.shape
    
    x1_valid = np.clip(src_pts[:, 0].astype(int), 0, w1 - 1)
    y1_valid = np.clip(src_pts[:, 1].astype(int), 0, h1 - 1)
    x2_valid = np.clip(dst_pts[:, 0].astype(int), 0, w2 - 1)
    y2_valid = np.clip(dst_pts[:, 1].astype(int), 0, h2 - 1)
    
    mean1 = np.mean(gray1[y1_valid, x1_valid])
    mean2 = np.mean(gray2[y2_valid, x2_valid])
    
    exposure_diff = abs(mean1 - mean2)
    force_multiband = exposure_diff > exposure_thresh
    
    return direction_msg, force_multiband, exposure_diff
