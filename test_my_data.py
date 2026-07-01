import os
import cv2
import numpy as np
from config_manager import get_config

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


def make_blend_mask(img1_mask, img2_mask, gamma_y=0.0):
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
            
        # 根据视差因子自适应调节融合带宽
        blend_width = 8 if gamma_y < 0.02 else 40
        mid = (l + r) // 2
        l_blend = max(l, mid - blend_width // 2)
        r_blend = min(r, mid + blend_width // 2)
        
        w1[y, l:l_blend] = 1.0
        w1[y, r_blend+1:r+1] = 0.0
        if r_blend >= l_blend:
            alpha = np.linspace(1.0, 0.0, r_blend - l_blend + 1)
            w1[y, l_blend:r_blend + 1] = alpha

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


def multi_band_blend(img1: np.ndarray, img2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray, levels: int = 3, gamma_y: float = 0.0) -> np.ndarray:
    """
    多频段金字塔融合，引入 gamma_y 动态调节底层 alpha 渐变宽度
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
                # 根据视差因子自适应调节底层线性混合的带宽
                blend_width = 8 if gamma_y < 0.02 else 40
                mid = (c_min + c_max) // 2
                l_blend = max(c_min, mid - blend_width // 2)
                r_blend = min(c_max, mid + blend_width // 2)
                
                if is_img1_left:
                    alpha[r, c_min:l_blend] = 1.0
                    alpha[r, r_blend+1:c_max+1] = 0.0
                    if r_blend >= l_blend:
                        alpha[r, l_blend:r_blend+1] = np.linspace(1.0, 0.0, r_blend - l_blend + 1)
                else:
                    alpha[r, c_min:l_blend] = 0.0
                    alpha[r, r_blend+1:c_max+1] = 1.0
                    if r_blend >= l_blend:
                        alpha[r, l_blend:r_blend+1] = np.linspace(0.0, 1.0, r_blend - l_blend + 1)
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


def run_stitching_pipeline(img1, img2, src_pts, dst_pts, blend_mode="linear", gamma_y=0.0):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    print(f"[A Engine] Image1: {w1}x{h1}, Image2: {w2}x{h2}, Mode: {blend_mode}, Gamma_Y: {gamma_y:.4f}")

    ransac_thresh = get_config("ransac_thresh", 5.0)
    # 使用 estimateAffinePartial2D 锁死透视自由度，防止扇形累积拉伸
    H_affine, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    if H_affine is None:
        raise ValueError("计算几何矩阵失败，可能是重叠区域不合理！")
        
    inlier_count = int(np.sum(mask))
    print(f"[A Engine] Affine inliers: {inlier_count}/{len(src_pts)}")
    if inlier_count < 10:
        raise ValueError(f"配准内点数量过少 ({inlier_count})，计算对齐矩阵可能存在错误！")

    # 升维回 3x3 矩阵，物理锁死第三行 [0, 0, 1]，彻底干掉透视畸变分量
    H_1to2 = np.vstack([H_affine, [0.0, 0.0, 1.0]])
    
    # 几何层锁死 Y 轴阶跃：根据视差因子判断是否为街景
    if gamma_y < 0.02:
        H_1to2[1, 2] = np.clip(H_1to2[1, 2], -2.0, 2.0)

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

    warped_mask1_orig = warped_mask1.copy()
    warped_mask2_orig = warped_mask2.copy()
    seam_mask = None

    print("[A Engine] 正在计算动态规划最佳缝合线(DpSeamFinder)以规避视差重影...")
    try:
        seam_finder = cv2.detail_DpSeamFinder("COLOR")
        # 转换至 float32 并计算最优接缝，这会原地修改传进去的 masks
        imgs_for_seam = [warped1.astype(np.float32), warped2.astype(np.float32)]
        corners = [(0, 0), (0, 0)]
        masks_for_seam = [warped_mask1.copy(), warped_mask2.copy()]
        seam_finder.find(imgs_for_seam, corners, masks_for_seam)
        warped_mask1 = masks_for_seam[0]
        warped_mask2 = masks_for_seam[1]
        
        # 提取接缝掩码用于后续 Inpainting
        kernel = np.ones((5, 5), np.uint8) # 适当放大形态学核确保接缝被完全覆盖
        seam_edge = cv2.morphologyEx(warped_mask1, cv2.MORPH_GRADIENT, kernel)
        overlap_area = cv2.bitwise_and(warped_mask1_orig, warped_mask2_orig)
        seam_mask = cv2.bitwise_and(seam_edge, overlap_area)
    except Exception as e:
        print(f"[A Engine] 计算接缝失败，回退至基础交叉融合: {e}")

    print(f"[A Engine] Blending images using {blend_mode}...")
    if blend_mode == "multiband":
        blended = multi_band_blend(warped1, warped2, warped_mask1, warped_mask2, levels=3, gamma_y=gamma_y)
    else:
        w1_map = make_blend_mask(warped_mask1 > 0, warped_mask2 > 0, gamma_y=gamma_y)
        w1_3ch = np.stack([w1_map, w1_map, w1_map], axis=-1)
        blended = (w1_3ch * warped1.astype(np.float32) +
                   (1.0 - w1_3ch) * warped2.astype(np.float32))
        blended = np.clip(blended, 0, 255).astype(np.uint8)

    if seam_mask is not None and np.count_nonzero(seam_mask) > 0:
        if gamma_y < 0.02:
            print("[A Engine] 街景模式：执行无痕图像修复，严格收窄修复掩码宽度...")
            kernel_erode = np.ones((2, 2), np.uint8)
            seam_thin = cv2.erode(seam_mask, kernel_erode, iterations=1)
            if np.count_nonzero(seam_thin) < 10:
                seam_thin = seam_mask
            blended = cv2.inpaint(blended, seam_thin, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
        else:
            print("[A Engine] 桌面/近景模式：适当释放缝合线修复宽度，平滑断层...")
            kernel_dilate = np.ones((3, 3), np.uint8)
            seam_thick = cv2.dilate(seam_mask, kernel_dilate, iterations=1)
            blended = cv2.inpaint(blended, seam_thick, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    final = auto_crop_black(blended)
    print(f"[A Engine] Auto-cropped final size: {final.shape[1]} x {final.shape[0]}")
    
    return final


def run_global_stitch_pipeline(imgs, match_func, blend_mode="linear", gamma_y=0.0, log_func=None):
    """
    全局两遍扫描拼接管线：
    Pass 1: 用原始图像计算所有相邻对的仿射矩阵，累乘获得全局变换
    Pass 2: 根据全局边界开辟超大画布，一次性将所有图片 warp 上去并融合
    """
    n = len(imgs)
    if n < 2:
        raise ValueError("有效图像不足2张")

    def _log(msg):
        if log_func:
            log_func(msg)
        print(msg)

    # ======== Pass 1: 计算所有相邻对的局部仿射矩阵 ========
    ransac_thresh = get_config("ransac_thresh", 5.0)
    
    # H_local[i] 表示 imgs[i] -> imgs[i-1] 的局部仿射变换 (3x3)
    H_locals = [None]  # imgs[0] 是锚点，无变换
    
    for i in range(1, n):
        _log(f"\n[Pass 1] 正在计算第 {i} <-> {i+1} 张图的特征匹配与仿射矩阵...")
        src_pts, dst_pts, _, _, good_matches = match_func(imgs[i-1], imgs[i])
        if good_matches is None:
            raise ValueError(f"第 {i+1} 张图与第 {i} 张图特征点匹配过少，拼接终止！")
        
        # src_pts 来自 imgs[i-1], dst_pts 来自 imgs[i]
        # H_affine: imgs[i-1] -> imgs[i], 我们要 imgs[i] -> imgs[i-1]
        H_affine, mask = cv2.estimateAffinePartial2D(
            dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh
        )
        if H_affine is None:
            raise ValueError(f"第 {i+1} 张图计算几何矩阵失败")
        
        inlier_count = int(np.sum(mask))
        _log(f"[Pass 1] Pair {i}-{i+1}: Affine inliers = {inlier_count}/{len(dst_pts)}")
        if inlier_count < 10:
            raise ValueError(f"第 {i+1} 张图配准内点过少 ({inlier_count})")
        
        H_3x3 = np.vstack([H_affine, [0.0, 0.0, 1.0]])
        
        # 街景模式锁死 dy
        if gamma_y < 0.02:
            H_3x3[1, 2] = np.clip(H_3x3[1, 2], -2.0, 2.0)
        
        H_locals.append(H_3x3)
    
    # ======== 累乘获得全局变换: H_global[i] = imgs[i] -> 锚点坐标系 ========
    _log("\n[Pass 1] 正在累乘计算全局变换矩阵...")
    H_globals = [np.eye(3, dtype=np.float64)]  # imgs[0] -> anchor = Identity
    for i in range(1, n):
        # H_global[i] = H_local[i] @ H_global[i-1]
        # H_local[i] 把 imgs[i] 映射到 imgs[i-1] 的坐标系
        # H_global[i-1] 把 imgs[i-1] 映射到锚点坐标系
        H_g = H_globals[i-1] @ H_locals[i]
        H_globals.append(H_g)
    
    # ======== 计算全局画布边界 ========
    _log("\n[Pass 2] 正在计算全局画布边界...")
    all_corners = []
    for i in range(n):
        h, w = imgs[i].shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H_globals[i])
        all_corners.append(warped_corners)
    
    all_corners_cat = np.concatenate(all_corners, axis=0)
    x_min = int(np.floor(np.min(all_corners_cat[:, 0, 0])))
    x_max = int(np.ceil(np.max(all_corners_cat[:, 0, 0])))
    y_min = int(np.floor(np.min(all_corners_cat[:, 0, 1])))
    y_max = int(np.ceil(np.max(all_corners_cat[:, 0, 1])))
    
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    _log(f"[Pass 2] 全局画布尺寸: {canvas_w} x {canvas_h}, 偏移: ({-x_min}, {-y_min})")
    
    # 全局平移矩阵
    M_shift = np.array([[1, 0, -x_min],
                         [0, 1, -y_min],
                         [0, 0, 1]], dtype=np.float64)
    
    # ======== Pass 2: 一次性渲染所有图片到全局画布 ========
    _log(f"\n[Pass 2] 正在将 {n} 张图片渲染到全局画布上...")
    
    warped_imgs = []
    warped_masks = []
    for i in range(n):
        H_final = M_shift @ H_globals[i]
        w_img = cv2.warpPerspective(imgs[i], H_final, (canvas_w, canvas_h),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        src_mask = np.full(imgs[i].shape[:2], 255, dtype=np.uint8)
        w_mask = cv2.warpPerspective(src_mask, H_final, (canvas_w, canvas_h),
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_imgs.append(w_img)
        warped_masks.append(w_mask)
        _log(f"  ✔ 第 {i+1}/{n} 张已渲染")
    
    # ======== 级联融合: 从左到右依次融合 ========
    _log(f"\n[Pass 2] 正在级联融合 {n} 张图片...")
    canvas = warped_imgs[0].copy()
    canvas_mask = warped_masks[0].copy()
    
    for i in range(1, n):
        _log(f"  融合第 {i+1}/{n} 张...")
        next_img = warped_imgs[i]
        next_mask = warped_masks[i]
        
        # DpSeamFinder 寻找最优接缝
        seam_mask = None
        canvas_mask_orig = canvas_mask.copy()
        next_mask_orig = next_mask.copy()
        try:
            seam_finder = cv2.detail_DpSeamFinder("COLOR")
            imgs_for_seam = [canvas.astype(np.float32), next_img.astype(np.float32)]
            corners_seam = [(0, 0), (0, 0)]
            masks_for_seam = [canvas_mask.copy(), next_mask.copy()]
            seam_finder.find(imgs_for_seam, corners_seam, masks_for_seam)
            
            blend_m1 = masks_for_seam[0]
            blend_m2 = masks_for_seam[1]
            
            kernel = np.ones((5, 5), np.uint8)
            seam_edge = cv2.morphologyEx(blend_m1, cv2.MORPH_GRADIENT, kernel)
            overlap_area = cv2.bitwise_and(canvas_mask_orig, next_mask_orig)
            seam_mask = cv2.bitwise_and(seam_edge, overlap_area)
        except Exception as e:
            blend_m1 = canvas_mask
            blend_m2 = next_mask
            _log(f"  [WARN] DpSeamFinder 失败，回退基础融合: {e}")
        
        # 融合
        if blend_mode == "multiband":
            blended = multi_band_blend(canvas, next_img, blend_m1, blend_m2, levels=3, gamma_y=gamma_y)
        else:
            w1_map = make_blend_mask(blend_m1 > 0, blend_m2 > 0, gamma_y=gamma_y)
            w1_3ch = np.stack([w1_map, w1_map, w1_map], axis=-1)
            blended = (w1_3ch * canvas.astype(np.float32) +
                       (1.0 - w1_3ch) * next_img.astype(np.float32))
            blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        # Inpaint 接缝
        if seam_mask is not None and np.count_nonzero(seam_mask) > 0:
            if gamma_y < 0.02:
                kernel_erode = np.ones((2, 2), np.uint8)
                seam_thin = cv2.erode(seam_mask, kernel_erode, iterations=1)
                if np.count_nonzero(seam_thin) < 10:
                    seam_thin = seam_mask
                blended = cv2.inpaint(blended, seam_thin, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
            else:
                kernel_dilate = np.ones((3, 3), np.uint8)
                seam_thick = cv2.dilate(seam_mask, kernel_dilate, iterations=1)
                blended = cv2.inpaint(blended, seam_thick, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        
        canvas = blended
        canvas_mask = cv2.bitwise_or(canvas_mask_orig, next_mask_orig)
    
    final = auto_crop_black(canvas)
    _log(f"\n[Pass 2] 全局拼接完成！最终尺寸: {final.shape[1]} x {final.shape[0]}")
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
    img_height = img1.shape[0]
    gamma_y = abs(dy) / img_height
    
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
    
    return direction_msg, force_multiband, exposure_diff, gamma_y
