import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def main():
    print("=" * 50)
    print("Panorama Stitching Pipeline")
    print("=" * 50)

    src_pts = np.load(os.path.join(IMG_DIR, "src_pts.npy"))
    dst_pts = np.load(os.path.join(IMG_DIR, "dst_pts.npy"))
    print(f"[1/6] Loaded {len(src_pts)} match point pairs")

    img1 = cv2.imread(os.path.join(IMG_DIR, "test1.jpg"))
    img2 = cv2.imread(os.path.join(IMG_DIR, "test2.jpg"))
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    print(f"[2/6] Image1: {w1}x{h1}, Image2: {w2}x{h2}")

    H_1to2, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inlier_count = int(np.sum(mask))
    print(f"[3/6] Homography inliers: {inlier_count}/{len(src_pts)}")

    H_2to1 = np.linalg.inv(H_1to2)
    print("[4/6] Computing canvas size...")
    canvas_w, canvas_h, offset_x, offset_y = compute_canvas_size(
        img1.shape, img2.shape, H_2to1
    )
    print(f"      Canvas: {canvas_w} x {canvas_h}, offset: ({offset_x}, {offset_y})")

    M_offset = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float64)

    H1_final = M_offset @ np.eye(3)
    H2_final = M_offset @ H_2to1

    warped1 = cv2.warpPerspective(img1, H1_final, (canvas_w, canvas_h))
    warped2 = cv2.warpPerspective(img2, H2_final, (canvas_w, canvas_h))
    print("[5/6] Warped both images to canvas")

    gray1 = cv2.cvtColor(warped1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(warped2, cv2.COLOR_BGR2GRAY)
    mask1 = gray1 > 3
    mask2 = gray2 > 3

    w1_map = make_blend_mask(mask1, mask2)

    w1_3ch = np.stack([w1_map, w1_map, w1_map], axis=-1)
    blended = (w1_3ch * warped1.astype(np.float32) +
               (1.0 - w1_3ch) * warped2.astype(np.float32))
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    final = auto_crop_black(blended)
    print(f"[6/6] Auto-cropped: {final.shape[1]} x {final.shape[0]}")

    out_path = os.path.join(OUT_DIR, "my_panorama.jpg")
    cv2.imwrite(out_path, final)
    print("=" * 50)
    print(f"Done! Panorama saved to: output/my_panorama.jpg")
    print("=" * 50)


if __name__ == "__main__":
    main()
