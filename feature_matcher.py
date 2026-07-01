import cv2
import numpy as np


def load_and_preprocess(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"image not found: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return img, gray


def grid_detect_and_compute(gray, n_total_features=3000, grid_rows=4, grid_cols=4):
    h, w = gray.shape
    cell_h = h // grid_rows
    cell_w = w // grid_cols
    features_per_cell = n_total_features // (grid_rows * grid_cols)

    all_kp = []
    all_des = []
    orb = cv2.ORB_create(nfeatures=features_per_cell, scaleFactor=1.2, nlevels=8, edgeThreshold=15)

    for r in range(grid_rows):
        for c in range(grid_cols):
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = x1 + cell_w if c < grid_cols - 1 else w
            y2 = y1 + cell_h if r < grid_rows - 1 else h

            cell = gray[y1:y2, x1:x2]
            kp_cell, des_cell = orb.detectAndCompute(cell, None)

            if kp_cell and des_cell is not None:
                for kp in kp_cell:
                    kp.pt = (kp.pt[0] + x1, kp.pt[1] + y1)
                all_kp.extend(kp_cell)
                all_des.append(des_cell)

    if all_des:
        all_des = np.vstack(all_des)
    else:
        all_des = None

    return all_kp, all_des


def match_two_images(img1_path, img2_path):
    img1, gray1 = load_and_preprocess(img1_path)
    img2, gray2 = load_and_preprocess(img2_path)

    print(f"image1 size: {gray1.shape}, image2 size: {gray2.shape}")

    kp1, des1 = grid_detect_and_compute(gray1, n_total_features=3000, grid_rows=4, grid_cols=4)
    kp2, des2 = grid_detect_and_compute(gray2, n_total_features=3000, grid_rows=4, grid_cols=4)

    print(f"image1 keypoints: {len(kp1)}, image2 keypoints: {len(kp2)}")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"[OK] good matches: {len(good_matches)}")

    if len(good_matches) < 4:
        print("[WARN] not enough matches")
        return None, None, None, None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    return src_pts, dst_pts, kp1, kp2, good_matches


if __name__ == "__main__":
    src_pts, dst_pts, kp1, kp2, good_matches = match_two_images("images/test1.jpg", "images/test2.jpg")

    if good_matches is not None:
        img1, _ = load_and_preprocess("images/test1.jpg")
        img2, _ = load_and_preprocess("images/test2.jpg")
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imwrite("images/match_result.jpg", match_img)
        print("[INFO] match visualization saved to images/match_result.jpg")

        np.save("images/src_pts.npy", src_pts)
        np.save("images/dst_pts.npy", dst_pts)

        print(f"[OK] match points count: {len(good_matches)}")
        print(f"[OK] src_pts saved to images/src_pts.npy, shape: {src_pts.shape}")
        print(f"[OK] dst_pts saved to images/dst_pts.npy, shape: {dst_pts.shape}")
    else:
        print("match failed")
