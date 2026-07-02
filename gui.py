import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np

from test_my_data import compute_canvas_size, make_blend_mask, auto_crop_black


def stitch_images(img1, img2, src_pts, dst_pts):
    src_pts = np.float32(src_pts).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
    H_1to2, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    H_2to1 = np.linalg.inv(H_1to2)

    canvas_w, canvas_h, offset_x, offset_y = compute_canvas_size(
        img1.shape, img2.shape, H_2to1
    )

    M_offset = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float64)

    H1_final = M_offset @ np.eye(3)
    H2_final = M_offset @ H_2to1

    warped1 = cv2.warpPerspective(img1, H1_final, (canvas_w, canvas_h))
    warped2 = cv2.warpPerspective(img2, H2_final, (canvas_w, canvas_h))

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

    return final


class StitchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("全景图像拼接系统")
        self.root.geometry("900x650")
        self.root.resizable(True, True)

        self.img_paths = []
        self.blend_mode = tk.StringVar(value="linear")

        self.style = ttk.Style()
        self.style.configure('Title.TLabel', font=('Helvetica', 18, 'bold'), foreground='#2c3e50')
        self.style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'), foreground='#34495e')
        self.style.configure('Status.TLabel', font=('Helvetica', 10), foreground='#7f8c8d')
        self.style.configure('Accent.TButton', font=('Helvetica', 10, 'bold'))
        self.style.map('Accent.TButton',
                       foreground=[('pressed', 'white'), ('active', '#2980b9')],
                       background=[('pressed', '#2c3e50'), ('active', '#ecf0f1')])

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        ttk.Label(main_frame, text="全景图像拼接系统", style='Title.TLabel').grid(
            row=0, column=0, columnspan=3, pady=(0, 20)
        )

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)

        ttk.Button(button_frame, text="选择图片", command=self.select_images, style='Accent.TButton').grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W
        )
        ttk.Button(button_frame, text="选择文件夹", command=self.select_folder, style='Accent.TButton').grid(
            row=0, column=1, padx=5, pady=5, sticky=tk.W
        )
        ttk.Button(button_frame, text="清空列表", command=self.clear_images, style='Accent.TButton').grid(
            row=0, column=2, padx=5, pady=5, sticky=tk.W
        )

        ttk.Label(main_frame, text="融合方式:", style='Header.TLabel').grid(
            row=2, column=0, padx=5, pady=(10, 5), sticky=tk.W
        )
        ttk.Radiobutton(main_frame, text="线性融合", variable=self.blend_mode, value="linear").grid(
            row=2, column=1, padx=5, pady=(10, 5), sticky=tk.W
        )
        ttk.Radiobutton(main_frame, text="多频段融合", variable=self.blend_mode, value="multiband").grid(
            row=2, column=2, padx=5, pady=(10, 5), sticky=tk.W
        )

        ttk.Button(main_frame, text="开始拼接", command=self.start_stitching, style='Accent.TButton').grid(
            row=3, column=0, padx=5, pady=15, sticky=tk.W
        )

        self.progress_label = ttk.Label(main_frame, text="状态: 等待选择图片", style='Status.TLabel')
        self.progress_label.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)

        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate', length=400)
        self.progress_bar.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(main_frame, text="已选择图片:", style='Header.TLabel').grid(
            row=6, column=0, padx=5, pady=(15, 5), sticky=tk.W
        )
        self.image_listbox = tk.Listbox(main_frame, width=90, height=8, bg='#f8f9fa',
                                        selectbackground='#3498db', selectforeground='white',
                                        font=('Consolas', 9))
        self.image_listbox.grid(row=7, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E))

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(7, weight=1)

    def select_images(self):
        paths = filedialog.askopenfilenames(
            title="选择图片（追加到列表）",
            filetypes=[("图片文件", "*.jpg;*.jpeg;*.png")]
        )

        if paths:
            for path in paths:
                if path not in self.img_paths:
                    self.img_paths.append(path)
            self.update_image_list()
            self.progress_label.config(text=f"已选择 {len(self.img_paths)} 张图片")

    def select_folder(self):
        folder = filedialog.askdirectory(title="选择图片文件夹（追加到列表）")

        if folder:
            valid_extensions = ('.jpg', '.jpeg', '.png')
            paths = sorted([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(valid_extensions)
            ])

            if paths:
                for path in paths:
                    if path not in self.img_paths:
                        self.img_paths.append(path)
                self.update_image_list()
                self.progress_label.config(text=f"从文件夹追加了 {len(paths)} 张图片，共 {len(self.img_paths)} 张")
            else:
                messagebox.showwarning("警告", "文件夹中未找到图片")

    def clear_images(self):
        self.img_paths = []
        self.update_image_list()
        self.progress_label.config(text="图片列表已清空")

    def update_image_list(self):
        self.image_listbox.delete(0, tk.END)
        for path in self.img_paths:
            self.image_listbox.insert(tk.END, os.path.basename(path))

    def start_stitching(self):
        if not self.img_paths:
            messagebox.showwarning("警告", "请先选择图片")
            return

        if len(self.img_paths) < 2:
            messagebox.showwarning("警告", "请至少选择两张图片")
            return

        self.progress_bar.start()
        self.progress_label.config(text="正在拼接...")

        thread = threading.Thread(target=self.stitch_in_background)
        thread.daemon = True
        thread.start()

    def stitch_in_background(self):
        try:
            img1 = cv2.imread(self.img_paths[0])
            img2 = cv2.imread(self.img_paths[1])

            if img1 is None:
                raise FileNotFoundError(f"无法读取图片: {self.img_paths[0]}")
            if img2 is None:
                raise FileNotFoundError(f"无法读取图片: {self.img_paths[1]}")

            images_dir = os.path.dirname(self.img_paths[0])
            src_pts_path = os.path.join(images_dir, "src_pts.npy")
            dst_pts_path = os.path.join(images_dir, "dst_pts.npy")

            if os.path.exists(src_pts_path) and os.path.exists(dst_pts_path):
                src_pts = np.load(src_pts_path)
                dst_pts = np.load(dst_pts_path)
                self.progress_label.config(text="使用预计算匹配点...")
            else:
                self.progress_label.config(text="特征匹配中...")
                orb = cv2.ORB_create(3000)
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                kp1, des1 = orb.detectAndCompute(gray1, None)
                kp2, des2 = orb.detectAndCompute(gray2, None)

                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = bf.knnMatch(des1, des2, k=2)

                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

            self.progress_label.config(text="计算单应矩阵...")
            panorama = stitch_images(img1, img2, src_pts, dst_pts)

            if getattr(sys, 'frozen', False):
                BASE_DIR = os.path.dirname(sys.executable)
            else:
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(BASE_DIR, "output")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, "my_panorama.jpg")
            cv2.imwrite(output_path, panorama)

            self.progress_label.config(text=f"拼接完成! 尺寸: {panorama.shape[1]}x{panorama.shape[0]}")
            self.show_result(output_path)

        except Exception as e:
            self.progress_label.config(text=f"拼接失败: {str(e)}")
            messagebox.showerror("错误", f"拼接过程中发生错误: {str(e)}")
        finally:
            self.progress_bar.stop()

    def show_result(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            messagebox.showerror("错误", "无法读取拼接结果")
            return

        height, width = img.shape[:2]
        max_size = 1000
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            img = cv2.resize(img, (int(width * scale), int(height * scale)))

        cv2.namedWindow("拼接结果", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("拼接结果", min(900, img.shape[1]), min(700, img.shape[0]))
        cv2.imshow("拼接结果", img)


if __name__ == "__main__":
    root = tk.Tk()
    app = StitchGUI(root)
    root.mainloop()
    cv2.destroyAllWindows()
