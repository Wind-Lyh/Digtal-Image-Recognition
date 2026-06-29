import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

from main import StitchPipeline


class StitchGUI:
    """全景图像拼接GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("全景图像拼接系统")
        self.root.geometry("800x600")
        
        self.img_paths = []
        self.pipeline = StitchPipeline()
        self.blend_mode = tk.StringVar(value="linear")
        
        self.create_widgets()
    
    def create_widgets(self):
        """创建GUI组件"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        ttk.Label(main_frame, text="全景图像拼接系统", font=('Helvetica', 16)).grid(
            row=0, column=0, columnspan=3, pady=10
        )
        
        ttk.Button(main_frame, text="选择图片", command=self.select_images).grid(
            row=1, column=0, padx=5, pady=5, sticky=tk.W
        )
        
        ttk.Button(main_frame, text="选择文件夹", command=self.select_folder).grid(
            row=1, column=1, padx=5, pady=5, sticky=tk.W
        )
        
        ttk.Label(main_frame, text="融合方式:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Radiobutton(main_frame, text="线性融合", variable=self.blend_mode, value="linear").grid(
            row=2, column=1, padx=5, pady=5, sticky=tk.W
        )
        ttk.Radiobutton(main_frame, text="多频段融合", variable=self.blend_mode, value="multiband").grid(
            row=2, column=2, padx=5, pady=5, sticky=tk.W
        )
        
        ttk.Button(main_frame, text="开始拼接", command=self.start_stitching).grid(
            row=3, column=0, padx=5, pady=10, sticky=tk.W
        )
        
        self.progress_label = ttk.Label(main_frame, text="状态: 等待选择图片")
        self.progress_label.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress_bar.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E))
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        
        self.image_listbox = tk.Listbox(main_frame, width=80, height=10)
        self.image_listbox.grid(row=6, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E))
    
    def select_images(self):
        """选择多张图片"""
        paths = filedialog.askopenfilenames(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg;*.jpeg;*.png")]
        )
        
        if paths:
            self.img_paths = list(paths)
            self.update_image_list()
            self.progress_label.config(text=f"已选择 {len(self.img_paths)} 张图片")
    
    def select_folder(self):
        """选择包含图片的文件夹"""
        folder = filedialog.askdirectory(title="选择图片文件夹")
        
        if folder:
            valid_extensions = ('.jpg', '.jpeg', '.png')
            paths = sorted([
                os.path.join(folder, f) 
                for f in os.listdir(folder) 
                if f.lower().endswith(valid_extensions)
            ])
            
            if paths:
                self.img_paths = paths
                self.update_image_list()
                self.progress_label.config(text=f"从文件夹选择了 {len(self.img_paths)} 张图片")
            else:
                messagebox.showwarning("警告", "文件夹中未找到图片")
    
    def update_image_list(self):
        """更新图片列表显示"""
        self.image_listbox.delete(0, tk.END)
        for path in self.img_paths:
            self.image_listbox.insert(tk.END, os.path.basename(path))
    
    def start_stitching(self):
        """开始拼接（在后台线程中执行）"""
        if not self.img_paths:
            messagebox.showwarning("警告", "请先选择图片")
            return
        
        self.progress_bar.start()
        self.progress_label.config(text="正在拼接...")
        
        thread = threading.Thread(target=self.stitch_in_background)
        thread.daemon = True
        thread.start()
    
    def stitch_in_background(self):
        """后台执行拼接"""
        try:
            output_path = self.pipeline.run_real(self.img_paths, blend_mode=self.blend_mode.get())
            
            if output_path and os.path.exists(output_path):
                self.progress_label.config(text=f"拼接完成! 结果已保存到 {output_path}")
                self.show_result(output_path)
            else:
                self.progress_label.config(text="拼接失败")
                messagebox.showerror("错误", "拼接过程中发生错误")
        except Exception as e:
            self.progress_label.config(text=f"拼接失败: {str(e)}")
            messagebox.showerror("错误", f"拼接过程中发生错误: {str(e)}")
        finally:
            self.progress_bar.stop()
    
    def show_result(self, image_path):
        """显示拼接结果"""
        result_window = tk.Toplevel(self.root)
        result_window.title("拼接结果")
        
        img = cv2.imread(image_path)
        if img is None:
            messagebox.showerror("错误", "无法读取拼接结果")
            return
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img_rgb.shape[:2]
        
        max_size = 800
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            width = int(width * scale)
            height = int(height * scale)
            img_rgb = cv2.resize(img_rgb, (width, height))
        
        photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        
        label = ttk.Label(result_window, image=photo)
        label.image = photo
        label.pack(padx=10, pady=10)
        
        ttk.Label(result_window, text=f"图片尺寸: {width} x {height}").pack(pady=5)


if __name__ == "__main__":
    root = tk.Tk()
    app = StitchGUI(root)
    root.mainloop()
