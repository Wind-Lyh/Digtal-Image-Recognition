import sys
import os
import io
from typing import List

import cv2
import numpy as np

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PySide6.QtCore import QThread, Signal, QObject, Qt
from PySide6.QtGui import QImage, QPixmap

# 导入 Qt Designer 生成的 UI 文件（请确保其命名为 ui_main.py 且类名为 Ui_MainWindow）
from ui_main import Ui_MainWindow

# 引入核心算法层
from core.homography_estimator import HomographyEstimator, HomographyEstimationError, InsufficientPointsError
from core.image_warper import ImageWarper
from core.image_blender import ImageBlender
from utils.image_io import crop_black_border, save_panorama
from feature_matcher import ORBFeatureExtractor


class EmittingStream(QObject):
    """
    用于将 sys.stdout (如 print 输出) 重定向到 Qt 信号，
    实现“日志劫持”在界面上实时显示。
    """
    textWritten = Signal(str)

    def write(self, text):
        if text.strip():  # 过滤空行
            self.textWritten.emit(str(text))

    def flush(self):
        pass


class WorkerSignals(QObject):
    """
    状态与数据的实时通信信号
    """
    progress = Signal(int)             # 进度条
    log = Signal(str)                  # 日志文本
    result_img = Signal(np.ndarray)    # 最终输出图像 (BGR)
    finished = Signal(bool, str)       # 结束状态 (是否成功, 报错信息)


class StitchWorker(QThread):
    """
    异步拼接核心 Worker (彻底分离 UI 和耗时计算，保障主界面极致流畅)
    """
    def __init__(self, img_paths: List[str], blend_mode: str):
        super().__init__()
        self.img_paths = img_paths
        self.blend_mode = blend_mode
        self.signals = WorkerSignals()

    def run(self):
        try:
            self.signals.log.emit("====== 开始全景拼接任务 ======")
            self.signals.log.emit(f"融合模式: {self.blend_mode}")
            self.signals.progress.emit(5)
            
            # 1. 图像读取
            imgs = []
            for path in self.img_paths:
                img = cv2.imread(path)
                if img is not None:
                    imgs.append(img)
                    self.signals.log.emit(f"成功读取图像: {os.path.basename(path)}，尺寸: {img.shape}")
                else:
                    self.signals.log.emit(f"[警告] 无法读取图像: {path}")

            if len(imgs) < 2:
                raise ValueError("有效图像不足2张，拼接终止。")

            self.signals.progress.emit(10)

            # 2. 向心拼接管道 (以中心图片为锚点，向两端扩展)
            # 自动计算锚点索引
            N = len(imgs)
            anchor_idx = N // 2
            self.signals.log.emit(f"采用向心拼接逻辑，自动选中中心图 (索引 {anchor_idx}) 作为锚点")
            
            result = imgs[anchor_idx]
            
            # 内部辅助函数：执行单次向心拼接
            def stitch_single(ref_img, warp_img, step_desc, step_progress):
                self.signals.log.emit(f"\n---> {step_desc}...")
                
                # ORB 提取
                self.signals.log.emit("提取 ORB 特征...")
                extractor = ORBFeatureExtractor(n_features=2000)
                gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
                gray_warp = cv2.cvtColor(warp_img, cv2.COLOR_BGR2GRAY)
                kp1, des1 = extractor.detect_and_compute(gray_ref)
                kp2, des2 = extractor.detect_and_compute(gray_warp)
                self.signals.log.emit(f"检测到特征点：基准图={len(kp1)}，新图={len(kp2)}")
                
                # BFMatcher 匹配
                self.signals.log.emit("进行特征匹配 (BFMatcher + KNN)...")
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = bf.knnMatch(des1, des2, k=2)
                
                # Lowe's ratio test 过滤
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                
                self.signals.log.emit(f"优质匹配点数量: {len(good_matches)}")
                if len(good_matches) < 10:
                    raise ValueError(f"优质匹配点数量不足 ({len(good_matches)} < 10)，无法计算单应矩阵！")
                
                # 提取坐标 (修正了原先映射错误的 Bug)
                # kp2 是待扭曲的新图，kp1 是基准主视图
                # 计算从 warp_img 到 ref_img 的单应矩阵
                src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
                dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
                
                # 单应矩阵计算
                self.signals.log.emit("计算单应矩阵 (RANSAC)...")
                H, mask = HomographyEstimator.compute(src_pts, dst_pts)
                inliers = np.sum(mask)
                self.signals.log.emit(f"单应矩阵计算成功！RANSAC 内点数量: {inliers}")
                
                self.signals.progress.emit(step_progress)
                
                # 透视投影变换
                self.signals.log.emit("执行透视投影变换并扩展画布...")
                canvas_width, canvas_height, dx, dy = ImageWarper.get_canvas_size([ref_img, warp_img], [np.eye(3), H])
                warped_ref = ImageWarper.warp_image(ref_img, np.eye(3), canvas_width, canvas_height, dx, dy)
                warped_warp = ImageWarper.warp_image(warp_img, H, canvas_width, canvas_height, dx, dy)
                
                # 图像融合
                self.signals.log.emit(f"开始图像融合，应用 {self.blend_mode} 算法...")
                return ImageBlender.blend(warped_ref, warped_warp, mode=self.blend_mode)

            # --- 右侧扩展 (向右拼接) ---
            for i in range(anchor_idx + 1, N):
                desc = f"正在将右侧第 {i} 张图向中心锚点图合并"
                prog = 10 + int(((i - anchor_idx) / N) * 70)
                result = stitch_single(result, imgs[i], desc, prog)
                
            # --- 左侧扩展 (向左拼接) ---
            for i in range(anchor_idx - 1, -1, -1):
                desc = f"正在将左侧第 {i} 张图向中心锚点图合并"
                prog = 10 + int(((anchor_idx - i + (N - anchor_idx - 1)) / N) * 70)
                result = stitch_single(result, imgs[i], desc, prog)
                
            self.signals.progress.emit(85)
            
            # 3. 结果后处理 (裁剪黑边)
            self.signals.log.emit("正在自动裁剪纯黑边...")
            final_img = crop_black_border(result)
            self.signals.progress.emit(95)
            
            # 保存本地
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            out_path = os.path.join(BASE_DIR, "output", "panorama_result.jpg")
            if save_panorama(final_img, out_path):
                self.signals.log.emit(f"高清全景图已成功保存至: {out_path}")
            
            self.signals.progress.emit(100)
            self.signals.log.emit("====== 拼接全部完成！ ======")
            
            # 传递 OpenCV 图像到主线程
            self.signals.result_img.emit(final_img)
            self.signals.finished.emit(True, "Success")
            
        except Exception as e:
            self.signals.log.emit(f"[致命错误] 拼接中断: {str(e)}")
            self.signals.finished.emit(False, str(e))


class MainWindow(QMainWindow, Ui_MainWindow):
    """
    负责调度算法层与 UI 层的控制层 (ViewModel)
    """
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.img_paths = []
        
        # 1. 信号与槽绑定
        self._connect_signals()
        
        # 2. 劫持系统 print 日志，使其输出到 UI 的 QTextEdit 中
        self.original_stdout = sys.stdout
        self.stdout_redirector = EmittingStream()
        self.stdout_redirector.textWritten.connect(self.append_log)
        sys.stdout = self.stdout_redirector
        
        self.append_log("系统初始化完成，PySide6 环境已就绪。")

    def _connect_signals(self):
        """绑定按钮等界面的交互事件 (注意：请核对 UI 组件的 objectName)"""
        if hasattr(self, 'btn_select_images'):
            self.btn_select_images.clicked.connect(self.select_images)
        if hasattr(self, 'btn_select_folder'):
            self.btn_select_folder.clicked.connect(self.select_folder)
        if hasattr(self, 'btn_clear'):
            self.btn_clear.clicked.connect(self.clear_images)
        if hasattr(self, 'btn_start'):
            self.btn_start.clicked.connect(self.start_stitching)

    def closeEvent(self, event):
        """窗口关闭时还原系统输出流，防止异常"""
        sys.stdout = self.original_stdout
        super().closeEvent(event)

    def append_log(self, text):
        """将日志追加到 textEdit_log"""
        if hasattr(self, 'textEdit_log'):
            self.textEdit_log.append(text)
            self.textEdit_log.ensureCursorVisible()
        else:
            print(text, file=self.original_stdout)

    def update_list_ui(self):
        """刷新 ListView / ListWidget 中的文件列表"""
        if hasattr(self, 'listWidget_images'):
            self.listWidget_images.clear()
            for path in self.img_paths:
                self.listWidget_images.addItem(os.path.basename(path))

    def select_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择待拼接图片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if paths:
            for p in paths:
                if p not in self.img_paths:
                    self.img_paths.append(p)
            self.update_list_ui()
            self.append_log(f"已载入 {len(paths)} 张图片。当前总计: {len(self.img_paths)}")

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
            new_paths = [
                os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith(valid_exts)
            ]
            new_paths.sort()
            for p in new_paths:
                if p not in self.img_paths:
                    self.img_paths.append(p)
            self.update_list_ui()
            self.append_log(f"批量导入完成。当前总计: {len(self.img_paths)} 张。")

    def clear_images(self):
        self.img_paths.clear()
        self.update_list_ui()
        if hasattr(self, 'label_preview'):
            self.label_preview.clear()
            self.label_preview.setText("预览区域")
        if hasattr(self, 'progressBar'):
            self.progressBar.setValue(0)
        self.append_log("任务列表与预览已清空。")

    def set_ui_enabled(self, enabled: bool):
        """交互锁死控制，防止多线程竞争导致崩溃"""
        widgets = ['btn_select_images', 'btn_select_folder', 
                   'btn_clear', 'btn_start', 'combo_blend_mode']
        for w_name in widgets:
            if hasattr(self, w_name):
                getattr(self, w_name).setEnabled(enabled)

    def start_stitching(self):
        if len(self.img_paths) < 2:
            QMessageBox.warning(self, "温馨提示", "请至少选择两张图片！")
            return
            
        # 从 UI 下拉框读取融合模式参数（修复了原 gui.py 无法传参的 Bug）
        # 约定：下拉项若包含 "multi" 字眼，则为 multiband，否则默认 linear
        blend_mode = "linear"
        if hasattr(self, 'combo_blend_mode'):
            mode_text = self.combo_blend_mode.currentText().lower()
            if "multi" in mode_text or "多频段" in mode_text:
                blend_mode = "multiband"
        
        self.append_log(f"-> 准备启动后台线程，模式={blend_mode}...")
        
        # 开启界面锁死
        self.set_ui_enabled(False)
        
        if hasattr(self, 'progressBar'):
            self.progressBar.setValue(0)
            
        # 实例化异步线程
        self.worker = StitchWorker(self.img_paths, blend_mode)
        
        # 绑定核心信号与槽
        self.worker.signals.log.connect(self.append_log)
        self.worker.signals.progress.connect(self.update_progress)
        self.worker.signals.result_img.connect(self.show_result_image)
        self.worker.signals.finished.connect(self.on_stitch_finished)
        
        # 启动计算
        self.worker.start()

    def update_progress(self, val: int):
        if hasattr(self, 'progressBar'):
            self.progressBar.setValue(val)

    def show_result_image(self, img_bgr: np.ndarray):
        """
        [极关键] 将后台 OpenCV 传出的 BGR 字节流转换为 Qt 的 QPixmap。
        包含：1. 通道翻转 (BGR -> RGB)
              2. BytesPerLine 字节对齐
              3. SmoothTransformation 高质量抗锯齿缩放
        """
        if not hasattr(self, 'label_preview'):
            return
            
        try:
            # 1. 颜色空间转换
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # 2. 解决底层 C++ 数据字节对齐问题
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            
            # 3. 构造内存安全的 QImage
            q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # 4. 高质量平滑缩放以适应 label_preview
            label_size = self.label_preview.size()
            scaled_pixmap = pixmap.scaled(
                label_size, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            self.label_preview.setPixmap(scaled_pixmap)
        except Exception as e:
            self.append_log(f"[UI 渲染错误] 无法渲染最终图像: {e}")

    def on_stitch_finished(self, success: bool, msg: str):
        """子线程结束后解除界面锁死"""
        self.set_ui_enabled(True)
        if not success:
            QMessageBox.critical(self, "拼接出错", f"发生错误:\n{msg}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
