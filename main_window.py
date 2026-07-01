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

# 引入开发者A的核心算法
from feature_matcher import match_images_from_memory
from test_my_data import run_stitching_pipeline, smart_topology_probe
from utils.image_io import crop_black_border, save_panorama


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
    update_combo = Signal(str)         # UI 下拉框联动信号


class StitchWorker(QThread):
    """
    异步拼接核心 Worker (彻底分离 UI 和耗时计算，保障主界面极致流畅)
    """
    def __init__(self, img_paths: List[str], blend_mode: str, smart_probe: bool = False):
        super().__init__()
        self.img_paths = img_paths
        self.blend_mode = blend_mode
        self.smart_probe = smart_probe
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

            # 2. 级联拼接管道 (调用 A 的算法作为唯一引擎)
            self.signals.log.emit("开始使用 A 引擎进行线性级联拼接...")
            
            panorama = imgs[0]
            
            for i in range(1, len(imgs)):
                next_img = imgs[i]
                self.signals.log.emit(f"\n---> 正在将第 {i+1}/{len(imgs)} 张图拼入全景...")
                
                # 步骤 A：内存提取特征与匹配
                self.signals.log.emit("提取特征点并进行匹配...")
                src_pts, dst_pts, kp1, kp2, good_matches = match_images_from_memory(panorama, next_img)
                
                if good_matches is None:
                    raise ValueError(f"第 {i+1} 张图与前一张图重叠度过低，特征点匹配过少，拼接被强制终止！")
                    
                self.signals.log.emit(f"成功提取优质匹配点对: {len(good_matches)}")
                
                # --- 智能探针逻辑 ---
                current_blend_mode = self.blend_mode
                if self.smart_probe:
                    self.signals.log.emit("🔍 [智能探针] 正在执行自适应检测...")
                    direction_msg, force_multiband, exposure_diff = smart_topology_probe(
                        panorama, next_img, src_pts, dst_pts, exposure_thresh=25
                    )
                    self.signals.log.emit(f"🔮 [智能探针] 自动识别空间布局：{direction_msg}")
                    
                    if force_multiband and current_blend_mode != "multiband":
                        self.signals.log.emit(f"⚠️ [智能探针] 检测到重叠区曝光差异过大 (Diff={exposure_diff:.1f})！")
                        self.signals.log.emit("⚠️ 为了保证光照平滑，系统已自动强制覆盖用户参数，启用 Multi-band 融合！")
                        current_blend_mode = "multiband"
                        self.signals.update_combo.emit("多频段融合 (Multi-band)")
                # ------------------
                
                # 步骤 B：调用 A 引擎核心流水线
                self.signals.log.emit(f"调用核心引擎流水线 (融合模式: {current_blend_mode})...")
                
                panorama = run_stitching_pipeline(
                    panorama, 
                    next_img, 
                    src_pts, 
                    dst_pts, 
                    blend_mode=current_blend_mode
                )
                
                # 更新进度
                prog = 10 + int((i / (len(imgs) - 1)) * 85)
                self.signals.progress.emit(prog)
            
            final_img = panorama
            
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
        """绑定按钮等界面的交互事件 (注意：请核对 UI 组件 of objectName)"""
        if hasattr(self, 'btn_select_images'):
            self.btn_select_images.clicked.connect(self.select_images)
        if hasattr(self, 'btn_select_folder'):
            self.btn_select_folder.clicked.connect(self.select_folder)
        if hasattr(self, 'btn_clear'):
            self.btn_clear.clicked.connect(self.clear_images)
        if hasattr(self, 'btn_smart_probe'):
            self.btn_smart_probe.clicked.connect(self.start_smart_stitching)
        if hasattr(self, 'btn_linear'):
            self.btn_linear.clicked.connect(self.start_linear_stitching)
        if hasattr(self, 'btn_multiband'):
            self.btn_multiband.clicked.connect(self.start_multiband_stitching)

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
                   'btn_clear', 'btn_smart_probe', 'btn_linear', 'btn_multiband']
        for w_name in widgets:
            if hasattr(self, w_name):
                getattr(self, w_name).setEnabled(enabled)

    def update_combo_ui(self, text: str):
        pass # 下拉框已被移除，故此函数空置

    def start_smart_stitching(self):
        self._trigger_stitching(blend_mode="linear", smart_probe=True)
        
    def start_linear_stitching(self):
        self._trigger_stitching(blend_mode="linear", smart_probe=False)

    def start_multiband_stitching(self):
        self._trigger_stitching(blend_mode="multiband", smart_probe=False)

    def _trigger_stitching(self, blend_mode: str, smart_probe: bool):
        if len(self.img_paths) < 2:
            QMessageBox.warning(self, "温馨提示", "请至少选择两张图片！")
            return
            
        probe_status = "开启" if smart_probe else "关闭"
        self.append_log(f"-> 准备启动后台线程，模式={blend_mode}，智能探针={probe_status}...")
        
        self.set_ui_enabled(False)
        
        if hasattr(self, 'progressBar'):
            self.progressBar.setValue(0)
            
        self.worker = StitchWorker(self.img_paths, blend_mode, smart_probe=smart_probe)
        
        self.worker.signals.progress.connect(self.progressBar.setValue if hasattr(self, 'progressBar') else lambda x: None)
        self.worker.signals.log.connect(self.append_log)
        self.worker.signals.result_img.connect(self.show_result_image)
        self.worker.signals.finished.connect(self.on_stitch_finished)
        self.worker.signals.update_combo.connect(self.update_combo_ui)
        
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
