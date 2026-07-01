import sys
import os
import io
import time
import uuid
from typing import List

import cv2
import numpy as np

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QListWidgetItem, QTableWidgetItem
from PySide6.QtCore import QThread, Signal, QObject, Qt, QSize
from PySide6.QtGui import QImage, QPixmap, QIcon
import datetime

from ui_main import Ui_MainWindow
from feature_matcher import match_images_from_memory
from test_my_data import run_stitching_pipeline, smart_topology_probe
from utils.image_io import crop_black_border, save_panorama
from config_manager import get_config, set_config, save_config
from history_window import HistoryWindow
import history_manager

class EmittingStream(QObject):
    textWritten = Signal(str)
    def write(self, text):
        if text.strip():
            self.textWritten.emit(str(text))
    def flush(self): pass

class WorkerSignals(QObject):
    progress = Signal(int)
    log = Signal(str)
    result_img = Signal(np.ndarray)
    finished = Signal(bool, str)
    update_combo = Signal(str)

class StitchWorker(QThread):
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
            
            imgs = []
            for path in self.img_paths:
                img = cv2.imread(path)
                if img is not None: imgs.append(img)
            
            if len(imgs) < 2:
                raise ValueError("有效图像不足2张，拼接终止。")

            panorama = imgs[0]
            for i in range(1, len(imgs)):
                next_img = imgs[i]
                self.signals.log.emit(f"\n---> 正在将第 {i+1}/{len(imgs)} 张图拼入全景...")
                
                src_pts, dst_pts, kp1, kp2, good_matches = match_images_from_memory(panorama, next_img)
                if good_matches is None:
                    raise ValueError(f"第 {i+1} 张图与前一张图重叠度过低，特征点匹配过少，拼接终止！")
                    
                current_blend_mode = self.blend_mode
                if self.smart_probe:
                    self.signals.log.emit("🔍 [智能探针] 正在执行自适应检测...")
                    direction_msg, force_multiband, exposure_diff = smart_topology_probe(
                        panorama, next_img, src_pts, dst_pts, exposure_thresh=25
                    )
                    self.signals.log.emit(f"🔮 [智能探针] 自动识别空间布局：{direction_msg}")
                    
                    if force_multiband and current_blend_mode != "multiband":
                        self.signals.log.emit(f"⚠️ [智能探针] 检测到重叠区曝光差异过大 (Diff={exposure_diff:.1f})！")
                        self.signals.log.emit("⚠️ 系统已自动强制覆盖用户参数，启用 Multi-band 融合！")
                        current_blend_mode = "multiband"
                        self.signals.update_combo.emit("多频段融合 (Multi-band)")
                
                panorama = run_stitching_pipeline(
                    panorama, next_img, src_pts, dst_pts, blend_mode=current_blend_mode
                )
                prog = 10 + int((i / (len(imgs) - 1)) * 85)
                self.signals.progress.emit(prog)
            
            self.signals.progress.emit(100)
            self.signals.log.emit("====== 拼接全部完成！ ======")
            self.signals.result_img.emit(panorama)
            self.signals.finished.emit(True, "Success")
            
        except Exception as e:
            self.signals.log.emit(f"[致命错误] 拼接中断: {str(e)}")
            self.signals.finished.emit(False, str(e))

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.img_paths = []
        
        self.history_cache = history_manager.load_history()
        self._current_result_bgr = None
        self.is_history_mode = False
        self.current_start_time = 0
        self.history_dialog = None
        
        self._init_config_ui()
        self._connect_signals()
        
        self.original_stdout = sys.stdout
        self.stdout_redirector = EmittingStream()
        self.stdout_redirector.textWritten.connect(self.append_log)
        sys.stdout = self.stdout_redirector
        self.append_log("系统初始化完成，PySide6 环境已就绪。")
        self.refresh_stats()

    def _init_config_ui(self):
        bm = get_config("blend_mode", "linear")
        if bm == "multiband":
            self.combo_blend_mode.setCurrentIndex(1)
        else:
            self.combo_blend_mode.setCurrentIndex(0)
            
        orb = get_config("orb_features", 3000)
        self.slider_orb.setValue(orb)
        self.label_orb_val.setText(str(orb))
        
        ransac = get_config("ransac_thresh", 5.0)
        self.slider_ransac.setValue(int(ransac * 10))
        self.label_ransac_val.setText(str(ransac))
        
        q = get_config("jpeg_quality", 95)
        self.slider_quality.setValue(q)
        self.label_quality_val.setText(str(q))

    def open_history_window(self):
        # 如果窗口已存在，则置顶
        if self.history_dialog is not None and self.history_dialog.isVisible():
            self.history_dialog.raise_()
            self.history_dialog.activateWindow()
            return
            
        self.history_dialog = HistoryWindow(self)
        self.history_dialog.preview_requested.connect(self.show_history_preview)
        self.history_dialog.show()

    def show_history_preview(self, idx):
        if idx is None or idx >= len(self.history_cache): return
        
        history_item = self.history_cache[idx]
        self.is_history_mode = True
        self.label_preview_title.setText("全景拼接预览区域 [⏳ 历史回看模式]")
        self.btn_smart_probe.setEnabled(False)
        self.btn_start.setEnabled(False)
        
        self.textEdit_log.setPlainText(history_item.get("log", ""))
        self.textEdit_log.verticalScrollBar().setValue(self.textEdit_log.verticalScrollBar().maximum())
        self.textEdit_log.append("\n[ ⏳ 当前处于历史回看模式，无法拼接。如需继续，请关闭弹窗并上传新图片 ]")
        
        if history_item.get("image_path") and os.path.exists(history_item["image_path"]):
            bgr = cv2.imread(history_item["image_path"])
            if bgr is not None:
                self.show_result_image(bgr, is_history=True)

    def refresh_stats(self):
        stats = history_manager.get_statistics()
        self.label_stats.setText(f"统计: 已拼接 {stats['total']} 次 | 成功率 {stats['success_rate']}% | 平均耗时 {stats['avg_time']}s")

    def _connect_signals(self):
        self.btn_select_images.clicked.connect(self.select_images)
        self.btn_select_folder.clicked.connect(self.select_folder)
        self.btn_clear.clicked.connect(self.clear_images)
        self.btn_smart_probe.clicked.connect(self.start_smart_stitching)
        self.btn_start.clicked.connect(self.start_manual_stitching)
        self.btn_history.clicked.connect(self.open_history_window)
        
        self.combo_blend_mode.currentTextChanged.connect(self._on_config_changed)
        self.slider_orb.valueChanged.connect(self._on_slider_orb)
        self.slider_ransac.valueChanged.connect(self._on_slider_ransac)
        self.slider_quality.valueChanged.connect(self._on_slider_quality)

    def _on_slider_orb(self, val):
        self.label_orb_val.setText(str(val))
        self._on_config_changed()
        
    def _on_slider_ransac(self, val):
        self.label_ransac_val.setText(str(val / 10.0))
        self._on_config_changed()
        
    def _on_slider_quality(self, val):
        self.label_quality_val.setText(str(val))
        self._on_config_changed()

    def _on_config_changed(self, _=None):
        mode_text = self.combo_blend_mode.currentText().lower()
        bm = "multiband" if "multi" in mode_text else "linear"
        set_config("blend_mode", bm)
        set_config("orb_features", self.slider_orb.value())
        set_config("ransac_thresh", self.slider_ransac.value() / 10.0)
        set_config("jpeg_quality", self.slider_quality.value())
        save_config()

    def _exit_history_mode_if_needed(self):
        if self.is_history_mode:
            self.is_history_mode = False
            self.label_preview_title.setText("全景拼接预览区域")
            self.btn_smart_probe.setEnabled(True)
            self.btn_start.setEnabled(True)
            self.append_log("\n====== 已退出历史回看模式，恢复拼接功能 ======")

    def on_btn_compare_clicked(self):
        # 已经被独立弹窗 HistoryWindow 接管，此方法废弃，但保留防止异常调用
        pass

    def closeEvent(self, event):
        sys.stdout = self.original_stdout
        super().closeEvent(event)

    def append_log(self, text):
        self.textEdit_log.append(text)
        self.textEdit_log.ensureCursorVisible()

    def update_list_ui(self):
        self.listWidget_images.clear()
        for path in self.img_paths:
            self.listWidget_images.addItem(os.path.basename(path))

    def select_images(self):
        self._exit_history_mode_if_needed()
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择待拼接图片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if paths:
            for p in paths:
                if p not in self.img_paths:
                    self.img_paths.append(p)
            self.update_list_ui()

    def select_folder(self):
        self._exit_history_mode_if_needed()
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
            new_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid_exts)]
            new_paths.sort()
            for p in new_paths:
                if p not in self.img_paths:
                    self.img_paths.append(p)
            self.update_list_ui()

    def clear_images(self):
        self._exit_history_mode_if_needed()
        self.img_paths.clear()
        self.update_list_ui()
        self.label_preview.clear()
        self.label_preview.setText("暂无全景图预览")
        self.progressBar.setValue(0)
        self.append_log("任务列表与预览已清空。")

    def clear_inputs_only(self):
        self.img_paths.clear()
        self.update_list_ui()
        self.append_log("\n[流水线] 上一批次已归档。输入区已重置，等待拖入新图片...")

    def clear_history(self):
        self.history_cache = []
        history_manager.save_history(self.history_cache)
        if self.history_dialog is not None and self.history_dialog.isVisible():
            self.history_dialog.populate_table()
        self.refresh_stats()
        self.append_log("历史记录已清空。")

    def set_ui_enabled(self, enabled: bool):
        widgets = ['btn_select_images', 'btn_select_folder', 'btn_clear', 'btn_smart_probe', 'btn_start', 'combo_blend_mode', 'slider_orb', 'slider_ransac', 'slider_quality', 'btn_clear_history']
        for w_name in widgets:
            if hasattr(self, w_name):
                getattr(self, w_name).setEnabled(enabled)

    def update_combo_ui(self, text: str):
        idx = self.combo_blend_mode.findText(text, Qt.MatchContains)
        if idx >= 0:
            self.combo_blend_mode.setCurrentIndex(idx)

    def start_smart_stitching(self):
        self._trigger_stitching(blend_mode=get_config("blend_mode", "linear"), smart_probe=True)
        
    def start_manual_stitching(self):
        self._trigger_stitching(blend_mode=get_config("blend_mode", "linear"), smart_probe=False)

    def _trigger_stitching(self, blend_mode: str, smart_probe: bool):
        if len(self.img_paths) < 2:
            QMessageBox.warning(self, "温馨提示", "请至少选择两张图片！")
            return
            
        self.current_start_time = time.time()
        self.append_log(f"\n-> 准备启动后台线程，模式={blend_mode}，智能探针={smart_probe}...")
        self.set_ui_enabled(False)
        self.progressBar.setValue(0)
            
        self.worker = StitchWorker(self.img_paths, blend_mode, smart_probe=smart_probe)
        self.worker.signals.progress.connect(self.progressBar.setValue)
        self.worker.signals.log.connect(self.append_log)
        self.worker.signals.result_img.connect(self.show_result_image)
        self.worker.signals.finished.connect(self.on_stitch_finished)
        self.worker.signals.update_combo.connect(self.update_combo_ui)
        self.worker.start()

    def show_result_image(self, img_bgr: np.ndarray, is_history: bool = False):
        if not hasattr(self, 'label_preview'): return
        if not is_history: self._current_result_bgr = img_bgr.copy()
            
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            label_size = self.label_preview.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label_preview.setPixmap(scaled_pixmap)
        except Exception as e:
            self.append_log(f"[UI 渲染错误] 无法渲染最终图像: {e}")

    def on_stitch_finished(self, success: bool, msg: str):
        self.set_ui_enabled(True)
        duration = time.time() - self.current_start_time
        
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "success" if success else "failed"
        
        img_path = ""
        if success and self._current_result_bgr is not None:
            uid = str(uuid.uuid4())[:8]
            img_path = os.path.join(history_manager.HISTORY_IMG_DIR, f"pano_{uid}.jpg")
            save_panorama(self._current_result_bgr, img_path)

        history_item = {
            "timestamp": ts,
            "num_imgs": len(self.img_paths),
            "duration": round(duration, 1),
            "blend_mode": get_config("blend_mode", "linear"),
            "status": status,
            "image_path": img_path,
            "img_paths": self.img_paths.copy(),
            "note": "",
            "log": self.textEdit_log.toPlainText()
        }
        
        self.history_cache.append(history_item)
        history_manager.save_history(self.history_cache)
        idx = len(self.history_cache) - 1
        
        self.refresh_stats()
        if self.history_dialog is not None and self.history_dialog.isVisible():
            self.history_dialog.populate_table()
            
        if success:
            self.clear_inputs_only()
        else:
            QMessageBox.critical(self, "拼接出错", f"发生错误:\n{msg}")

    def on_history_clicked(self, item):
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
