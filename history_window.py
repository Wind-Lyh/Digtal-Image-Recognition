import os
import cv2
import numpy as np

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
                               QPushButton, QHeaderView, QMessageBox, QAbstractItemView, QLabel,
                               QSplitter, QWidget, QInputDialog, QLineEdit)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage

from custom_widgets import SyncGraphicsView
import history_manager

class ComparisonDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("全景拼接 - 合成前后双屏联动对比")
        self.resize(1200, 700)
        
        layout = QVBoxLayout(self)
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left Panel (Before)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(QLabel("<b>合成前 (原始图片序列)</b>"))
        self.view_before = SyncGraphicsView()
        left_layout.addWidget(self.view_before)
        
        # Right Panel (After)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(QLabel("<b>合成后 (全景图)</b>"))
        self.view_after = SyncGraphicsView()
        right_layout.addWidget(self.view_after)
        
        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)
        layout.addWidget(self.splitter)
        
        # Bind events
        self.view_before.horizontalScrollBar().valueChanged.connect(self.view_after.horizontalScrollBar().setValue)
        self.view_after.horizontalScrollBar().valueChanged.connect(self.view_before.horizontalScrollBar().setValue)
        self.view_before.verticalScrollBar().valueChanged.connect(self.view_after.verticalScrollBar().setValue)
        self.view_after.verticalScrollBar().valueChanged.connect(self.view_before.verticalScrollBar().setValue)
        
        self.view_before.zoom_requested.connect(self.apply_zoom)
        self.view_after.zoom_requested.connect(self.apply_zoom)

    def apply_zoom(self, factor):
        self.view_before.do_scale(factor)
        self.view_after.do_scale(factor)

    def load_data(self, img_paths, result_path):
        # Load Result (After)
        if result_path and os.path.exists(result_path):
            self.view_after.set_image(QPixmap(result_path))
            
        # Load Originals (Before) as a horizontal strip
        valid_imgs = []
        for p in img_paths:
            if os.path.exists(p):
                img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    valid_imgs.append(img)
                    
        if valid_imgs:
            # Resize all images to the same height for hconcat
            target_h = valid_imgs[0].shape[0]
            resized_imgs = []
            for img in valid_imgs:
                h, w = img.shape[:2]
                if h != target_h:
                    new_w = int(w * (target_h / h))
                    img = cv2.resize(img, (new_w, target_h))
                resized_imgs.append(img)
                
            strip_bgr = cv2.hconcat(resized_imgs)
            strip_rgb = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2RGB)
            strip_rgb = np.ascontiguousarray(strip_rgb)
            h, w, ch = strip_rgb.shape
            q_img = QImage(strip_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.view_before.set_image(QPixmap.fromImage(q_img))


class HistoryWindow(QDialog):
    preview_requested = Signal(int) # Emits idx of clicked item

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📜 历史档案库")
        self.resize(900, 500)
        # 非模态弹出，允许与主界面交互
        self.setWindowModality(Qt.NonModal)
        
        self.history_cache = history_manager.load_history()
        self.init_ui()
        self.populate_table()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "任务编号", "时间", "数量", "耗时(s)", "融合模式", "状态", "备注"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        
        layout.addWidget(self.table)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_edit_note = QPushButton("✍️ 编辑选中备注")
        self.btn_compare = QPushButton("📊 对比合成前后")
        self.btn_clear = QPushButton("🗑️ 清空所有记录")
        
        self.btn_edit_note.clicked.connect(self.edit_note)
        self.btn_compare.clicked.connect(self.open_comparison)
        self.btn_clear.clicked.connect(self.clear_history)
        
        btn_layout.addWidget(self.btn_edit_note)
        btn_layout.addWidget(self.btn_compare)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_clear)
        
        layout.addLayout(btn_layout)

    def populate_table(self):
        self.table.setRowCount(0)
        # 递减顺序：最新的记录排在最上面
        for row_idx, i in enumerate(range(len(self.history_cache) - 1, -1, -1)):
            item = self.history_cache[i]
            self.table.insertRow(row_idx)
            
            self.table.setItem(row_idx, 0, QTableWidgetItem(f"Task #{i+1}"))
            self.table.setItem(row_idx, 1, QTableWidgetItem(item.get('timestamp', '')))
            self.table.setItem(row_idx, 2, QTableWidgetItem(str(item.get('num_imgs', 0))))
            self.table.setItem(row_idx, 3, QTableWidgetItem(str(item.get('duration', 0))))
            self.table.setItem(row_idx, 4, QTableWidgetItem(item.get('blend_mode', '未知')))
            status_text = "✅ 成功" if item.get('status') == 'success' else "❌ 失败"
            self.table.setItem(row_idx, 5, QTableWidgetItem(status_text))
            self.table.setItem(row_idx, 6, QTableWidgetItem(item.get('note', '')))
            
            # 存储原始索引，确保预览/编辑/对比功能正确映射
            self.table.item(row_idx, 0).setData(Qt.UserRole, i)

    def on_selection_changed(self):
        selected = self.table.selectedItems()
        if not selected: return
        row = selected[0].row()
        idx = self.table.item(row, 0).data(Qt.UserRole)
        self.preview_requested.emit(idx)

    def edit_note(self):
        selected = self.table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "提示", "请先选择一条记录！")
            return
            
        row = selected[0].row()
        idx = self.table.item(row, 0).data(Qt.UserRole)
        old_note = self.history_cache[idx].get('note', '')
        
        new_note, ok = QInputDialog.getText(self, "编辑备注", "随手记:", QLineEdit.Normal, old_note)
        if ok and new_note != old_note:
            self.history_cache[idx]['note'] = new_note
            history_manager.save_history(self.history_cache)
            self.table.item(row, 6).setText(new_note)

    def open_comparison(self):
        selected = self.table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "提示", "请先选择一条记录！")
            return
            
        row = selected[0].row()
        idx = self.table.item(row, 0).data(Qt.UserRole)
        item = self.history_cache[idx]
        
        if item.get('status') != 'success' or not item.get('img_paths'):
            QMessageBox.warning(self, "提示", "该记录没有原图路径或未成功拼接，无法对比！")
            return
            
        dialog = ComparisonDialog(self)
        dialog.load_data(item.get('img_paths', []), item.get('image_path', ''))
        dialog.exec()

    def clear_history(self):
        reply = QMessageBox.question(self, "确认", "确定要清空所有历史记录吗？", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.history_cache = []
            history_manager.save_history(self.history_cache)
            self.populate_table()
