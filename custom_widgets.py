import os
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, 
                               QPushButton, QLineEdit, QGraphicsView, QGraphicsScene, 
                               QGraphicsPixmapItem, QSplitter, QCheckBox)
from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtGui import QPixmap, QDesktopServices, QWheelEvent

class HistoryCardWidget(QWidget):
    # 自定义信号
    locate_requested = Signal(int)
    restore_requested = Signal(int)
    note_changed = Signal(int, str)
    compare_checked = Signal(int, bool)
    card_clicked = Signal(int)
    
    def __init__(self, idx, item_data, parent=None):
        super().__init__(parent)
        self.idx = idx
        self.item_data = item_data
        
        self.init_ui()
        
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # 1. 勾选框
        self.checkbox = QCheckBox()
        self.checkbox.stateChanged.connect(self._on_check_changed)
        main_layout.addWidget(self.checkbox)
        
        # 2. 缩略图
        self.lbl_thumb = QLabel()
        self.lbl_thumb.setFixedSize(160, 90)
        self.lbl_thumb.setStyleSheet("background-color: #2f3542; border-radius: 4px;")
        self.lbl_thumb.setAlignment(Qt.AlignCenter)
        if self.item_data.get("image_path") and os.path.exists(self.item_data["image_path"]):
            pixmap = QPixmap(self.item_data["image_path"]).scaled(
                160, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.lbl_thumb.setPixmap(pixmap)
        else:
            self.lbl_thumb.setText("无全景图")
            self.lbl_thumb.setStyleSheet("color: white;")
        main_layout.addWidget(self.lbl_thumb)
        
        # 3. 信息与备注区
        info_layout = QVBoxLayout()
        
        status = "✅ 成功" if self.item_data.get("status") == "success" else "❌ 失败"
        ts = self.item_data.get("timestamp", "")
        imgs_count = self.item_data.get("num_imgs", 0)
        
        self.lbl_info = QLabel(f"<b>[{ts}]</b> {status} ({imgs_count}张图)")
        info_layout.addWidget(self.lbl_info)
        
        note_layout = QHBoxLayout()
        self.lbl_note = QLabel("备注:")
        self.edit_note = QLineEdit()
        self.edit_note.setPlaceholderText("在此输入备注，回车自动保存...")
        self.edit_note.setText(self.item_data.get("note", ""))
        self.edit_note.editingFinished.connect(self._on_note_changed)
        note_layout.addWidget(self.lbl_note)
        note_layout.addWidget(self.edit_note)
        
        info_layout.addLayout(note_layout)
        main_layout.addLayout(info_layout, stretch=1)
        
        # 4. 按钮区
        btn_layout = QVBoxLayout()
        self.btn_locate = QPushButton("📁 定位原图")
        self.btn_restore = QPushButton("🔄 恢复任务")
        
        # 旧记录兼容：若无原图路径则置灰
        if not self.item_data.get("img_paths"):
            self.btn_locate.setEnabled(False)
            self.btn_locate.setToolTip("旧记录无原图路径数据")
            self.btn_restore.setEnabled(False)
            self.btn_restore.setToolTip("旧记录无原图路径数据")
            
        self.btn_locate.clicked.connect(self._on_locate)
        self.btn_restore.clicked.connect(self._on_restore)
        
        btn_layout.addWidget(self.btn_locate)
        btn_layout.addWidget(self.btn_restore)
        main_layout.addLayout(btn_layout)

    def _on_check_changed(self, state):
        self.compare_checked.emit(self.idx, state == Qt.Checked)

    def _on_note_changed(self):
        new_note = self.edit_note.text().strip()
        if new_note != self.item_data.get("note", ""):
            self.note_changed.emit(self.idx, new_note)

    def _on_locate(self):
        self.locate_requested.emit(self.idx)

    def _on_restore(self):
        self.restore_requested.emit(self.idx)

    def mousePressEvent(self, event):
        self.card_clicked.emit(self.idx)
        super().mousePressEvent(event)

class SyncGraphicsView(QGraphicsView):
    # 将滚轮事件抛出，实现两个视口的同比例缩放
    zoom_requested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_obj = QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene_obj.addItem(self.pixmap_item)
        
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        # 固定缩放中心点为视图中心，保证左右屏缩放行为完全一致
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        
    def set_image(self, pixmap):
        self.pixmap_item.setPixmap(pixmap)
        self.scene_obj.setSceneRect(self.pixmap_item.boundingRect())
        self.fitInView(self.scene_obj.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            factor = 1.15
        else:
            factor = 1.0 / 1.15
        self.zoom_requested.emit(factor)
        
    def do_scale(self, factor):
        self.scale(factor, factor)


class DualImageViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        
        self.splitter = QSplitter(Qt.Horizontal)
        
        self.view_left = SyncGraphicsView()
        self.view_right = SyncGraphicsView()
        
        self.splitter.addWidget(self.view_left)
        self.splitter.addWidget(self.view_right)
        layout.addWidget(self.splitter)
        
        # ====== 核心：绑定滚动条，实现无缝联动平移 ======
        self.view_left.horizontalScrollBar().valueChanged.connect(self.view_right.horizontalScrollBar().setValue)
        self.view_right.horizontalScrollBar().valueChanged.connect(self.view_left.horizontalScrollBar().setValue)
        self.view_left.verticalScrollBar().valueChanged.connect(self.view_right.verticalScrollBar().setValue)
        self.view_right.verticalScrollBar().valueChanged.connect(self.view_left.verticalScrollBar().setValue)
        
        # ====== 核心：绑定缩放信号，实现同比例联动缩放 ======
        self.view_left.zoom_requested.connect(self.apply_zoom)
        self.view_right.zoom_requested.connect(self.apply_zoom)

    def apply_zoom(self, factor):
        self.view_left.do_scale(factor)
        self.view_right.do_scale(factor)

    def load_images(self, path1, path2):
        if os.path.exists(path1):
            self.view_left.set_image(QPixmap(path1))
        if os.path.exists(path2):
            self.view_right.set_image(QPixmap(path2))

    def reset_view(self):
        # 恢复单窗模式时，可以只加载一张图，或者隐藏一个 view
        self.view_left.set_image(QPixmap())
        self.view_right.set_image(QPixmap())
