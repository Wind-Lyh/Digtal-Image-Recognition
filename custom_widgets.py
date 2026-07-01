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


class InteractiveImageViewer(QWidget):
    """
    专业交互式全景图看图组件
    - 鼠标左键拖拽平移
    - 滚轮以鼠标悬停点为中心平滑缩放
    - 底部悬浮工具栏: 旋转 / 镜像 / 适应窗口 / 1:1 / 网格
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # === QGraphicsView 核心 ===
        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self._view.setDragMode(QGraphicsView.ScrollHandDrag)
        self._view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._view.setRenderHints(
            self._view.renderHints()
            | self._view.renderHints().__class__(0x04)  # SmoothPixmapTransform
        )
        self._view.setStyleSheet("border: none; background-color: #1e272e;")
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 拦截滚轮
        self._view.wheelEvent = self._on_wheel

        layout.addWidget(self._view, 1)

        # === 占位提示文字 ===
        self._placeholder = QLabel("暂无全景图预览")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            "color: #a4b0be; font-size: 16px; background: transparent;"
        )
        self._placeholder.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._placeholder.setParent(self._view.viewport())

        # === 网格叠加层 ===
        self._grid_visible = False
        self._grid_lines = []

        # === 悬浮工具栏 ===
        toolbar = QWidget(self)
        toolbar.setStyleSheet(
            "background-color: rgba(30,39,46,0.85); border-radius: 6px;"
        )
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(8, 4, 8, 4)
        tb_layout.setSpacing(4)

        btn_style = (
            "QPushButton { background: transparent; color: white; font-size: 18px;"
            " border: none; padding: 4px 8px; border-radius: 4px; }"
            "QPushButton:hover { background-color: rgba(255,255,255,0.15); }"
            "QPushButton:pressed { background-color: rgba(255,255,255,0.25); }"
        )

        self.btn_rotate = QPushButton("↻")
        self.btn_rotate.setToolTip("顺时针旋转 90°")
        self.btn_mirror = QPushButton("⇔")
        self.btn_mirror.setToolTip("水平镜像翻转")
        self.btn_fit = QPushButton("⊞")
        self.btn_fit.setToolTip("适应窗口 (Fit)")
        self.btn_actual = QPushButton("1:1")
        self.btn_actual.setToolTip("实际像素大小")
        self.btn_grid = QPushButton("▦")
        self.btn_grid.setToolTip("网格辅助线")

        for btn in [self.btn_rotate, self.btn_mirror, self.btn_fit, self.btn_actual, self.btn_grid]:
            btn.setStyleSheet(btn_style)
            btn.setFixedSize(36, 30)
            tb_layout.addWidget(btn)

        toolbar.adjustSize()
        self._toolbar = toolbar

        # 信号绑定
        self.btn_rotate.clicked.connect(self._rotate_90)
        self.btn_mirror.clicked.connect(self._mirror_h)
        self.btn_fit.clicked.connect(self.fit_in_view)
        self.btn_actual.clicked.connect(self._actual_pixels)
        self.btn_grid.clicked.connect(self._toggle_grid)

        self._rotation = 0  # 累计旋转角度

    # ========== 公共 API ==========

    def set_image_from_bgr(self, img_bgr):
        """接收 OpenCV BGR ndarray 并展示"""
        import cv2
        from PySide6.QtGui import QImage as _QImage
        import numpy as np
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = _QImage(img_rgb.data, w, h, bytes_per_line, _QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self._set_pixmap(pixmap)

    def set_image_from_path(self, path):
        """从文件路径加载"""
        if path and os.path.exists(path):
            self._set_pixmap(QPixmap(path))

    def clear_image(self):
        self._pixmap_item.setPixmap(QPixmap())
        self._scene.setSceneRect(0, 0, 0, 0)
        self._placeholder.show()
        self._rotation = 0
        self._clear_grid()

    def fit_in_view(self):
        if self._pixmap_item.pixmap() and not self._pixmap_item.pixmap().isNull():
            self._view.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    # ========== 内部方法 ==========

    def _set_pixmap(self, pixmap):
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self._placeholder.hide()
        self._rotation = 0
        self._clear_grid()
        self.fit_in_view()

    def _on_wheel(self, event):
        if event.angleDelta().y() > 0:
            factor = 1.15
        else:
            factor = 1.0 / 1.15
        self._view.scale(factor, factor)

    def _rotate_90(self):
        self._rotation = (self._rotation + 90) % 360
        self._view.rotate(90)

    def _mirror_h(self):
        self._view.scale(-1, 1)

    def _actual_pixels(self):
        self._view.resetTransform()
        if self._rotation != 0:
            self._view.rotate(self._rotation)

    def _toggle_grid(self):
        self._grid_visible = not self._grid_visible
        if self._grid_visible:
            self._draw_grid()
        else:
            self._clear_grid()

    def _draw_grid(self):
        self._clear_grid()
        pixmap = self._pixmap_item.pixmap()
        if pixmap is None or pixmap.isNull():
            return
        from PySide6.QtGui import QPen, QColor
        w = pixmap.width()
        h = pixmap.height()
        pen = QPen(QColor(255, 255, 255, 80), max(1, min(w, h) / 500))
        for i in range(1, 3):
            x = w * i / 3
            line = self._scene.addLine(x, 0, x, h, pen)
            self._grid_lines.append(line)
        for i in range(1, 3):
            y = h * i / 3
            line = self._scene.addLine(0, y, w, y, pen)
            self._grid_lines.append(line)

    def _clear_grid(self):
        for line in self._grid_lines:
            self._scene.removeItem(line)
        self._grid_lines.clear()

    # ========== 布局事件 ==========

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 将悬浮工具栏定位在底部中央
        tw = self._toolbar.sizeHint().width()
        th = self._toolbar.sizeHint().height()
        x = (self.width() - tw) // 2
        y = self.height() - th - 10
        self._toolbar.setGeometry(x, y, tw, th)
        # 占位文字居中
        self._placeholder.setGeometry(0, 0, self._view.viewport().width(), self._view.viewport().height())

