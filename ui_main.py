from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
    QListWidget, QMainWindow, QProgressBar, QPushButton,
    QSizePolicy, QSplitter, QTextEdit, QVBoxLayout,
    QWidget, QFrame)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1100, 750)
        MainWindow.setMinimumSize(QSize(900, 600))
        
        # 现代暗系/浅系精致配色 QSS 样式表
        MainWindow.setStyleSheet(u"""
            QMainWindow {
                background-color: #f5f6fa;
            }
            QWidget {
                font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
                font-size: 13px;
                color: #2f3640;
            }
            QFrame#left_panel {
                background-color: #ffffff;
                border-radius: 8px;
                border: 1px solid #dcdde1;
            }
            QPushButton {
                background-color: #353b48;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #718093;
            }
            QPushButton:pressed {
                background-color: #2f3640;
            }
            QPushButton#btn_start {
                background-color: #44bd32;
            }
            QPushButton#btn_start:hover {
                background-color: #4cd137;
            }
            QPushButton#btn_start:pressed {
                background-color: #44bd32;
            }
            QPushButton#btn_clear {
                background-color: #e84118;
            }
            QPushButton#btn_clear:hover {
                background-color: #c23616;
            }
            QPushButton#btn_clear:pressed {
                background-color: #e84118;
            }
            QPushButton#btn_smart_probe {
                background-color: #6c5ce7;
                font-size: 14px;
                padding: 10px 15px;
            }
            QPushButton#btn_smart_probe:hover {
                background-color: #8278fc;
            }
            QPushButton#btn_smart_probe:pressed {
                background-color: #574b90;
            }
            QPushButton#btn_linear {
                background-color: #487eb0;
            }
            QPushButton#btn_linear:hover {
                background-color: #54a0ff;
            }
            QPushButton#btn_linear:pressed {
                background-color: #273c75;
            }
            QPushButton#btn_multiband {
                background-color: #e1b12c;
            }
            QPushButton#btn_multiband:hover {
                background-color: #fbc531;
            }
            QPushButton#btn_multiband:pressed {
                background-color: #b8860b;
            }
            QComboBox {
                border: 1px solid #dcdde1;
                border-radius: 4px;
                padding: 5px 10px;
                background-color: white;
            }
            QComboBox::drop-down {
                border: none;
            }
            QListWidget {
                border: 1px solid #dcdde1;
                border-radius: 4px;
                background-color: #f8f9fa;
            }
            QTextEdit {
                border: 1px solid #dcdde1;
                border-radius: 4px;
                background-color: #2f3542;
                color: #f1f2f6;
                font-family: "Consolas", monospace;
                font-size: 12px;
            }
            QProgressBar {
                border: 1px solid #dcdde1;
                border-radius: 4px;
                text-align: center;
                background-color: #f8f9fa;
            }
            QProgressBar::chunk {
                background-color: #00a8ff;
                border-radius: 4px;
            }
            QLabel#label_preview_title {
                font-size: 16px;
                font-weight: bold;
                color: #2f3640;
            }
        """)
        
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        
        # 整体水平布局
        self.main_layout = QHBoxLayout(self.centralwidget)
        self.main_layout.setSpacing(15)
        self.main_layout.setObjectName(u"main_layout")
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        
        # 使用 QSplitter 实现可拖动分割
        self.splitter = QSplitter(Qt.Horizontal, self.centralwidget)
        self.splitter.setObjectName(u"splitter")
        
        # ----------------- 左侧控制面板 -----------------
        self.left_panel = QFrame(self.splitter)
        self.left_panel.setObjectName(u"left_panel")
        self.left_panel_layout = QVBoxLayout(self.left_panel)
        self.left_panel_layout.setSpacing(10)
        self.left_panel_layout.setObjectName(u"left_panel_layout")
        self.left_panel_layout.setContentsMargins(12, 12, 12, 12)
        
        # 标题标签
        self.label_title = QLabel(self.left_panel)
        self.label_title.setObjectName(u"label_title")
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.label_title.setFont(font)
        self.label_title.setText(u"全景图像拼接系统")
        self.left_panel_layout.addWidget(self.label_title)
        
        # 按钮组合布局 (选择图片、选择文件夹、清空列表)
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setObjectName(u"buttons_layout")
        
        self.btn_select_images = QPushButton(self.left_panel)
        self.btn_select_images.setObjectName(u"btn_select_images")
        self.btn_select_images.setText(u"选择图片")
        self.buttons_layout.addWidget(self.btn_select_images)
        
        self.btn_select_folder = QPushButton(self.left_panel)
        self.btn_select_folder.setObjectName(u"btn_select_folder")
        self.btn_select_folder.setText(u"选择文件夹")
        self.buttons_layout.addWidget(self.btn_select_folder)
        
        self.btn_clear = QPushButton(self.left_panel)
        self.btn_clear.setObjectName(u"btn_clear")
        self.btn_clear.setText(u"清空列表")
        self.buttons_layout.addWidget(self.btn_clear)
        
        self.left_panel_layout.addLayout(self.buttons_layout)
        
        # 列表标签
        self.label_list = QLabel(self.left_panel)
        self.label_list.setObjectName(u"label_list")
        self.label_list.setText(u"待拼接图片列表:")
        self.left_panel_layout.addWidget(self.label_list)
        
        # 图片列表展示
        self.listWidget_images = QListWidget(self.left_panel)
        self.listWidget_images.setObjectName(u"listWidget_images")
        self.left_panel_layout.addWidget(self.listWidget_images)
        
        # 🌟 智能全自动融合按钮 (第一行，占满一整行)
        self.btn_smart_probe = QPushButton(self.left_panel)
        self.btn_smart_probe.setObjectName(u"btn_smart_probe")
        self.btn_smart_probe.setText(u"🌟 智能全自动融合")
        self.left_panel_layout.addWidget(self.btn_smart_probe)

        # 线性融合 与 多频段融合 按钮水平布局 (第二行)
        self.manual_blend_layout = QHBoxLayout()
        self.manual_blend_layout.setObjectName(u"manual_blend_layout")

        self.btn_linear = QPushButton(self.left_panel)
        self.btn_linear.setObjectName(u"btn_linear")
        self.btn_linear.setText(u"线性融合拼接")
        self.manual_blend_layout.addWidget(self.btn_linear)

        self.btn_multiband = QPushButton(self.left_panel)
        self.btn_multiband.setObjectName(u"btn_multiband")
        self.btn_multiband.setText(u"多频段融合拼接")
        self.manual_blend_layout.addWidget(self.btn_multiband)

        self.left_panel_layout.addLayout(self.manual_blend_layout)
        
        # 进度条
        self.progressBar = QProgressBar(self.left_panel)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setValue(0)
        self.left_panel_layout.addWidget(self.progressBar)
        
        # 日志终端输出
        self.label_log = QLabel(self.left_panel)
        self.label_log.setObjectName(u"label_log")
        self.label_log.setText(u"实时运行日志:")
        self.left_panel_layout.addWidget(self.label_log)
        
        self.textEdit_log = QTextEdit(self.left_panel)
        self.textEdit_log.setObjectName(u"textEdit_log")
        self.textEdit_log.setReadOnly(True)
        self.left_panel_layout.addWidget(self.textEdit_log)
        
        # ----------------- 右侧预览面板 -----------------
        self.right_panel = QFrame(self.splitter)
        self.right_panel.setObjectName(u"right_panel")
        self.right_panel.setStyleSheet(u"background-color: #2f3542; border-radius: 8px;")
        self.right_panel_layout = QVBoxLayout(self.right_panel)
        self.right_panel_layout.setSpacing(10)
        self.right_panel_layout.setObjectName(u"right_panel_layout")
        self.right_panel_layout.setContentsMargins(12, 12, 12, 12)
        
        # 预览区域顶部标题
        self.label_preview_title = QLabel(self.right_panel)
        self.label_preview_title.setObjectName(u"label_preview_title")
        self.label_preview_title.setStyleSheet(u"color: white; font-size: 15px; font-weight: bold;")
        self.label_preview_title.setText(u"全景拼接预览区域")
        self.right_panel_layout.addWidget(self.label_preview_title)
        
        # 预览主体 QLabel
        self.label_preview = QLabel(self.right_panel)
        self.label_preview.setObjectName(u"label_preview")
        self.label_preview.setAlignment(Qt.AlignCenter)
        self.label_preview.setStyleSheet(u"border: 2px dashed #718093; border-radius: 6px; color: #a4b0be; background-color: #1e272e;")
        self.label_preview.setText(u"暂无全景图预览\n\n请在左侧添加图片并点击“开始拼接”")
        self.right_panel_layout.addWidget(self.label_preview)
        
        # 设定 splitter 初始比例 (左侧占比 40%，右侧占比 60%)
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.right_panel)
        self.splitter.setSizes([440, 660])
        
        self.main_layout.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)
        
        QMetaObject.connectSlotsByName(MainWindow)
