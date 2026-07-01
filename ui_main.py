from PySide6.QtCore import QSize, Qt, QMetaObject
from PySide6.QtGui import QFont, QIcon, QPixmap, QImage
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
    QListWidget, QMainWindow, QProgressBar, QPushButton,
    QSplitter, QTextEdit, QVBoxLayout, QWidget, QFrame, QTabWidget, QSlider, QTableWidget, QHeaderView)
from custom_widgets import InteractiveImageViewer

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1300, 850)
        MainWindow.setMinimumSize(QSize(1000, 650))
        
        MainWindow.setStyleSheet(u"""
            QMainWindow { background-color: #f5f6fa; }
            QWidget { font-family: "Microsoft YaHei", "Segoe UI", sans-serif; font-size: 13px; color: #2f3640; }
            QFrame#left_panel, QFrame#right_panel { background-color: #ffffff; border-radius: 8px; border: 1px solid #dcdde1; }
            QPushButton { background-color: #353b48; color: white; border: none; padding: 8px 15px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #718093; }
            QPushButton:pressed { background-color: #2f3640; }
            QPushButton#btn_start { background-color: #44bd32; }
            QPushButton#btn_start:hover { background-color: #4cd137; }
            QPushButton#btn_clear, QPushButton#btn_clear_history { background-color: #e84118; }
            QPushButton#btn_clear:hover, QPushButton#btn_clear_history:hover { background-color: #c23616; }
            QPushButton#btn_smart_probe { background-color: #6c5ce7; font-size: 14px; padding: 10px 15px; }
            QPushButton#btn_smart_probe:hover { background-color: #8278fc; }
            QComboBox { border: 1px solid #dcdde1; border-radius: 4px; padding: 5px 10px; background-color: white; }
            QComboBox::drop-down { border: none; }
            QListWidget { border: 1px solid #dcdde1; border-radius: 4px; background-color: #f8f9fa; }
            QTextEdit { border: 1px solid #dcdde1; border-radius: 4px; background-color: #2f3542; color: #f1f2f6; font-family: "Consolas", monospace; font-size: 12px; }
            QProgressBar { border: 1px solid #dcdde1; border-radius: 4px; text-align: center; background-color: #f8f9fa; }
            QProgressBar::chunk { background-color: #00a8ff; border-radius: 4px; }
            QTabWidget::pane { border: 1px solid #dcdde1; border-radius: 4px; background: white; }
            QTabBar::tab { background: #f5f6fa; border: 1px solid #dcdde1; border-bottom: none; padding: 8px 15px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: white; font-weight: bold; border-top: 2px solid #6c5ce7; }
            QTableWidget { background-color: white; color: #2f3640; gridline-color: #dcdde1; border: 1px solid #dcdde1; border-radius: 4px; }
            QHeaderView::section { background-color: #f5f6fa; color: #2f3640; padding: 6px; border: 1px solid #dcdde1; font-weight: bold; }
        """)
        
        self.centralwidget = QWidget(MainWindow)
        
        self.main_layout = QVBoxLayout(self.centralwidget)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        
        # === Top Header Bar ===
        self.header_layout = QHBoxLayout()
        self.label_title = QLabel(u"🔗 多图无缝全景拼接系统")
        self.label_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        
        self.label_stats = QLabel(u"统计: 已拼接 0 次 | 成功率 0% | 平均耗时 0.0s")
        self.label_stats.setStyleSheet("font-size: 14px; color: #7f8c8d; font-weight: bold;")
        self.label_stats.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        self.header_layout.addWidget(self.label_title)
        self.header_layout.addStretch()
        self.header_layout.addWidget(self.label_stats)
        self.main_layout.addLayout(self.header_layout)
        
        # === Middle Splitter (Left, Center, Right) ===
        self.middle_splitter = QSplitter(Qt.Horizontal, self.centralwidget)
        
        # --- Left Panel (20%) ---
        self.left_panel = QFrame(self.middle_splitter)
        self.left_panel.setObjectName(u"left_panel")
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(12, 12, 12, 12)
        
        self.buttons_layout = QHBoxLayout()
        self.btn_select_images = QPushButton(u"选择图片")
        self.btn_select_folder = QPushButton(u"选择文件夹")
        self.btn_clear = QPushButton(u"清空")
        self.btn_clear.setObjectName(u"btn_clear")
        self.buttons_layout.addWidget(self.btn_select_images)
        self.buttons_layout.addWidget(self.btn_select_folder)
        self.buttons_layout.addWidget(self.btn_clear)
        self.left_layout.addLayout(self.buttons_layout)
        
        self.label_list = QLabel(u"待拼接图片列表:")
        self.left_layout.addWidget(self.label_list)
        self.listWidget_images = QListWidget()
        self.left_layout.addWidget(self.listWidget_images)
        
        self.btn_start_street = QPushButton(u"🚗 水平街道/航拍全景（极清防重影）")
        self.btn_start_street.setObjectName(u"btn_start_street")
        self.btn_start_street.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; font-size: 13px; padding: 10px; border-radius: 4px;")
        self.btn_start_street.setToolTip(u"提示：适合无人机航拍、街景外立面、长幅广告牌等远景拍摄。极清防重影。")
        self.left_layout.addWidget(self.btn_start_street)
        
        self.btn_start_desktop = QPushButton(u"🏠 室内/桌面近景（宽带防断层）")
        self.btn_start_desktop.setObjectName(u"btn_start_desktop")
        self.btn_start_desktop.setStyleSheet("background-color: #e67e22; color: white; font-weight: bold; font-size: 13px; padding: 10px; border-radius: 4px;")
        self.btn_start_desktop.setToolTip(u"提示：适合桌面、室内房间、近距离带角度拍摄。宽带柔和过渡，防物体断裂。")
        self.left_layout.addWidget(self.btn_start_desktop)
        
        self.btn_history = QPushButton(u"📜 历史档案库")
        self.btn_history.setObjectName(u"btn_history")
        self.btn_history.setStyleSheet("background-color: #9b59b6; font-size: 14px; padding: 10px;")
        self.left_layout.addWidget(self.btn_history)
        
        self.progressBar = QProgressBar()
        self.progressBar.setValue(0)
        self.left_layout.addWidget(self.progressBar)
        
        # --- Center Panel (55%) ---
        self.center_panel = QFrame(self.middle_splitter)
        self.center_layout = QVBoxLayout(self.center_panel)
        self.center_layout.setContentsMargins(0, 0, 0, 0)
        self.center_splitter = QSplitter(Qt.Vertical, self.center_panel)
        
        self.preview_frame = QFrame(self.center_splitter)
        self.preview_layout = QVBoxLayout(self.preview_frame)
        self.preview_layout.setContentsMargins(0, 0, 0, 0)
        
        self.label_preview_title = QLabel(u"全景拼接预览区域")
        self.label_preview_title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        self.preview_layout.addWidget(self.label_preview_title)
        
        # 交互式看图组件（替代静态 QLabel）
        self.image_viewer = InteractiveImageViewer()
        self.preview_layout.addWidget(self.image_viewer, 1)
        
        self.log_frame = QFrame(self.center_splitter)
        self.log_layout = QVBoxLayout(self.log_frame)
        self.label_log_title = QLabel(u"实时运行日志")
        self.label_log_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.textEdit_log = QTextEdit()
        self.textEdit_log.setObjectName(u"textEdit_log")
        self.textEdit_log.setReadOnly(True)
        self.log_layout.addWidget(self.label_log_title)
        self.log_layout.addWidget(self.textEdit_log)
        
        self.center_splitter.setSizes([500, 200])
        self.center_layout.addWidget(self.center_splitter)
        
        # --- Right Panel (25%) ---
        self.right_panel = QFrame(self.middle_splitter)
        self.right_panel.setObjectName(u"right_panel")
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(12, 12, 12, 12)
        self.right_layout.setSpacing(15)
        
        self.label_config_title = QLabel(u"⚙️ 参数配置 (全局渗透)")
        self.label_config_title.setStyleSheet("font-size: 15px; font-weight: bold;")
        self.right_layout.addWidget(self.label_config_title)
        
        self.label_blend = QLabel(u"融合模式:")
        self.combo_blend_mode = QComboBox()
        self.combo_blend_mode.setObjectName(u"combo_blend_mode")
        self.combo_blend_mode.addItems([u"线性融合 (Linear)", u"多频段融合 (Multi-band)"])
        self.right_layout.addWidget(self.label_blend)
        self.right_layout.addWidget(self.combo_blend_mode)
        
        # ORB
        self.orb_layout = QHBoxLayout()
        self.label_orb = QLabel(u"ORB特征点数:")
        self.label_orb_val = QLabel(u"3000")
        self.label_orb_val.setObjectName(u"label_orb_val")
        self.orb_layout.addWidget(self.label_orb)
        self.orb_layout.addStretch()
        self.orb_layout.addWidget(self.label_orb_val)
        self.right_layout.addLayout(self.orb_layout)
        self.slider_orb = QSlider(Qt.Horizontal)
        self.slider_orb.setObjectName(u"slider_orb")
        self.slider_orb.setRange(1000, 5000)
        self.slider_orb.setSingleStep(500)
        self.slider_orb.setValue(3000)
        self.right_layout.addWidget(self.slider_orb)
        
        # RANSAC
        self.ransac_layout = QHBoxLayout()
        self.label_ransac = QLabel(u"RANSAC阈值:")
        self.label_ransac_val = QLabel(u"5.0")
        self.label_ransac_val.setObjectName(u"label_ransac_val")
        self.ransac_layout.addWidget(self.label_ransac)
        self.ransac_layout.addStretch()
        self.ransac_layout.addWidget(self.label_ransac_val)
        self.right_layout.addLayout(self.ransac_layout)
        self.slider_ransac = QSlider(Qt.Horizontal)
        self.slider_ransac.setObjectName(u"slider_ransac")
        self.slider_ransac.setRange(10, 80)
        self.slider_ransac.setSingleStep(5)
        self.slider_ransac.setValue(50)
        self.right_layout.addWidget(self.slider_ransac)
        
        # Quality
        self.quality_layout = QHBoxLayout()
        self.label_quality = QLabel(u"输出质量 (JPEG):")
        self.label_quality_val = QLabel(u"95")
        self.label_quality_val.setObjectName(u"label_quality_val")
        self.quality_layout.addWidget(self.label_quality)
        self.quality_layout.addStretch()
        self.quality_layout.addWidget(self.label_quality_val)
        self.right_layout.addLayout(self.quality_layout)
        self.slider_quality = QSlider(Qt.Horizontal)
        self.slider_quality.setObjectName(u"slider_quality")
        self.slider_quality.setRange(1, 100)
        self.slider_quality.setValue(95)
        self.right_layout.addWidget(self.slider_quality)
        
        self.right_layout.addStretch()
        
        self.middle_splitter.setSizes([260, 715, 325])
        self.main_layout.addWidget(self.middle_splitter, 3) # stretch factor 3
        
        # 移除底部的 tabWidget，因为历史记录已独立为弹窗
        
        MainWindow.setCentralWidget(self.centralwidget)
        QMetaObject.connectSlotsByName(MainWindow)
