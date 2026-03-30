"""
============================================================================
PhysicRain - Apple Style GUI
============================================================================
基于 PyQt5 的 iOS/macOS 风格图形界面
============================================================================
"""

import sys
import os
import glob
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QDoubleSpinBox, QSpinBox, QComboBox,
    QPushButton, QFileDialog, QLineEdit, QStackedWidget,
    QGraphicsDropShadowEffect, QSizePolicy, QProgressBar,
    QScrollArea, QFrame, QGridLayout, QSpacerItem
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QPropertyAnimation, QRect,
    QEasingCurve, pyqtProperty, QSize, QPoint, QTimer, QRectF
)
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QIcon, QPalette, QColor,
    QPainter, QPainterPath, QBrush, QPen, QLinearGradient,
    QRadialGradient, QFontDatabase
)

from rain_renderer import render_rain_mask


# ===========================================================================
# Apple 配色常量
# ===========================================================================

class AppleColors:
    BG            = '#1c1c1e'
    BG_SECONDARY  = '#2c2c2e'
    BG_TERTIARY   = '#3a3a3c'
    CARD          = '#2c2c2e'
    CARD_HOVER    = '#363638'
    ACCENT        = '#0a84ff'
    ACCENT_DARK   = '#0066cc'
    GREEN         = '#30d158'
    RED           = '#ff453a'
    ORANGE        = '#ff9f0a'
    YELLOW        = '#ffd60a'
    TEXT          = '#ffffff'
    TEXT_SEC      = '#8e8e93'
    TEXT_TER      = '#636366'
    SEPARATOR     = 'rgba(84, 84, 88, 0.65)'
    TOGGLE_BG     = '#39393d'
    TOGGLE_ON     = '#30d158'
    SIDEBAR_BG    = '#1c1c1e'
    SIDEBAR_SEL   = 'rgba(10, 132, 255, 0.22)'
    TITLE_BAR     = '#161618'


# ===========================================================================
# 预设场景
# ===========================================================================

PRESETS = {
    '自定义': None,
    '毛毛雨': {
        'rain_rate': 2.0, 'wind_speed': 0.5, 'wind_direction': 315.0,
        'exposure_time': 1/30, 'turbulence_deg': 1.0,
    },
    '春雨': {
        'rain_rate': 10.0, 'wind_speed': 1.5, 'wind_direction': 350.0,
        'exposure_time': 1/30, 'turbulence_deg': 2.0,
    },
    '大雨': {
        'rain_rate': 35.0, 'wind_speed': 4.0, 'wind_direction': 330.0,
        'exposure_time': 1/30, 'turbulence_deg': 3.5,
    },
    '暴雨': {
        'rain_rate': 70.0, 'wind_speed': 8.0, 'wind_direction': 315.0,
        'exposure_time': 1/25, 'turbulence_deg': 6.0,
    },
    '台风': {
        'rain_rate': 120.0, 'wind_speed': 15.0, 'wind_direction': 270.0,
        'exposure_time': 1/20, 'turbulence_deg': 10.0,
    },
}


# ===========================================================================
# 渲染线程
# ===========================================================================

class RenderThread(QThread):
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            mask = render_rain_mask(**self.params)
            self.finished.emit(mask)
        except Exception as e:
            self.error.emit(str(e))


# ===========================================================================
# iOS Toggle Switch
# ===========================================================================

class AppleToggle(QWidget):
    toggled = pyqtSignal(bool)

    def __init__(self, checked=False, parent=None):
        super().__init__(parent)
        self.setFixedSize(51, 31)
        self.setCursor(Qt.PointingHandCursor)
        self._checked = checked
        self._circle_pos = 25.0 if checked else 4.0
        self._anim = QPropertyAnimation(self, b'circle_pos')
        self._anim.setDuration(200)
        self._anim.setEasingCurve(QEasingCurve.InOutCubic)

    def get_circle_pos(self):
        return self._circle_pos

    def set_circle_pos(self, pos):
        self._circle_pos = pos
        self.update()

    circle_pos = pyqtProperty(float, get_circle_pos, set_circle_pos)

    def isChecked(self):
        return self._checked

    def setChecked(self, checked):
        if self._checked != checked:
            self._checked = checked
            target = 25.0 if checked else 4.0
            self._anim.setStartValue(self._circle_pos)
            self._anim.setEndValue(target)
            self._anim.start()
            self.toggled.emit(checked)

    def mousePressEvent(self, e):
        self.setChecked(not self._checked)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # 背景胶囊
        bg_color = QColor(AppleColors.TOGGLE_ON) if self._checked else QColor(AppleColors.TOGGLE_BG)
        p.setBrush(QBrush(bg_color))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(0, 0, 51, 31, 15.5, 15.5)

        # 白色圆形滑块
        p.setBrush(QBrush(QColor('#ffffff')))
        shadow_pen = QPen(QColor(0, 0, 0, 40))
        shadow_pen.setWidth(1)
        p.setPen(shadow_pen)
        p.drawEllipse(QRectF(self._circle_pos, 2.5, 26, 26))
        p.end()


# ===========================================================================
# Apple 风格卡片
# ===========================================================================

class AppleCard(QFrame):
    def __init__(self, title='', parent=None):
        super().__init__(parent)
        self.setObjectName('appleCard')

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        if title:
            title_label = QLabel(title)
            title_label.setStyleSheet(f'''
                color: {AppleColors.TEXT_SEC};
                font-size: 13px;
                font-weight: 600;
                text-transform: uppercase;
                padding: 8px 16px 6px 16px;
                background: transparent;
            ''')
            self._layout.addWidget(title_label)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        self._layout.addWidget(self._content)

        self.setStyleSheet(f'''
            #appleCard {{
                background-color: {AppleColors.CARD};
                border-radius: 12px;
                border: none;
            }}
        ''')

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

    def addRow(self, widget):
        if self._content_layout.count() > 0:
            sep = QFrame()
            sep.setFixedHeight(1)
            sep.setStyleSheet(f'background-color: {AppleColors.SEPARATOR}; margin-left: 16px;')
            self._content_layout.addWidget(sep)
        self._content_layout.addWidget(widget)


# ===========================================================================
# Apple 风格参数行 (滑块)
# ===========================================================================

class AppleSliderRow(QWidget):
    def __init__(self, label, min_val, max_val, default, step=1.0,
                 decimals=1, suffix='', parent=None):
        super().__init__(parent)
        self.setFixedHeight(58)
        self._scale = 10 ** decimals

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(2)

        top = QHBoxLayout()
        top.setSpacing(0)

        self.label = QLabel(label)
        self.label.setStyleSheet(f'color: {AppleColors.TEXT}; font-size: 15px;')
        top.addWidget(self.label)

        top.addStretch()

        self.value_label = QLabel()
        self.value_label.setStyleSheet(f'color: {AppleColors.TEXT_SEC}; font-size: 15px;')
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top.addWidget(self.value_label)

        layout.addLayout(top)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(min_val * self._scale))
        self.slider.setMaximum(int(max_val * self._scale))
        self.slider.setValue(int(default * self._scale))
        self.slider.setSingleStep(max(1, int(step * self._scale)))
        self.slider.setFixedHeight(22)
        layout.addWidget(self.slider)

        self._suffix = suffix
        self._decimals = decimals
        self._update_label(int(default * self._scale))
        self.slider.valueChanged.connect(self._update_label)

    def _update_label(self, val):
        real = val / self._scale
        if self._decimals == 0:
            self.value_label.setText(f'{int(real)}{self._suffix}')
        else:
            self.value_label.setText(f'{real:.{self._decimals}f}{self._suffix}')

    def value(self):
        return self.slider.value() / self._scale

    def setValue(self, val):
        self.slider.setValue(int(val * self._scale))

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self.slider.setEnabled(enabled)
        opacity = 1.0 if enabled else 0.4
        self.label.setStyleSheet(
            f'color: {AppleColors.TEXT}; font-size: 15px; opacity: {opacity};'
        )


# ===========================================================================
# Apple 风格按钮
# ===========================================================================

class AppleButton(QPushButton):
    def __init__(self, text, primary=True, parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(44)
        self.setCursor(Qt.PointingHandCursor)
        self.setFont(QFont('Segoe UI', 15, QFont.DemiBold))

        if primary:
            self.setStyleSheet(f'''
                QPushButton {{
                    background-color: {AppleColors.ACCENT};
                    color: white;
                    border: none;
                    border-radius: 12px;
                    padding: 0 24px;
                    font-size: 15px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background-color: {AppleColors.ACCENT_DARK};
                }}
                QPushButton:pressed {{
                    background-color: #004999;
                }}
                QPushButton:disabled {{
                    background-color: {AppleColors.BG_TERTIARY};
                    color: {AppleColors.TEXT_TER};
                }}
            ''')
        else:
            self.setStyleSheet(f'''
                QPushButton {{
                    background-color: {AppleColors.BG_TERTIARY};
                    color: {AppleColors.ACCENT};
                    border: none;
                    border-radius: 12px;
                    padding: 0 24px;
                    font-size: 15px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background-color: #48484a;
                }}
                QPushButton:pressed {{
                    background-color: {AppleColors.BG_SECONDARY};
                }}
                QPushButton:disabled {{
                    background-color: {AppleColors.BG_TERTIARY};
                    color: {AppleColors.TEXT_TER};
                }}
            ''')


# ===========================================================================
# 侧边栏导航
# ===========================================================================

class SidebarItem(QPushButton):
    def __init__(self, text, icon_char='', parent=None):
        super().__init__(parent)
        display = f'  {icon_char}  {text}' if icon_char else f'  {text}'
        self.setText(display)
        self.setFixedHeight(36)
        self.setCursor(Qt.PointingHandCursor)
        self.setCheckable(True)
        self._is_selected = False
        self._update_style()

    def set_selected(self, selected):
        self._is_selected = selected
        self.setChecked(selected)
        self._update_style()

    def _update_style(self):
        if self._is_selected:
            self.setStyleSheet(f'''
                QPushButton {{
                    background-color: {AppleColors.SIDEBAR_SEL};
                    color: {AppleColors.ACCENT};
                    border: none;
                    border-radius: 8px;
                    text-align: left;
                    padding-left: 8px;
                    font-size: 14px;
                    font-weight: 600;
                }}
            ''')
        else:
            self.setStyleSheet(f'''
                QPushButton {{
                    background-color: transparent;
                    color: {AppleColors.TEXT_SEC};
                    border: none;
                    border-radius: 8px;
                    text-align: left;
                    padding-left: 8px;
                    font-size: 14px;
                    font-weight: 400;
                }}
                QPushButton:hover {{
                    background-color: rgba(255, 255, 255, 0.06);
                    color: {AppleColors.TEXT};
                }}
            ''')


class AppleSidebar(QWidget):
    page_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(200)
        self.setStyleSheet(f'background-color: {AppleColors.SIDEBAR_BG};')

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(2)

        self.items = []
        nav_items = [
            ('渲染参数', '\u2601'),     # cloud
            ('深度图',   '\u25a6'),     # square
            ('相机',     '\u25ce'),     # bullseye
            ('输出设置', '\u2b1c'),     # square
            ('批量处理', '\u25a4'),     # squares
        ]

        for i, (text, icon) in enumerate(nav_items):
            item = SidebarItem(text, icon)
            item.clicked.connect(lambda checked, idx=i: self._on_click(idx))
            layout.addWidget(item)
            self.items.append(item)

        layout.addStretch()

        # 预设选择
        preset_label = QLabel('  预设场景')
        preset_label.setStyleSheet(f'''
            color: {AppleColors.TEXT_TER};
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            padding-top: 12px;
        ''')
        layout.addWidget(preset_label)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PRESETS.keys())
        self.preset_combo.setStyleSheet(f'''
            QComboBox {{
                background-color: {AppleColors.BG_TERTIARY};
                color: {AppleColors.TEXT};
                border: none;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 13px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {AppleColors.BG_SECONDARY};
                color: {AppleColors.TEXT};
                border: 1px solid {AppleColors.BG_TERTIARY};
                border-radius: 8px;
                selection-background-color: {AppleColors.ACCENT};
            }}
        ''')
        layout.addWidget(self.preset_combo)

        self.items[0].set_selected(True)

    def _on_click(self, idx):
        for i, item in enumerate(self.items):
            item.set_selected(i == idx)
        self.page_changed.emit(idx)


# ===========================================================================
# 自定义标题栏
# ===========================================================================

class AppleTitleBar(QWidget):
    def __init__(self, parent_window, parent=None):
        super().__init__(parent)
        self._window = parent_window
        self.setFixedHeight(38)
        self.setStyleSheet(f'background-color: {AppleColors.TITLE_BAR};')
        self._drag_pos = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(8)

        # macOS 红黄绿三圆点
        for color, action in [('#ff5f57', 'close'), ('#febc2e', 'min'), ('#28c840', 'max')]:
            btn = QPushButton()
            btn.setFixedSize(14, 14)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(f'''
                QPushButton {{
                    background-color: {color};
                    border: none;
                    border-radius: 7px;
                }}
                QPushButton:hover {{
                    background-color: {color};
                    border: 1px solid rgba(0,0,0,0.2);
                }}
            ''')
            if action == 'close':
                btn.clicked.connect(self._window.close)
            elif action == 'min':
                btn.clicked.connect(self._window.showMinimized)
            elif action == 'max':
                btn.clicked.connect(self._toggle_max)
            layout.addWidget(btn)

        layout.addSpacing(20)

        title = QLabel('PhysicRain')
        title.setStyleSheet(f'''
            color: {AppleColors.TEXT_SEC};
            font-size: 13px;
            font-weight: 600;
        ''')
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title, stretch=1)

        layout.addSpacing(62)

    def _toggle_max(self):
        if self._window.isMaximized():
            self._window.showNormal()
        else:
            self._window.showMaximized()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPos() - self._window.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag_pos and e.buttons() == Qt.LeftButton:
            self._window.move(e.globalPos() - self._drag_pos)

    def mouseReleaseEvent(self, e):
        self._drag_pos = None

    def mouseDoubleClickEvent(self, e):
        self._toggle_max()


# ===========================================================================
# 通知条 (InfoBar)
# ===========================================================================

class InfoBar(QWidget):
    def __init__(self, text, bar_type='info', parent=None):
        super().__init__(parent)
        self.setFixedHeight(44)

        colors = {
            'info':    AppleColors.ACCENT,
            'success': AppleColors.GREEN,
            'error':   AppleColors.RED,
            'warning': AppleColors.ORANGE,
        }
        color = colors.get(bar_type, AppleColors.ACCENT)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)

        dot = QLabel('\u25cf')
        dot.setStyleSheet(f'color: {color}; font-size: 10px;')
        dot.setFixedWidth(16)
        layout.addWidget(dot)

        label = QLabel(text)
        label.setStyleSheet(f'color: {AppleColors.TEXT}; font-size: 13px;')
        layout.addWidget(label, stretch=1)

        self.setStyleSheet(f'''
            background-color: {AppleColors.CARD};
            border-radius: 12px;
            border: 1px solid {color};
        ''')

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

    def show_toast(self, duration=3000):
        self.show()
        QTimer.singleShot(duration, self.deleteLater)


# ===========================================================================
# 页面：渲染参数
# ===========================================================================

class RenderParamsPage(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet('QScrollArea { border: none; background: transparent; }')

        container = QWidget()
        container.setStyleSheet('background: transparent;')
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # 标题
        title = QLabel('渲染参数')
        title.setStyleSheet(f'color: {AppleColors.TEXT}; font-size: 28px; font-weight: bold;')
        layout.addWidget(title)

        subtitle = QLabel('调整物理参数控制雨滴效果')
        subtitle.setStyleSheet(f'color: {AppleColors.TEXT_SEC}; font-size: 14px;')
        layout.addWidget(subtitle)

        layout.addSpacing(4)

        # 降雨参数卡片
        rain_card = AppleCard('降雨')
        self.rain_rate = AppleSliderRow('降雨量', 0.5, 200.0, 25.0, 0.5, 1, ' mm/h')
        self.wind_speed = AppleSliderRow('风速', 0.0, 30.0, 2.0, 0.1, 1, ' m/s')

        # 风向：罗盘下拉选择
        self.wind_dir_widget = QWidget()
        self.wind_dir_widget.setFixedHeight(52)
        wd_layout = QHBoxLayout(self.wind_dir_widget)
        wd_layout.setContentsMargins(16, 0, 16, 0)
        wd_label = QLabel('风向')
        wd_label.setStyleSheet(f'color: {AppleColors.TEXT}; font-size: 15px;')
        wd_layout.addWidget(wd_label)
        wd_layout.addStretch()
        self.wind_dir_combo = QComboBox()
        self._compass_values = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
                                180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]
        compass_labels = [
            'N  北风(迎面)', 'NNE 北偏东22°', 'NE 东北风', 'ENE 东偏北67°',
            'E  东风(右侧)', 'ESE 东偏南112°', 'SE 东南风', 'SSE 南偏东157°',
            'S  南风(顺风)', 'SSW 南偏西202°', 'SW 西南风', 'WSW 西偏南247°',
            'W  西风(左侧)', 'WNW 西偏北292°', 'NW 西北风', 'NNW 北偏西337°',
        ]
        self.wind_dir_combo.addItems(compass_labels)
        self.wind_dir_combo.setCurrentIndex(14)  # NW
        self.wind_dir_combo.setFixedWidth(175)
        self.wind_dir_combo.setStyleSheet(f'''
            QComboBox {{
                background-color: {AppleColors.BG_TERTIARY};
                color: {AppleColors.TEXT};
                border: none; border-radius: 8px;
                padding: 4px 10px; font-size: 13px;
            }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox QAbstractItemView {{
                background-color: {AppleColors.BG_SECONDARY};
                color: {AppleColors.TEXT};
                border: 1px solid {AppleColors.BG_TERTIARY};
                selection-background-color: {AppleColors.ACCENT};
            }}
        ''')
        wd_layout.addWidget(self.wind_dir_combo)

        # 风向微调 (0-360°)
        self.wind_dir_fine = AppleSliderRow('微调', 0, 360, 315, 1, 0, '°')
        self.wind_dir_combo.currentIndexChanged.connect(self._on_wind_combo)

        rain_card.addRow(self.rain_rate)
        rain_card.addRow(self.wind_speed)
        rain_card.addRow(self.wind_dir_widget)
        rain_card.addRow(self.wind_dir_fine)
        layout.addWidget(rain_card)

        # 快门与湍流卡片
        motion_card = AppleCard('运动')
        self.exposure = AppleSliderRow('快门时间', 0.001, 0.2, 1/30, 0.001, 3, ' s')
        self.turbulence = AppleSliderRow('湍流强度', 0.0, 15.0, 2.5, 0.1, 1, ' °')
        motion_card.addRow(self.exposure)
        motion_card.addRow(self.turbulence)
        layout.addWidget(motion_card)

        # 光学参数卡片
        optics_card = AppleCard('光学')
        self.focus_dist = AppleSliderRow('对焦距离', 1.0, 100.0, 12.0, 0.5, 1, ' m')
        self.depth_scale = AppleSliderRow('场景深度', 10.0, 500.0, 100.0, 1.0, 0, ' m')
        optics_card.addRow(self.focus_dist)
        optics_card.addRow(self.depth_scale)
        layout.addWidget(optics_card)

        # 预览 + 按钮
        layout.addSpacing(8)

        # 预览区
        self.preview_card = QFrame()
        self.preview_card.setStyleSheet(f'''
            background-color: {AppleColors.CARD};
            border-radius: 12px;
        ''')
        preview_shadow = QGraphicsDropShadowEffect()
        preview_shadow.setBlurRadius(20)
        preview_shadow.setColor(QColor(0, 0, 0, 60))
        preview_shadow.setOffset(0, 2)
        self.preview_card.setGraphicsEffect(preview_shadow)

        preview_layout = QVBoxLayout(self.preview_card)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.setSpacing(8)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(300, 300)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setStyleSheet(f'''
            background-color: #111113;
            border-radius: 8px;
            color: {AppleColors.TEXT_TER};
            font-size: 14px;
        ''')
        self.preview_label.setText('点击「渲染」生成预览')
        preview_layout.addWidget(self.preview_label, stretch=1)

        self.stats_label = QLabel('')
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setStyleSheet(f'color: {AppleColors.TEXT_TER}; font-size: 12px;')
        preview_layout.addWidget(self.stats_label)

        layout.addWidget(self.preview_card, stretch=1)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        self.render_btn = AppleButton('渲染', primary=True)
        self.save_btn = AppleButton('保存', primary=False)
        self.save_btn.setEnabled(False)
        btn_layout.addWidget(self.render_btn)
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)

        self.setWidget(container)

    def _on_wind_combo(self, idx):
        """罗盘方向选择时同步更新微调滑块"""
        if 0 <= idx < len(self._compass_values):
            self.wind_dir_fine.setValue(self._compass_values[idx])


# ===========================================================================
# 页面：深度图
# ===========================================================================

class DepthMapPage(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet('QScrollArea { border: none; background: transparent; }')

        container = QWidget()
        container.setStyleSheet('background: transparent;')
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        title = QLabel('深度图')
        title.setStyleSheet(f'color: {AppleColors.TEXT}; font-size: 28px; font-weight: bold;')
        layout.addWidget(title)

        subtitle = QLabel('提供深度图可实现更真实的透视效果')
        subtitle.setStyleSheet(f'color: {AppleColors.TEXT_SEC}; font-size: 14px;')
        layout.addWidget(subtitle)

        layout.addSpacing(4)

        # 开关卡片
        toggle_card = AppleCard()
        toggle_row = QWidget()
        toggle_row.setFixedHeight(52)
        toggle_lay = QHBoxLayout(toggle_row)
        toggle_lay.setContentsMargins(16, 0, 16, 0)
        lbl = QLabel('使用自定义深度图')
        lbl.setStyleSheet(f'color: {AppleColors.TEXT}; font-size: 15px;')
        toggle_lay.addWidget(lbl)
        toggle_lay.addStretch()
        self.use_depth_toggle = AppleToggle(False)
        toggle_lay.addWidget(self.use_depth_toggle)
        toggle_card.addRow(toggle_row)
        layout.addWidget(toggle_card)

        # 文件选择卡片
        self.file_card = AppleCard('文件')
        file_row = QWidget()
        file_row.setFixedHeight(52)
        file_lay = QHBoxLayout(file_row)
        file_lay.setContentsMargins(16, 0, 16, 0)
        self.depth_path_edit = QLineEdit()
        self.depth_path_edit.setPlaceholderText('选择深度图文件...')
        self.depth_path_edit.setStyleSheet(f'''
            QLineEdit {{
                background: {AppleColors.BG_TERTIARY};
                border: none;
                border-radius: 8px;
                padding: 6px 12px;
                color: {AppleColors.TEXT};
                font-size: 13px;
            }}
        ''')
        file_lay.addWidget(self.depth_path_edit)
        self.browse_btn = AppleButton('浏览', primary=False)
        self.browse_btn.setFixedWidth(72)
        self.browse_btn.setFixedHeight(32)
        file_lay.addWidget(self.browse_btn)
        self.file_card.addRow(file_row)
        self.file_card.setEnabled(False)
        layout.addWidget(self.file_card)

        # 默认深度模式
        mode_card = AppleCard('默认深度模式')
        mode_row = QWidget()
        mode_row.setFixedHeight(52)
        mode_lay = QHBoxLayout(mode_row)
        mode_lay.setContentsMargins(16, 0, 16, 0)
        mode_lbl = QLabel('生成方式')
        mode_lbl.setStyleSheet(f'color: {AppleColors.TEXT}; font-size: 15px;')
        mode_lay.addWidget(mode_lbl)
        mode_lay.addStretch()
        self.depth_mode_combo = QComboBox()
        self.depth_mode_combo.addItems(['gradient', 'uniform', 'radial'])
        self.depth_mode_combo.setFixedWidth(140)
        self.depth_mode_combo.setStyleSheet(f'''
            QComboBox {{
                background-color: {AppleColors.BG_TERTIARY};
                color: {AppleColors.TEXT};
                border: none;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 13px;
            }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox QAbstractItemView {{
                background-color: {AppleColors.BG_SECONDARY};
                color: {AppleColors.TEXT};
                border: 1px solid {AppleColors.BG_TERTIARY};
                selection-background-color: {AppleColors.ACCENT};
            }}
        ''')
        mode_lay.addWidget(self.depth_mode_combo)
        mode_card.addRow(mode_row)
        layout.addWidget(mode_card)

        # 深度图预览
        self.depth_preview_label = QLabel()
        self.depth_preview_label.setFixedSize(240, 160)
        self.depth_preview_label.setAlignment(Qt.AlignCenter)
        self.depth_preview_label.setStyleSheet(f'''
            background-color: {AppleColors.CARD};
            border-radius: 12px;
            color: {AppleColors.TEXT_TER};
            font-size: 13px;
        ''')
        self.depth_preview_label.setText('深度图预览')
        layout.addWidget(self.depth_preview_label, alignment=Qt.AlignCenter)

        layout.addStretch()
        self.setWidget(container)


# ===========================================================================
# 页面：相机参数
# ===========================================================================

class CameraPage(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet('QScrollArea { border: none; background: transparent; }')

        container = QWidget()
        container.setStyleSheet('background: transparent;')
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        title = QLabel('相机参数')
        title.setStyleSheet(f'color: {AppleColors.TEXT}; font-size: 28px; font-weight: bold;')
        layout.addWidget(title)

        subtitle = QLabel('自定义相机内参与外参')
        subtitle.setStyleSheet(f'color: {AppleColors.TEXT_SEC}; font-size: 14px;')
        layout.addWidget(subtitle)

        layout.addSpacing(4)

        toggle_card = AppleCard()
        toggle_row = QWidget()
        toggle_row.setFixedHeight(52)
        toggle_lay = QHBoxLayout(toggle_row)
        toggle_lay.setContentsMargins(16, 0, 16, 0)
        lbl = QLabel('自定义相机参数')
        lbl.setStyleSheet(f'color: {AppleColors.TEXT}; font-size: 15px;')
        toggle_lay.addWidget(lbl)
        toggle_lay.addStretch()
        self.custom_cam_toggle = AppleToggle(False)
        toggle_lay.addWidget(self.custom_cam_toggle)
        toggle_card.addRow(toggle_row)
        layout.addWidget(toggle_card)

        # 内参卡片
        self.intrinsic_card = AppleCard('内参')
        self.focal_length = AppleSliderRow('焦距', 10.0, 200.0, 35.0, 1.0, 1, ' mm')
        self.sensor_w = AppleSliderRow('传感器宽', 10.0, 70.0, 36.0, 0.5, 1, ' mm')
        self.sensor_h = AppleSliderRow('传感器高', 10.0, 70.0, 24.0, 0.5, 1, ' mm')
        self.intrinsic_card.addRow(self.focal_length)
        self.intrinsic_card.addRow(self.sensor_w)
        self.intrinsic_card.addRow(self.sensor_h)
        self.intrinsic_card.setEnabled(False)
        layout.addWidget(self.intrinsic_card)

        # 外参卡片
        self.extrinsic_card = AppleCard('外参')
        self.cam_height = AppleSliderRow('相机高度', 0.5, 20.0, 1.6, 0.1, 1, ' m')
        self.cam_pitch = AppleSliderRow('俯仰角', -45.0, 45.0, 0.0, 0.5, 1, ' °')
        self.extrinsic_card.addRow(self.cam_height)
        self.extrinsic_card.addRow(self.cam_pitch)
        self.extrinsic_card.setEnabled(False)
        layout.addWidget(self.extrinsic_card)

        # 默认参数说明
        info_label = QLabel('关闭自定义时使用默认参数：焦距 35mm / 全画幅 36x24mm / 高度 1.6m')
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f'color: {AppleColors.TEXT_TER}; font-size: 12px; padding: 4px 4px;')
        layout.addWidget(info_label)

        layout.addStretch()
        self.setWidget(container)


# ===========================================================================
# 页面：输出设置
# ===========================================================================

class OutputPage(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet('QScrollArea { border: none; background: transparent; }')

        container = QWidget()
        container.setStyleSheet('background: transparent;')
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        title = QLabel('输出设置')
        title.setStyleSheet(f'color: {AppleColors.TEXT}; font-size: 28px; font-weight: bold;')
        layout.addWidget(title)

        subtitle = QLabel('配置输出图像尺寸与随机种子')
        subtitle.setStyleSheet(f'color: {AppleColors.TEXT_SEC}; font-size: 14px;')
        layout.addWidget(subtitle)

        layout.addSpacing(4)

        spinbox_style = f'''
            QSpinBox {{
                background: {AppleColors.BG_TERTIARY};
                border: none;
                border-radius: 8px;
                padding: 6px 12px;
                color: {AppleColors.TEXT};
                font-size: 14px;
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                width: 20px;
                border: none;
            }}
        '''

        size_card = AppleCard('图像尺寸')

        # 宽度行
        w_row = QWidget()
        w_row.setFixedHeight(52)
        w_lay = QHBoxLayout(w_row)
        w_lay.setContentsMargins(16, 0, 16, 0)
        w_lbl = QLabel('宽度')
        w_lbl.setStyleSheet(f'color: {AppleColors.TEXT}; font-size: 15px;')
        w_lay.addWidget(w_lbl)
        w_lay.addStretch()
        self.out_width = QSpinBox()
        self.out_width.setRange(64, 4096)
        self.out_width.setValue(512)
        self.out_width.setSuffix(' px')
        self.out_width.setFixedWidth(120)
        self.out_width.setStyleSheet(spinbox_style)
        w_lay.addWidget(self.out_width)
        size_card.addRow(w_row)

        # 高度行
        h_row = QWidget()
        h_row.setFixedHeight(52)
        h_lay = QHBoxLayout(h_row)
        h_lay.setContentsMargins(16, 0, 16, 0)
        h_lbl = QLabel('高度')
        h_lbl.setStyleSheet(f'color: {AppleColors.TEXT}; font-size: 15px;')
        h_lay.addWidget(h_lbl)
        h_lay.addStretch()
        self.out_height = QSpinBox()
        self.out_height.setRange(64, 4096)
        self.out_height.setValue(512)
        self.out_height.setSuffix(' px')
        self.out_height.setFixedWidth(120)
        self.out_height.setStyleSheet(spinbox_style)
        h_lay.addWidget(self.out_height)
        size_card.addRow(h_row)

        # 锁定比例行
        lock_row = QWidget()
        lock_row.setFixedHeight(52)
        lock_lay = QHBoxLayout(lock_row)
        lock_lay.setContentsMargins(16, 0, 16, 0)
        lock_lbl = QLabel('锁定 1:1 比例')
        lock_lbl.setStyleSheet(f'color: {AppleColors.TEXT}; font-size: 15px;')
        lock_lay.addWidget(lock_lbl)
        lock_lay.addStretch()
        self.lock_toggle = AppleToggle(True)
        lock_lay.addWidget(self.lock_toggle)
        size_card.addRow(lock_row)

        layout.addWidget(size_card)

        # 种子卡片
        seed_card = AppleCard('随机种子')
        seed_row = QWidget()
        seed_row.setFixedHeight(52)
        seed_lay = QHBoxLayout(seed_row)
        seed_lay.setContentsMargins(16, 0, 16, 0)
        seed_lbl = QLabel('种子值')
        seed_lbl.setStyleSheet(f'color: {AppleColors.TEXT}; font-size: 15px;')
        seed_lay.addWidget(seed_lbl)
        seed_desc = QLabel('-1 = 随机')
        seed_desc.setStyleSheet(f'color: {AppleColors.TEXT_TER}; font-size: 12px;')
        seed_lay.addWidget(seed_desc)
        seed_lay.addStretch()
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 999999)
        self.seed_spin.setValue(42)
        self.seed_spin.setSpecialValueText('随机')
        self.seed_spin.setFixedWidth(120)
        self.seed_spin.setStyleSheet(spinbox_style)
        seed_lay.addWidget(self.seed_spin)
        seed_card.addRow(seed_row)
        layout.addWidget(seed_card)

        layout.addStretch()
        self.setWidget(container)


# ===========================================================================
# 页面：批量处理
# ===========================================================================

class BatchPage(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet('QScrollArea { border: none; background: transparent; }')

        container = QWidget()
        container.setStyleSheet('background: transparent;')
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        title = QLabel('批量处理')
        title.setStyleSheet(f'color: {AppleColors.TEXT}; font-size: 28px; font-weight: bold;')
        layout.addWidget(title)

        subtitle = QLabel('批量为深度图生成雨滴掩码')
        subtitle.setStyleSheet(f'color: {AppleColors.TEXT_SEC}; font-size: 14px;')
        layout.addWidget(subtitle)

        layout.addSpacing(4)

        line_edit_style = f'''
            QLineEdit {{
                background: {AppleColors.BG_TERTIARY};
                border: none;
                border-radius: 8px;
                padding: 6px 12px;
                color: {AppleColors.TEXT};
                font-size: 13px;
            }}
        '''

        # 输入目录
        input_card = AppleCard('输入')
        input_row = QWidget()
        input_row.setFixedHeight(52)
        input_lay = QHBoxLayout(input_row)
        input_lay.setContentsMargins(16, 0, 16, 0)
        self.batch_dir_edit = QLineEdit()
        self.batch_dir_edit.setPlaceholderText('选择深度图目录...')
        self.batch_dir_edit.setStyleSheet(line_edit_style)
        input_lay.addWidget(self.batch_dir_edit)
        self.batch_browse_btn = AppleButton('浏览', primary=False)
        self.batch_browse_btn.setFixedWidth(72)
        self.batch_browse_btn.setFixedHeight(32)
        input_lay.addWidget(self.batch_browse_btn)
        input_card.addRow(input_row)
        layout.addWidget(input_card)

        # 输出目录
        output_card = AppleCard('输出')
        output_row = QWidget()
        output_row.setFixedHeight(52)
        output_lay = QHBoxLayout(output_row)
        output_lay.setContentsMargins(16, 0, 16, 0)
        self.batch_out_edit = QLineEdit('rain_masks')
        self.batch_out_edit.setPlaceholderText('输出目录...')
        self.batch_out_edit.setStyleSheet(line_edit_style)
        output_lay.addWidget(self.batch_out_edit)
        self.batch_out_btn = AppleButton('浏览', primary=False)
        self.batch_out_btn.setFixedWidth(72)
        self.batch_out_btn.setFixedHeight(32)
        output_lay.addWidget(self.batch_out_btn)
        output_card.addRow(output_row)
        layout.addWidget(output_card)

        # 进度
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(f'''
            QProgressBar {{
                background: {AppleColors.BG_TERTIARY};
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background: {AppleColors.ACCENT};
                border-radius: 3px;
            }}
        ''')
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel('')
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet(f'color: {AppleColors.TEXT_SEC}; font-size: 13px;')
        layout.addWidget(self.progress_label)

        self.batch_btn = AppleButton('开始批量渲染', primary=True)
        layout.addWidget(self.batch_btn)

        layout.addStretch()
        self.setWidget(container)


# ===========================================================================
# 主窗口
# ===========================================================================

class RainRendererApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setMinimumSize(1100, 750)
        self.resize(1200, 800)

        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'rain_icon.ico')
        if os.path.isfile(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.current_mask = None
        self.render_thread = None
        self.depth_path = None

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        central = QWidget()
        central.setStyleSheet(f'background-color: {AppleColors.BG};')
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 标题栏
        self.title_bar = AppleTitleBar(self)
        root_layout.addWidget(self.title_bar)

        # 主体：侧边栏 + 内容
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # 侧边栏
        self.sidebar = AppleSidebar()
        body_layout.addWidget(self.sidebar)

        # 分隔线
        sep = QFrame()
        sep.setFixedWidth(1)
        sep.setStyleSheet(f'background-color: {AppleColors.SEPARATOR};')
        body_layout.addWidget(sep)

        # 内容区
        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f'background-color: {AppleColors.BG};')

        self.render_page = RenderParamsPage()
        self.depth_page = DepthMapPage()
        self.camera_page = CameraPage()
        self.output_page = OutputPage()
        self.batch_page = BatchPage()

        self.stack.addWidget(self.render_page)
        self.stack.addWidget(self.depth_page)
        self.stack.addWidget(self.camera_page)
        self.stack.addWidget(self.output_page)
        self.stack.addWidget(self.batch_page)

        body_layout.addWidget(self.stack, stretch=1)
        root_layout.addWidget(body, stretch=1)

        # 全局滑块样式
        self.setStyleSheet(self.styleSheet() + f'''
            QSlider::groove:horizontal {{
                background: {AppleColors.BG_TERTIARY};
                height: 4px;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: white;
                width: 20px;
                height: 20px;
                margin: -8px 0;
                border-radius: 10px;
            }}
            QSlider::sub-page:horizontal {{
                background: {AppleColors.ACCENT};
                border-radius: 2px;
            }}
        ''')

    def _connect_signals(self):
        # 侧边栏导航
        self.sidebar.page_changed.connect(self.stack.setCurrentIndex)

        # 预设
        self.sidebar.preset_combo.currentTextChanged.connect(self._on_preset)

        # 渲染/保存
        self.render_page.render_btn.clicked.connect(self._on_render)
        self.render_page.save_btn.clicked.connect(self._on_save)

        # 深度图
        self.depth_page.use_depth_toggle.toggled.connect(self._on_depth_toggle)
        self.depth_page.browse_btn.clicked.connect(self._browse_depth)

        # 相机
        self.camera_page.custom_cam_toggle.toggled.connect(self._on_cam_toggle)

        # 输出尺寸锁定
        self.output_page.lock_toggle.toggled.connect(self._on_lock_ratio)
        self.output_page.out_width.valueChanged.connect(self._on_width_changed)

        # 批量处理
        self.batch_page.batch_browse_btn.clicked.connect(self._browse_batch_dir)
        self.batch_page.batch_out_btn.clicked.connect(self._browse_batch_out)
        self.batch_page.batch_btn.clicked.connect(self._on_batch_render)

    # ---- 事件处理 ----

    def _on_preset(self, name):
        preset = PRESETS.get(name)
        if preset is None:
            return
        p = self.render_page
        p.rain_rate.setValue(preset['rain_rate'])
        p.wind_speed.setValue(preset['wind_speed'])
        p.wind_dir_fine.setValue(preset['wind_direction'])
        p.exposure.setValue(preset['exposure_time'])
        p.turbulence.setValue(preset['turbulence_deg'])

    def _on_depth_toggle(self, checked):
        self.depth_page.file_card.setEnabled(checked)
        self.depth_page.depth_mode_combo.setEnabled(not checked)
        can_set_size = not checked
        self.output_page.out_width.setEnabled(can_set_size)
        self.output_page.out_height.setEnabled(can_set_size)

    def _on_cam_toggle(self, checked):
        self.camera_page.intrinsic_card.setEnabled(checked)
        self.camera_page.extrinsic_card.setEnabled(checked)

    def _on_lock_ratio(self, checked):
        if checked:
            self.output_page.out_height.setValue(self.output_page.out_width.value())

    def _on_width_changed(self, val):
        if self.output_page.lock_toggle.isChecked():
            self.output_page.out_height.blockSignals(True)
            self.output_page.out_height.setValue(val)
            self.output_page.out_height.blockSignals(False)

    def _browse_depth(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择深度图', '',
            '图片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*)'
        )
        if path:
            self.depth_path = path
            self.depth_page.depth_path_edit.setText(path)
            self._show_depth_preview(path)

    def _show_depth_preview(self, path):
        try:
            raw = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape
                scale = min(240 / w, 160 / h)
                new_w, new_h = int(w * scale), int(h * scale)
                preview = cv2.resize(img, (new_w, new_h))
                preview_color = cv2.applyColorMap(preview, cv2.COLORMAP_INFERNO)
                preview_rgb = cv2.cvtColor(preview_color, cv2.COLOR_BGR2RGB)
                qimg = QImage(preview_rgb.data, new_w, new_h,
                              new_w * 3, QImage.Format_RGB888)
                self.depth_page.depth_preview_label.setPixmap(
                    QPixmap.fromImage(qimg))
                self.depth_page.depth_preview_label.setText('')
        except Exception:
            self.depth_page.depth_preview_label.setText('预览失败')

    def _browse_batch_dir(self):
        d = QFileDialog.getExistingDirectory(self, '选择深度图目录')
        if d:
            self.batch_page.batch_dir_edit.setText(d)

    def _browse_batch_out(self):
        d = QFileDialog.getExistingDirectory(self, '选择输出目录')
        if d:
            self.batch_page.batch_out_edit.setText(d)

    def _get_render_params(self):
        rp = self.render_page
        op = self.output_page
        params = {
            'rain_rate': rp.rain_rate.value(),
            'wind_speed': rp.wind_speed.value(),
            'wind_direction': rp.wind_dir_fine.value(),
            'exposure_time': rp.exposure.value(),
            'turbulence_deg': rp.turbulence.value(),
            'focus_distance': rp.focus_dist.value(),
            'depth_scale': rp.depth_scale.value(),
            'image_width': op.out_width.value(),
            'image_height': op.out_height.value(),
        }

        if self.depth_page.use_depth_toggle.isChecked() and self.depth_path:
            params['depth_path'] = self.depth_path
        else:
            params['depth_path'] = None
            params['depth_mode'] = self.depth_page.depth_mode_combo.currentText()

        cp = self.camera_page
        if cp.custom_cam_toggle.isChecked():
            params['focal_length'] = cp.focal_length.value()
            params['sensor_width'] = cp.sensor_w.value()
            params['sensor_height'] = cp.sensor_h.value()
            params['cam_height'] = cp.cam_height.value()
            params['cam_pitch'] = cp.cam_pitch.value()
        else:
            params['focal_length'] = 35.0
            params['sensor_width'] = 36.0
            params['sensor_height'] = 24.0
            params['cam_height'] = 1.6
            params['cam_pitch'] = 0.0

        seed_val = op.seed_spin.value()
        params['seed'] = seed_val if seed_val >= 0 else None
        return params

    def _on_render(self):
        if self.render_thread and self.render_thread.isRunning():
            return

        params = self._get_render_params()

        self.render_page.render_btn.setEnabled(False)
        self.render_page.render_btn.setText('渲染中...')

        self.render_thread = RenderThread(params)
        self.render_thread.finished.connect(self._on_render_done)
        self.render_thread.error.connect(self._on_render_error)
        self.render_thread.start()

    def _on_render_done(self, mask):
        self.current_mask = mask
        self._display_mask(mask)

        total = mask.size
        rain_px = int(np.sum(mask > 0))
        coverage = rain_px / total * 100
        avg_b = np.mean(mask[mask > 0]) if rain_px > 0 else 0
        h, w = mask.shape

        self.render_page.stats_label.setText(
            f'{w} x {h}    |    {rain_px:,} 雨滴像素    |    '
            f'{coverage:.1f}% 覆盖率    |    亮度 {avg_b:.0f}/255'
        )

        self.render_page.render_btn.setEnabled(True)
        self.render_page.render_btn.setText('渲染')
        self.render_page.save_btn.setEnabled(True)

        self._show_info_bar('渲染完成', 'success')

    def _on_render_error(self, err):
        self.render_page.render_btn.setEnabled(True)
        self.render_page.render_btn.setText('渲染')
        self._show_info_bar(f'渲染失败: {err}', 'error')

    def _display_mask(self, mask):
        h, w = mask.shape
        rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        bytes_per_line = w * 3
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label = self.render_page.preview_label
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(label.size(), Qt.KeepAspectRatio,
                               Qt.SmoothTransformation)
        label.setPixmap(scaled)

    def _on_save(self):
        if self.current_mask is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, '保存雨滴掩码', 'rain_mask.png',
            'PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp);;所有文件 (*)'
        )
        if path:
            cv2.imwrite(path, self.current_mask)
            self._show_info_bar(f'已保存到 {os.path.basename(path)}', 'success')

    def _on_batch_render(self):
        depth_dir = self.batch_page.batch_dir_edit.text().strip()
        output_dir = self.batch_page.batch_out_edit.text().strip()

        if not depth_dir or not os.path.isdir(depth_dir):
            self._show_info_bar('请选择有效的深度图目录', 'error')
            return

        depth_files = sorted(glob.glob(os.path.join(depth_dir, '*_depth.png')))
        if not depth_files:
            depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.png')))
        if not depth_files:
            self._show_info_bar('目录中没有找到深度图', 'warning')
            return

        os.makedirs(output_dir, exist_ok=True)

        bp = self.batch_page
        bp.progress_bar.setVisible(True)
        bp.progress_bar.setMaximum(len(depth_files))
        bp.progress_bar.setValue(0)
        bp.batch_btn.setEnabled(False)

        params = self._get_render_params()

        for idx, dpath in enumerate(depth_files):
            filename = os.path.basename(dpath)
            out_name = filename.replace('_depth', '_rain_mask')
            if out_name == filename:
                base, ext = os.path.splitext(filename)
                out_name = f'{base}_rain_mask{ext}'
            out_path = os.path.join(output_dir, out_name)

            seed = params.get('seed')
            img_seed = (seed + idx) if seed is not None else None

            try:
                mask = render_rain_mask(
                    depth_path=dpath,
                    rain_rate=params['rain_rate'],
                    wind_speed=params['wind_speed'],
                    wind_direction=params['wind_direction'],
                    exposure_time=params['exposure_time'],
                    turbulence_deg=params['turbulence_deg'],
                    depth_scale=params['depth_scale'],
                    focal_length=params['focal_length'],
                    sensor_width=params['sensor_width'],
                    sensor_height=params['sensor_height'],
                    seed=img_seed,
                )
                cv2.imwrite(out_path, mask)
            except Exception as e:
                bp.progress_label.setText(f'{filename} 失败: {e}')

            bp.progress_bar.setValue(idx + 1)
            bp.progress_label.setText(f'{idx+1} / {len(depth_files)}')
            QApplication.processEvents()

        bp.batch_btn.setEnabled(True)
        self._show_info_bar(
            f'批量完成: {len(depth_files)} 张 -> {output_dir}', 'success'
        )

    def _show_info_bar(self, text, bar_type='info'):
        bar = InfoBar(text, bar_type, self)
        bar.setFixedWidth(min(500, self.width() - 240))
        bar.move(
            self.width() - bar.width() - 20,
            self.height() - 64
        )
        bar.show_toast(3500)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_mask is not None:
            self._display_mask(self.current_mask)


# ===========================================================================
# 入口
# ===========================================================================

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont('Segoe UI', 10))
    app.setStyle('Fusion')

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(AppleColors.BG))
    palette.setColor(QPalette.WindowText, QColor(AppleColors.TEXT))
    palette.setColor(QPalette.Base, QColor(AppleColors.BG_SECONDARY))
    palette.setColor(QPalette.Text, QColor(AppleColors.TEXT))
    palette.setColor(QPalette.Button, QColor(AppleColors.BG_TERTIARY))
    palette.setColor(QPalette.ButtonText, QColor(AppleColors.TEXT))
    palette.setColor(QPalette.Highlight, QColor(AppleColors.ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor('#ffffff'))
    app.setPalette(palette)

    window = RainRendererApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
