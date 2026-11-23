"""Frameless translucent always-on-top PySide6 subtitle overlay window."""

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPaintEvent, QPainterPath
from PySide6.QtWidgets import QLabel, QSizeGrip, QVBoxLayout, QWidget

from rtt.config import AppConfig


class SubtitleOverlay(QWidget):
    """Hardware-composited floating subtitle overlay with drag, resize, and click-through."""

    geometry_changed = Signal(list)

    def __init__(self, config: AppConfig | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("RealtimeTranslator Subtitles")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMinimumSize(320, 90)
        self.resize(760, 140)

        self._drag_offset: QPoint | None = None
        self._click_through = False
        self._display_mode = "bilingual"
        self._source_color = "#CBD5E1"
        self._translation_color = "#FFFFFF"
        self._font_family = "Segoe UI"
        self._font_size = 22

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 10)
        layout.setSpacing(6)

        self.source_label = QLabel("")
        self.source_label.setWordWrap(True)
        self.source_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.translation_label = QLabel("Ready for live translation...")
        self.translation_label.setWordWrap(True)
        self.translation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.source_label)
        layout.addWidget(self.translation_label, 1)

        self._grip = QSizeGrip(self)
        layout.addWidget(self._grip, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

        if config is not None:
            self.apply_config(config)
        else:
            self._refresh_typography()

    def apply_config(self, config: AppConfig) -> None:
        self._font_family = config.overlay_font_family or "Segoe UI"
        self._font_size = int(config.overlay_font_size or 22)
        self._source_color = config.overlay_source_color or "#CBD5E1"
        self._translation_color = config.overlay_translation_color or "#FFFFFF"
        self._refresh_typography()
        self.set_display_mode(config.overlay_display_mode)
        if config.overlay_geometry and len(config.overlay_geometry) == 4:
            x, y, w, h = config.overlay_geometry
            self.setGeometry(int(x), int(y), max(320, int(w)), max(90, int(h)))
        self.set_click_through(config.overlay_click_through)

    def _refresh_typography(self) -> None:
        src_size = max(10, int(round(self._font_size * 0.72)))
        self.source_label.setFont(QFont(self._font_family, src_size))
        self.source_label.setStyleSheet(
            f"color: {self._source_color}; background: transparent;"
        )
        trn_font = QFont(self._font_family, self._font_size)
        trn_font.setBold(True)
        self.translation_label.setFont(trn_font)
        self.translation_label.setStyleSheet(
            f"color: {self._translation_color}; background: transparent;"
        )

    def set_font(self, family: str, size: int) -> None:
        self._font_family = family or "Segoe UI"
        self._font_size = max(10, int(size))
        self._refresh_typography()

    def set_colors(self, source_color: str, translation_color: str) -> None:
        self._source_color = source_color
        self._translation_color = translation_color
        self._refresh_typography()

    def set_display_mode(self, mode: str) -> None:
        norm = (mode or "bilingual").lower()
        if norm not in {"bilingual", "translation", "original"}:
            norm = "bilingual"
        self._display_mode = norm
        self.source_label.setVisible(norm in {"bilingual", "original"})
        self.translation_label.setVisible(norm in {"bilingual", "translation"})

    def set_click_through(self, enabled: bool) -> None:
        self._click_through = bool(enabled)
        was_visible = self.isVisible()
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, self._click_through)
        self._grip.setVisible(not self._click_through)
        if was_visible:
            self.show()

    def show_pending(self, seq_id: int, src: str) -> None:
        self.source_label.setText(src)
        if self._display_mode == "translation":
            self.translation_label.setText(f"... {src}")
        else:
            self.translation_label.setText("Translating...")

    def show_translation(self, seq_id: int, src: str, trn: str | None) -> None:
        self.source_label.setText(src)
        self.translation_label.setText(trn if trn else "(translation unavailable)")

    def current_geometry(self) -> list[int]:
        g = self.geometry()
        return [g.x(), g.y(), g.width(), g.height()]

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(self.rect().adjusted(1, 1, -1, -1), 12.0, 12.0)
        painter.fillPath(path, QColor(15, 15, 20, 190))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self._click_through and event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            not self._click_through
            and self._drag_offset is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset is not None:
            self._drag_offset = None
            self.geometry_changed.emit(self.current_geometry())
            event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.geometry_changed.emit(self.current_geometry())
