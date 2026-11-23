"""Frameless translucent always-on-top PySide6 subtitle overlay window."""

from dataclasses import dataclass

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPaintEvent, QPainterPath
from PySide6.QtWidgets import QLabel, QSizeGrip, QVBoxLayout, QWidget

from rtt.config import AppConfig


@dataclass
class _OverlayEntry:
    """Single source/translation line pair tracked in the overlay history."""

    seq_id: int
    source: str
    translation: str | None = None
    pending: bool = True


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
        self._max_lines = 3
        self._entries: list[_OverlayEntry] = []

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
        self.set_max_lines(getattr(config, "overlay_max_lines", 3))
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
        if self._entries:
            self._render_entries()

    def set_max_lines(self, max_lines: int) -> None:
        self._max_lines = max(1, int(max_lines))
        if len(self._entries) > self._max_lines:
            self._entries = self._entries[-self._max_lines :]
        self._render_entries()

    def clear(self) -> None:
        self._entries.clear()
        self.source_label.setText("")
        self.translation_label.setText("Ready for live translation...")

    def set_click_through(self, enabled: bool) -> None:
        self._click_through = bool(enabled)
        was_visible = self.isVisible()
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, self._click_through)
        self._grip.setVisible(not self._click_through)
        if was_visible:
            self.show()

    def _format_translation_line(self, entry: _OverlayEntry) -> str:
        if entry.pending:
            if self._display_mode == "translation":
                return f"... {entry.source}"
            return "Translating..."
        return entry.translation if entry.translation else "(translation unavailable)"

    def _render_entries(self) -> None:
        if not self._entries:
            return
        self.source_label.setText("\n".join(e.source for e in self._entries))
        self.translation_label.setText(
            "\n".join(self._format_translation_line(e) for e in self._entries)
        )

    def show_pending(self, seq_id: int, src: str) -> None:
        for entry in self._entries:
            if entry.seq_id == seq_id:
                entry.source = src
                entry.translation = None
                entry.pending = True
                break
        else:
            self._entries.append(
                _OverlayEntry(seq_id=seq_id, source=src, translation=None, pending=True)
            )
            if len(self._entries) > self._max_lines:
                self._entries = self._entries[-self._max_lines :]
        self._render_entries()

    def show_translation(
        self,
        seq_id: int,
        src: str,
        trn: str | None,
        merged_ids: list[int] | None = None,
    ) -> None:
        if merged_ids:
            remove_ids = {mid for mid in merged_ids if mid != seq_id}
            if remove_ids:
                self._entries = [e for e in self._entries if e.seq_id not in remove_ids]

        for entry in self._entries:
            if entry.seq_id == seq_id:
                entry.source = src
                entry.translation = trn
                entry.pending = False
                break
        else:
            self._entries.append(
                _OverlayEntry(seq_id=seq_id, source=src, translation=trn, pending=False)
            )
            if len(self._entries) > self._max_lines:
                self._entries = self._entries[-self._max_lines :]
        self._render_entries()

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
