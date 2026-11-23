"""Dark desktop theme stylesheet for RealtimeTranslator."""

DARK_QSS = """
QMainWindow, QWidget {
    background-color: #14171f;
    color: #e5e9f0;
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #2a303d;
    border-radius: 6px;
    margin-top: 10px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
    color: #93c5fd;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QLineEdit, QComboBox, QSpinBox {
    background-color: #1d222d;
    border: 1px solid #333b4d;
    border-radius: 4px;
    padding: 5px 8px;
    color: #e5e9f0;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border-color: #60a5fa;
}
QPushButton {
    background-color: #252b38;
    border: 1px solid #394256;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #2f3747;
    border-color: #60a5fa;
}
QPushButton#primaryBtn {
    background-color: #2563eb;
    border-color: #3b82f6;
    color: #ffffff;
    padding: 8px 16px;
}
QPushButton#primaryBtn:hover {
    background-color: #1d4ed8;
}
QPushButton#stopBtn {
    background-color: #dc2626;
    border-color: #ef4444;
    color: #ffffff;
    padding: 8px 16px;
}
QProgressBar {
    background-color: #1d222d;
    border: 1px solid #333b4d;
    border-radius: 4px;
    text-align: center;
    height: 14px;
}
QProgressBar::chunk {
    background-color: #34d399;
    border-radius: 3px;
}
QTableWidget {
    background-color: #181c26;
    alternate-background-color: #1d222e;
    gridline-color: #2a303d;
    border: 1px solid #2a303d;
    border-radius: 4px;
}
QHeaderView::section {
    background-color: #1f2431;
    color: #cbd5e1;
    border: none;
    border-right: 1px solid #2a303d;
    border-bottom: 1px solid #2a303d;
    padding: 6px;
    font-weight: 600;
}
QTabWidget::pane {
    border: 1px solid #2a303d;
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #1d222d;
    border: 1px solid #2a303d;
    padding: 6px 14px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background-color: #283040;
    color: #93c5fd;
    font-weight: 600;
}
QLabel#warningLabel {
    color: #fbbf24;
    font-size: 12px;
}
QLabel#statusLabel {
    color: #94a3b8;
    font-size: 12px;
}
"""
