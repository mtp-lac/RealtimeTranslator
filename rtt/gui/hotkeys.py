"""Windows global hotkey registration (`Ctrl+Shift+T`, `Ctrl+Shift+M`) via RegisterHotKey."""

import ctypes
from ctypes import wintypes
import os
from typing import Callable

from PySide6.QtCore import QAbstractNativeEventFilter, QObject, Signal

WM_HOTKEY = 0x0312
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_NOREPEAT = 0x4000

VK_T = 0x54
VK_M = 0x4D

HOTKEY_TOGGLE_OVERLAY = 1
HOTKEY_TOGGLE_MUTE = 2


class _HotkeySignals(QObject):
    toggle_overlay = Signal()
    toggle_mute = Signal()


class GlobalHotkeys(QAbstractNativeEventFilter):
    """Native Windows event filter handling system-wide translation hotkeys."""

    def __init__(
        self,
        on_toggle_overlay: Callable[[], None] | None = None,
        on_toggle_mute: Callable[[], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        register: bool = True,
    ):
        super().__init__()
        self._signals = _HotkeySignals()
        if on_toggle_overlay is not None:
            self._signals.toggle_overlay.connect(on_toggle_overlay)
        if on_toggle_mute is not None:
            self._signals.toggle_mute.connect(on_toggle_mute)
        self._on_status = on_status
        self._registered_ids: list[int] = []
        if register:
            self.register_all()

    def register_all(self) -> None:
        if os.name != "nt":
            return
        user32 = ctypes.windll.user32
        mods = MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT
        failures: list[str] = []
        for hk_id, vk, label in (
            (HOTKEY_TOGGLE_OVERLAY, VK_T, "Ctrl+Shift+T"),
            (HOTKEY_TOGGLE_MUTE, VK_M, "Ctrl+Shift+M"),
        ):
            if hk_id in self._registered_ids:
                continue
            try:
                ok = bool(user32.RegisterHotKey(None, hk_id, mods, vk))
            except Exception:
                ok = False
            if ok:
                self._registered_ids.append(hk_id)
            else:
                failures.append(label)

        if failures and self._on_status is not None:
            self._on_status(f"Could not register global hotkey(s): {', '.join(failures)}")

    def unregister_all(self) -> None:
        if os.name != "nt" or not self._registered_ids:
            self._registered_ids.clear()
            return
        user32 = ctypes.windll.user32
        for hk_id in list(self._registered_ids):
            try:
                user32.UnregisterHotKey(None, hk_id)
            except Exception:
                pass
        self._registered_ids.clear()

    def nativeEventFilter(self, eventType, message):
        if eventType == b"windows_generic_MSG" and message is not None:
            try:
                msg = wintypes.MSG.from_address(int(message))
            except (TypeError, ValueError):
                return False, 0
            if msg.message == WM_HOTKEY:
                hk_id = int(msg.wParam)
                if hk_id == HOTKEY_TOGGLE_OVERLAY:
                    self._signals.toggle_overlay.emit()
                    return True, 0
                if hk_id == HOTKEY_TOGGLE_MUTE:
                    self._signals.toggle_mute.emit()
                    return True, 0
        return False, 0
