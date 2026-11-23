"""Register NVIDIA CUDA/cuDNN wheel DLL directories on Windows before importing CTranslate2."""

import os
from pathlib import Path
import site
import sys
import sysconfig

_REGISTERED = False
_DLL_HANDLES: list[object] = []


def register_nvidia_dll_dirs() -> list[str]:
    """Add installed `nvidia/*/bin` wheel directories to DLL search path and PATH."""
    global _REGISTERED
    if _REGISTERED or os.name != "nt":
        return []

    candidates: list[Path] = []
    for sp in list(site.getsitepackages()) + [
        sysconfig.get_paths().get("purelib", ""),
        sysconfig.get_paths().get("platlib", ""),
        str(Path(sys.prefix) / "Lib" / "site-packages"),
    ]:
        if not sp:
            continue
        nvidia_dir = Path(sp) / "nvidia"
        if nvidia_dir.is_dir():
            for bin_dir in sorted(nvidia_dir.glob("*/bin")):
                if bin_dir.is_dir():
                    candidates.append(bin_dir.resolve())

    added: list[str] = []
    seen: set[str] = set()
    current_path = os.environ.get("PATH", "")
    path_parts = current_path.split(os.pathsep) if current_path else []

    for bin_dir in candidates:
        s = str(bin_dir)
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_HANDLES.append(os.add_dll_directory(s))
            except OSError:
                pass
        if s not in path_parts:
            path_parts.insert(0, s)
        added.append(s)

    if added:
        os.environ["PATH"] = os.pathsep.join(path_parts)
    _REGISTERED = True
    return added
