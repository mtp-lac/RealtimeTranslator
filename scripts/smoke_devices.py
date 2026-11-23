"""Smoke test: enumerate WASAPI devices and capture 3 seconds from default loopback."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rtt.audio.capture import AudioCapture
from rtt.audio.devices import default_loopback_index, list_wasapi_devices
from rtt.audio.resample import rms
from rtt.audio.ring_buffer import AudioRingBuffer


def main() -> int:
    devices = list_wasapi_devices()
    print(f"Found {len(devices)} WASAPI devices:")
    for dev in devices:
        kind = "[Loopback]" if dev.is_loopback else "[Mic]     "
        print(f"  {dev.index:3d} {kind} {dev.name} ({dev.sample_rate} Hz, {dev.channels} ch)")

    lb_idx = default_loopback_index()
    target = next((d for d in devices if d.index == lb_idx), None)
    if target is None:
        target = next((d for d in devices if d.is_loopback), None)
    if target is None:
        print("No WASAPI loopback device found.", file=sys.stderr)
        return 1

    print(f"\nCapturing 3.0s from default loopback: [{target.index}] {target.name}")
    ring = AudioRingBuffer(max_seconds=10.0)
    levels = []
    cap = AudioCapture(target, ring, on_level=levels.append)
    cap.start()
    try:
        for step in range(6):
            time.sleep(0.5)
            chunk = ring.read_available()
            print(f"  t={(step + 1) * 0.5:.1f}s | samples={chunk.size:5d} | rms={rms(chunk):.6f}")
    finally:
        cap.stop()
    print(f"Capture complete. Level callbacks received: {len(levels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
