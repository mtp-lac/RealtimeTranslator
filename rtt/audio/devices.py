"""Windows WASAPI audio device enumeration via PyAudioWPatch."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioDeviceInfo:
    """Metadata describing an audio capture or WASAPI loopback endpoint."""

    index: int
    name: str
    sample_rate: int
    channels: int
    is_loopback: bool


def list_wasapi_devices() -> list[AudioDeviceInfo]:
    """Return available WASAPI microphones and loopback render devices."""
    import pyaudiowpatch as pyaudio

    devices: list[AudioDeviceInfo] = []
    with pyaudio.PyAudio() as p:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["hostApi"] != wasapi_info["index"]:
                continue
            is_lb = bool(info.get("isLoopbackDevice", False))
            if info["maxInputChannels"] > 0 or is_lb:
                devices.append(
                    AudioDeviceInfo(
                        index=int(info["index"]),
                        name=str(info["name"]),
                        sample_rate=int(info["defaultSampleRate"]),
                        channels=max(1, int(info["maxInputChannels"])),
                        is_loopback=is_lb,
                    )
                )
    return devices


def default_loopback_index() -> int | None:
    """Return the index of the default Windows WASAPI loopback device, or None."""
    try:
        import pyaudiowpatch as pyaudio

        with pyaudio.PyAudio() as p:
            info = p.get_default_wasapi_loopback()
            return int(info["index"])
    except Exception:
        return None
