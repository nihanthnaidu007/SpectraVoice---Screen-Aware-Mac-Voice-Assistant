#!/usr/bin/env python3
"""List PyAudio input/output devices so their indices can go in config.yaml.

Run on the machine that runs the assistant:

    python scripts/list_audio_devices.py

Then set `microphone.input_device` / `tts.output_device` to the printed index.
"""

import sys


def main() -> int:
    try:
        import pyaudio
    except ImportError:
        print("PyAudio is not installed — run `pip install -r requirements.txt` first.")
        return 1

    pa = pyaudio.PyAudio()
    try:
        try:
            default_input = pa.get_default_input_device_info()
            default_output = pa.get_default_output_device_info()
        except OSError:
            default_input = default_output = None

        print(f"{'idx':>3}  {'in':>2}  {'out':>3}  name")
        for index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(index)
            is_default = ""
            if default_input and index == default_input["index"]:
                is_default += " [default input]"
            if default_output and index == default_output["index"]:
                is_default += " [default output]"
            print(
                f"{index:>3}  {info.get('maxInputChannels', 0):>2}  {info.get('maxOutputChannels', 0):>3}  "
                f"{info.get('name', '?')}{is_default}"
            )
    finally:
        pa.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
