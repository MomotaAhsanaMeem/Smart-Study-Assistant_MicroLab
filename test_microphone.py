#!/usr/bin/env python3
import sys
import time
import struct
import math

# List of wake commands supported by the lamp system
WAKE_WORDS = ["activate", "turn on", "lamp on", "wake", "start"]

def print_banner():
    print("=========================================================")
    print("🎤  USB MICROPHONE & SPEECH RECOGNITION TEST SCRIPT  🎤")
    print("=========================================================")
    print("This script will help you verify that your USB microphone is")
    print("working, receiving audio, and transcribing commands.")
    print("=========================================================\n")

def list_and_find_microphone():
    try:
        import pyaudio
    except ImportError:
        print("❌ Error: 'pyaudio' is not installed in the python environment.")
        print("Please run 'make install' to set up all dependencies.")
        sys.exit(1)
        
    p = pyaudio.PyAudio()
    device_count = p.get_device_count()
    
    print("🔍 Scanning all audio input devices...")
    print(f"Found {device_count} total audio devices:\n")
    
    bt_devices = []
    usb_devices = []
    default_devices = []
    other_devices = []
    
    for i in range(device_count):
        try:
            info = p.get_device_info_by_index(i)
            name = info.get('name', 'Unknown')
            name_lower = name.lower()
            max_input_channels = info.get('maxInputChannels', 0)
            
            # Print details of every input-capable device
            if max_input_channels > 0:
                print(f"  [Index {i}]: Device: '{name}'")
                print(f"             Input Channels: {max_input_channels}")
                print(f"             Default Sample Rate: {int(info.get('defaultSampleRate', 0))} Hz\n")
                
                # Categorize
                if any(kw in name_lower for kw in ['bluetooth', 'bluez', 'headset', 'handsfree', 'sony', 'wf-']):
                    bt_devices.append((i, name))
                elif any(kw in name_lower for kw in ['usb', 'pnp', 'microphone', 'mic']):
                    usb_devices.append((i, name))
                elif any(kw in name_lower for kw in ['default', 'pulse', 'pipewire']):
                    default_devices.append((i, name))
                else:
                    other_devices.append((i, name))
        except Exception as e:
            print(f"  [Index {i}]: Error reading device: {e}")
            
    p.terminate()
    
    if default_devices:
        print(f"✨ SUCCESS: Automatically identified Default/PipeWire Mic: '{default_devices[0][1]}' at index {default_devices[0][0]}")
        return default_devices[0][0]
    elif bt_devices:
        print(f"✨ SUCCESS: Automatically identified Bluetooth Mic: '{bt_devices[0][1]}' at index {bt_devices[0][0]}")
        return bt_devices[0][0]
    elif usb_devices:
        print(f"✨ SUCCESS: Automatically identified USB Mic: '{usb_devices[0][1]}' at index {usb_devices[0][0]}")
        return usb_devices[0][0]
    elif other_devices:
        print(f"✨ SUCCESS: Automatically identified Mic: '{other_devices[0][1]}' at index {other_devices[0][0]}")
        return other_devices[0][0]
    else:
        print("⚠️  WARNING: Could not automatically detect any audio input device.")
        print("Using the default system input device (index None).")
        return None

def run_vu_meter(device_index):
    try:
        import pyaudio
    except ImportError:
        print("❌ Error: 'pyaudio' is not installed.")
        return
        
    p = pyaudio.PyAudio()
    channels = 1
    sample_rate = 16000
    
    try:
        if device_index is not None:
            info = p.get_device_info_by_index(device_index)
            channels = min(1, info.get('maxInputChannels', 1))
            sample_rate = int(info.get('defaultSampleRate', 16000))
    except Exception:
        pass

    print("\n---------------------------------------------------------")
    print("🔊 STARTING LIVE VOLUME / VU METER")
    print("Speak into the microphone to see the level bar change.")
    print("Press Ctrl+C to return to the main menu.")
    print("---------------------------------------------------------\n")
    
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024
        )
    except Exception as e:
        print(f"❌ Error opening audio stream: {e}")
        print("Ensure the USB microphone is connected and not in use by another program.")
        p.terminate()
        return

    try:
        while True:
            try:
                data = stream.read(1024, exception_on_overflow=False)
            except Exception:
                continue
                
            count = len(data) / 2
            if count == 0:
                continue
            shorts = struct.unpack(f"{int(count)}h", data)
            
            # Calculate stats
            val_min = min(shorts)
            val_max = max(shorts)
            val_mean = sum(shorts) / count
            
            # Subtract mean to remove DC offset (DC bias correction)
            shorts_no_dc = [s - val_mean for s in shorts]
            sum_squares_no_dc = sum((s / 32768.0) ** 2 for s in shorts_no_dc)
            rms_no_dc = math.sqrt(sum_squares_no_dc / count)
            
            # Raw RMS
            sum_squares_raw = sum((s / 32768.0) ** 2 for s in shorts)
            rms_raw = math.sqrt(sum_squares_raw / count)
            
            # Draw visual bar using corrected RMS (max 30 chars)
            bar_length = int(rms_no_dc * 150)
            bar_length = min(30, bar_length)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            
            # Check for saturation / clipping
            saturated_msg = ""
            if val_min <= -32760 or val_max >= 32760:
                saturated_msg = "⚠️  SATURATED"
            elif val_min == val_max:
                saturated_msg = "⚠️  FLAT LINE"
                
            sys.stdout.write(
                f"\rLevel: [{bar}] RMS(Corrected): {rms_no_dc:.4f} | RMS(Raw): {rms_raw:.4f} | "
                f"Min/Max: {val_min}/{val_max} | Mean: {val_mean:.1f} {saturated_msg}   "
            )
            sys.stdout.flush()
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n\nStopped VU Meter.")
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        p.terminate()

def test_listening(device_index):
    try:
        import speech_recognition as sr
    except ImportError:
        print("❌ Error: 'speech_recognition' is not installed in the python environment.")
        print("Please run 'make install' to set up all dependencies.")
        sys.exit(1)
        
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 1500
    recognizer.dynamic_energy_threshold = True
    
    print("\n---------------------------------------------------------")
    print("🔊 CALIBRATION PHASE")
    print("---------------------------------------------------------")
    print("Please remain quiet. Calibrating microphone for background noise...")
    
    try:
        with sr.Microphone(device_index=device_index) as source:
            recognizer.adjust_for_ambient_noise(source, duration=2.0)
            print(f"Calibration finished. Adjusted energy threshold to: {int(recognizer.energy_threshold)}")
            print("---------------------------------------------------------")
            print("🎙️  STARTING CONTINUOUS LISTENING LOOP")
            print("Press Ctrl+C at any time to return to the main menu.")
            print("---------------------------------------------------------\n")
            
            while True:
                try:
                    print("🔊 [LISTENING...] Speak a command clearly now...")
                    # Capture audio: timeout of 5s to start speaking, 3s limit on phrase duration
                    audio = recognizer.listen(source, timeout=5.0, phrase_time_limit=3.0)
                    
                    print("⚡ [PROCESSING...] Audio captured. Sending to Google Speech API...")
                    
                    try:
                        text = recognizer.recognize_google(audio).lower()
                        print(f"📝 [HEARD]: \"{text}\"")
                        
                        # Check if any wake words match
                        matched = [word for word in WAKE_WORDS if word in text]
                        if matched:
                            print(f"👉 [MATCH]: Yes! Found matching wake phrase(s): {matched}")
                            print("           🔥 (The system would ACTIVATE under these conditions)")
                        else:
                            print("❌ [NO MATCH]: Recognized text does not match any wake word.")
                            
                    except sr.UnknownValueError:
                        print("❔ [HEARD]: Could not understand the audio. Speak more clearly.")
                    except sr.RequestError as e:
                        print(f"❌ [API ERROR]: Google Speech Recognition service request failed: {e}")
                        print("               (Check your Raspberry Pi's internet connection)")
                        
                except sr.WaitTimeoutError:
                    print("⏳ [TIMEOUT]: No speech detected for 5 seconds. Retrying...")
                    print("💡 Tip: If it keeps timing out, you can run the live VU Meter to check if")
                    print("        the microphone is receiving audio, or adjust volume using 'alsamixer'.")
                print("-" * 50)
                
    except KeyboardInterrupt:
        print("\n\nListening loop stopped by user.")
    except Exception as e:
        print(f"\n❌ Execution Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. If you get a 'Device Index Error', check if the USB mic is plugged in.")
        print("2. Ensure that your Raspberry Pi has internet access for the Google Speech API.")

if __name__ == "__main__":
    try:
        print_banner()
        idx = list_and_find_microphone()
        
        # Prompt the user if they want to override the index
        print(f"\nWe will use device index: {idx}")
        print("Press Enter to accept, or type a custom index number:")
        user_input = input("Index (default: Press Enter): ").strip()
        if user_input:
            idx = int(user_input)
            print(f"Using manual override index: {idx}")
            
        while True:
            print("\n---------------------------------------------------------")
            print("Select an option to run:")
            print("  [1] Test Speech Recognition & Wake Words")
            print("  [2] Run Live Volume / VU Level Meter (Check if mic is hearing sound)")
            print("  [3] Exit")
            print("---------------------------------------------------------")
            choice = input("Option (1/2/3): ").strip()
            
            if choice == "1":
                test_listening(idx)
            elif choice == "2":
                run_vu_meter(idx)
            elif choice == "3" or choice == "":
                print("Exiting test script.")
                break
            else:
                print("Invalid choice. Please select 1, 2, or 3.")
                
    except KeyboardInterrupt:
        print("\n\nTest script terminated gracefully.")
        sys.exit(0)
