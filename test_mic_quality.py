import os
import sys
import time
import struct
import math
import wave
import numpy as np

def detect_usb_microphone():
    try:
        import pyaudio
    except ImportError:
        print("Error: 'pyaudio' is not installed.")
        sys.exit(1)
        
    p = pyaudio.PyAudio()
    device_count = p.get_device_count()
    detected_index = None
    
    for i in range(device_count):
        try:
            info = p.get_device_info_by_index(i)
            name = info.get('name', '')
            if info.get('maxInputChannels', 0) > 0 and any(kw in name.lower() for kw in ['usb', 'pnp', 'microphone']):
                detected_index = i
                print(f"Found USB Mic: '{name}' at index {i}")
                break
        except Exception:
            pass
            
    if detected_index is None:
        print("Warning: USB Mic not detected by name. Using default input device.")
    p.terminate()
    return detected_index

def record_audio(device_index, filename="test_recording.wav", duration=5):
    import pyaudio
    p = pyaudio.PyAudio()
    
    channels = 1
    sample_rate = 16000
    
    if device_index is not None:
        try:
            info = p.get_device_info_by_index(device_index)
            channels = min(1, info.get('maxInputChannels', 1))
            # Fall back to default if sample rate is unusual
            sample_rate = int(info.get('defaultSampleRate', 16000))
        except Exception as e:
            print(f"Error reading device info: {e}")
            
    print(f"\nRecording configuration: Channels={channels}, Sample Rate={sample_rate}Hz")
    print("---------------------------------------------------------")
    print(f"🎤 RECORDING WILL START IN 2 SECONDS FOR A DURATION OF {duration} SECONDS.")
    print("Please speak a test sentence clearly when the recording starts!")
    print("---------------------------------------------------------")
    
    for i in range(2, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
        
    print("🔴 RECORDING STARTED... SPEAK NOW!")
    
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
        print(f"Failed to open audio stream: {e}")
        p.terminate()
        return None, None, None
        
    frames = []
    num_chunks = int(sample_rate / 1024 * duration)
    
    for _ in range(num_chunks):
        try:
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        except Exception as e:
            print(f"Read error: {e}")
            
    print("⏹️ RECORDING FINISHED.")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save to file
    audio_data = b"".join(frames)
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(audio_data)
    wf.close()
    print(f"Saved recording to {filename}")
    
    return audio_data, sample_rate, channels

def analyze_audio(audio_data, sample_rate):
    if not audio_data:
        print("No audio data to analyze.")
        return
        
    count = len(audio_data) // 2
    shorts = struct.unpack(f"{count}h", audio_data)
    
    val_min = min(shorts)
    val_max = max(shorts)
    val_mean = sum(shorts) / count
    
    # Check clipping
    clipped_high = sum(1 for s in shorts if s >= 32760)
    clipped_low = sum(1 for s in shorts if s <= -32760)
    total_clipped = clipped_high + clipped_low
    clip_percent = (total_clipped / count) * 100
    
    # Corrected RMS (DC bias removed)
    shorts_no_dc = [s - val_mean for s in shorts]
    sum_squares_no_dc = sum((s / 32768.0) ** 2 for s in shorts_no_dc)
    rms_no_dc = math.sqrt(sum_squares_no_dc / count)
    
    # Raw RMS
    sum_squares_raw = sum((s / 32768.0) ** 2 for s in shorts)
    rms_raw = math.sqrt(sum_squares_raw / count)
    
    # Variance & Standard Deviation
    variance = sum((s - val_mean) ** 2 for s in shorts) / count
    std_dev = math.sqrt(variance)
    
    print("\n=========================================================")
    print("📊 AUDIO QUALITY ANALYSIS REPORT")
    print("=========================================================")
    print(f"Total Audio Samples: {count}")
    print(f"Min Sample Value:    {val_min}")
    print(f"Max Sample Value:    {val_max}")
    print(f"Mean (DC Offset):    {val_mean:.2f} (Ideal: ~0)")
    print(f"Raw RMS Power:       {rms_raw:.4f}")
    print(f"Corrected RMS Power: {rms_no_dc:.4f}")
    print(f"Signal Standard Dev: {std_dev:.2f}")
    print(f"Clipped Samples:     {total_clipped} ({clip_percent:.2f}% of recording)")
    
    print("\n---------------------------------------------------------")
    print("🔍 DIAGNOSIS & INTERPRETATION:")
    print("---------------------------------------------------------")
    
    # Interpret DC offset
    if abs(val_mean) > 500:
        print("⚠️ HIGH DC OFFSET: There is a strong electrical bias in the mic input.")
    else:
        print("✅ DC OFFSET OK: The average level is centered near zero.")
        
    # Interpret Clipping / Saturation
    if clip_percent > 5.0:
        print("❌ CRITICAL CLIPPING: The volume/gain is set too high or there is hardware saturation.")
    elif clip_percent > 0.5:
        print("⚠️ MODERATE CLIPPING: Some words or loud noises are clipping. Turn down input volume.")
    else:
        print("✅ NO SIGNIFICANT CLIPPING: Signal level stays within limits.")
        
    # Interpret Signal presence
    if std_dev < 10.0:
        print("❌ SILENT / DEAD LINE: The microphone is not registering any change in sound.")
    elif rms_no_dc < 0.005:
        print("⚠️ VERY LOW SIGNAL: The microphone volume is extremely low.")
    elif rms_no_dc > 0.7:
        print("❌ EXTREMELY LOUD/NOISY: Constant loud hum, buzz, or feedback.")
    else:
        print("✅ SIGNAL STRENGTH GOOD: Standard speech signal levels detected.")
        
    # Generate ASCII Waveform representation (10 time slices)
    print("\n---------------------------------------------------------")
    print("📈 ASCII WAVEFORM PREVIEW (Amplitude over Time)")
    print("---------------------------------------------------------")
    slice_size = count // 50
    for i in range(50):
        chunk = shorts[i * slice_size : (i + 1) * slice_size]
        if not chunk:
            continue
        chunk_max = max(abs(s) for s in chunk)
        bar_len = int((chunk_max / 32768.0) * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"Slice {i+1:02d}: [{bar}] Max Amp: {chunk_max}")
        
    # Attempt speech recognition transcription to "listen"
    try:
        import speech_recognition as sr
        print("\n---------------------------------------------------------")
        print("🤖 GOOGLE SPEECH TRANSCRIPTION TEST (Listening to verify text)")
        print("---------------------------------------------------------")
        recognizer = sr.Recognizer()
        with sr.AudioFile("test_recording.wav") as source:
            audio_file_data = recognizer.record(source)
        
        print("Transcribing...")
        text = recognizer.recognize_google(audio_file_data)
        print(f"✨ TRANSCRIPTION SUCCESS: \"{text}\"")
        print("✅ Audio is intelligible and successfully processed!")
    except ImportError:
        print("\nNote: speech_recognition library not available to perform transcription.")
    except sr.UnknownValueError:
        print("\n❔ TRANSCRIPTION FAILED: Could not understand speech in the audio.")
        print("Possible causes: Speaking too quietly, high ambient noise, or muffled audio quality.")
    except sr.RequestError as e:
        print(f"\n❌ TRANSCRIPTION API ERROR: Google Speech service failed: {e}")
    except Exception as e:
        print(f"\nCould not run speech transcription: {e}")
        
    print("=========================================================\n")

if __name__ == "__main__":
    idx = detect_usb_microphone()
    data, rate, channels = record_audio(idx, duration=5)
    if data:
        analyze_audio(data, rate)
