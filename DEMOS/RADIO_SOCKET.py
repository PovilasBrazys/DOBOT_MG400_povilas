import socket
import time
import numpy as np
import pyaudio

HOST = '192.168.1.6'  # IP of the Dobot MG400
PORT = 8000

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

AMPLITUDE_GAIN = 20  # increase responsiveness

def map_amplitude_to_y(amplitude):
    norm = min((amplitude * AMPLITUDE_GAIN) / 30000, 1.0)
    return 50 - (norm * 100)

def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("Connected to robot")

        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            peak = np.abs(audio_data).max()
            y = map_amplitude_to_y(peak)

            message = f"Y={round(y, 2)}"
            s.sendall(message.encode())
            print(f"Sent: {message}")

            time.sleep(0.1)

if __name__ == '__main__':
    main()
