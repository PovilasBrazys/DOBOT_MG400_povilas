import vlc
import time

# M-1 Radio stream URL
stream_url = "https://radio.m-1.fm/m1plius/aacp64"

# Create VLC media player instance
player = vlc.MediaPlayer(stream_url)

print("Playing M-1 Radio...")
player.play()


# Keep the program running while the stream plays
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping playback...")
    player.stop()