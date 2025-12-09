from yt_dlp import YoutubeDL

url = "https://www.youtube.com/watch?v=XXXXXXXX"

ydl_opts = {
    "format": "best[ext=mp4]/best",
}

with YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
    stream_url = info["url"]

print(stream_url)