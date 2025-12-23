import os
import cv2
import numpy as np
from yt_dlp import YoutubeDL

# ─── Configuration ────────────────────────────────────────────────────────────
YOUTUBE_URL         = "https://www.youtube.com/watch?v=MNn9qKG2UFI"
DOWNLOAD_DIR        = "./downloads"
VIDEO_FILENAME      = "yt_video.mp4"
THRESHOLD_ZERO_BITS = 17       # require ≥17 zeros in XOR per pixel
SIMILARITY_RATIO    = 0.70     # require ≥70% pixels similar
MAX_SECONDS         = 30       # only process up to 30 seconds
# ───────────────────────────────────────────────────────────────────────────────

def download_with_ytdlp(url, out_dir, filename):
    """Download best progressive MP4 via yt-dlp."""
    os.makedirs(out_dir, exist_ok=True)
    ydl_opts = {
        'format': 'mp4[ext=mp4]+bestaudio/best',
        'outtmpl': os.path.join(out_dir, filename),
        'noplaylist': True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
    print(f"Downloaded video to: {path}")
    return path

def frame_to_24int(frame):
    """Pack an H×W×3 RGB frame into a flat array of 24-bit ints."""
    r = frame[:, :, 0].astype(np.uint32)
    g = frame[:, :, 1].astype(np.uint32)
    b = frame[:, :, 2].astype(np.uint32)
    return ((r << 16) | (g << 8) | b).ravel()

def is_similar_arr(a, b):
    """Return True if ≥SIMILARITY_RATIO of pixels have ≥THRESHOLD_ZERO_BITS zeros in XOR."""
    xor = np.bitwise_xor(a, b)
    ones = np.unpackbits(xor.view(np.uint8)).reshape(-1, 24).sum(axis=1)
    zeros = 24 - ones
    matched = np.count_nonzero(zeros >= THRESHOLD_ZERO_BITS)
    ratio = matched / zeros.size
    print(f"    Matched {matched}/{zeros.size} pixels → {ratio:.1%}")
    return ratio >= SIMILARITY_RATIO

def find_redundant_ranges(int_frames, fps):
    """
    For each base frame:
      • Try offsets [fps, fps//2, fps//4, fps//8, 1]
      • On match at cand: mark base+1..cand redundant, base=cand+1
      • If no match: base += 10
    """
    N = len(int_frames)
    redundant = set()
    base = 0

    # Precompute [fps, fps//2, fps//4, ..., 1]
    offs, step = [], int(fps)
    while step >= 1:
        offs.append(step)
        step //= 2
    if offs[-1] != 1:
        offs.append(1)

    while base < N - 1:
        print(f"\n== Base frame: {base} ==")
        candidates = [base + o for o in offs if base + o < N]
        print("  Compare against frames:", candidates)

        for cand in candidates:
            print(f"  Comparing frame {base} ↔ frame {cand} ...")
            if is_similar_arr(int_frames[base], int_frames[cand]):
                # mark redundant
                redundant.update(range(base+1, cand+1))
                print(f"    → Match! Marking frames {base+1}–{cand} redundant.")
                base = cand + 1
                break
            else:
                print(f"    → No match with frame {cand}.")
        else:
            new_base = min(base + 10, N - 1)
            print(f"  No match at any offset; advancing base to {new_base}.")
            base = new_base

    return sorted(redundant)

def main():
    # 1) download
    video_path = download_with_ytdlp(YOUTUBE_URL, DOWNLOAD_DIR, VIDEO_FILENAME)
    # 2) open with OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video at {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT), MAX_SECONDS * fps))
    print(f"Loaded {total_frames} frames @ {fps:.2f} FPS")

    # 3) read and pack frames
    frames = []
    for _ in range(total_frames):
        ret, f = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()

    int_frames = [frame_to_24int(f) for f in frames]

    # 4) find redundant
    redundant = find_redundant_ranges(int_frames, fps)

    # 5) summary
    print(f"\nAll redundant frames ({len(redundant)}): {redundant}")

if __name__ == "__main__":
    main()
