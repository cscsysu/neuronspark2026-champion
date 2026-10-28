import argparse
import csv
import math
import wave
from pathlib import Path


def read_wav_features(path):
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sampwidth = wav.getsampwidth()
        frames = wav.getnframes()
        if sampwidth != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported by this example script")
        raw = wav.readframes(frames)

    values = []
    for i in range(0, len(raw), sampwidth * channels):
        total = 0
        for ch in range(channels):
            offset = i + ch * sampwidth
            sample = int.from_bytes(raw[offset:offset + 2], "little", signed=True)
            total += sample
        values.append(total / channels / 32768.0)

    if not values:
        return {
            "duration_sec": 0.0,
            "rms": 0.0,
            "zcr": 0.0,
            "mean_abs": 0.0,
            "sample_rate": sample_rate,
        }

    rms = math.sqrt(sum(x * x for x in values) / len(values))
    mean_abs = sum(abs(x) for x in values) / len(values)
    zc = sum(1 for a, b in zip(values, values[1:]) if (a >= 0) != (b >= 0))
    zcr = zc / max(len(values) - 1, 1)
    return {
        "duration_sec": len(values) / sample_rate,
        "rms": rms,
        "zcr": zcr,
        "mean_abs": mean_abs,
        "sample_rate": sample_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract lightweight WAV features for NS-2026-05 baselines.")
    parser.add_argument("--csv", required=True, help="train.csv or test.csv")
    parser.add_argument("--data-root", default=".", help="dataset root containing audio paths")
    parser.add_argument("--output", required=True, help="output feature CSV")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    rows = []
    with open(args.csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for item in reader:
            audio_path = data_root / item["audio"]
            feats = read_wav_features(audio_path)
            rows.append({"id": item["id"], **feats})

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "duration_sec", "rms", "zcr", "mean_abs", "sample_rate"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
