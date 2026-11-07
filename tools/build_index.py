#!/usr/bin/env python3
"""Build data/lectures.json from the audio files in assets/.

Folder layout (one folder per lecture):

    assets/
      ben-crystal-original-pronunciation/
        lecture.json                 optional: title, text corrections
        full/                        the whole lecture (one or several parts)
          Part 1.mp3
        01 What was that.mp3         the file name IS the phrase text
        02 What we can learn from it.mp3
      another-lecture/
        ...

Rules for file names
  * The name without the extension is the text the student must say.
  * An optional number at the start sets the order: "01 ", "1. ", "01_", "01-".
  * Underscores become spaces. Punctuation is optional; it is ignored by the analysis.
  * Characters that are not allowed in file names (for example "?") can be added
    in lecture.json under "texts".

lecture.json (all fields optional)
    {
      "title": "Ben Crystal on Original Pronunciation",
      "texts": { "01 What was that.mp3": "What was that?" }
    }

Run:  python3 tools/build_index.py     (GitHub Actions runs it on every push)
Uses only the Python standard library.
"""
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
OUT = ROOT / "data" / "lectures.json"
AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".oga", ".webm", ".aac", ".flac"}
NUM_PREFIX = re.compile(r"^\s*(\d+)\s*[.)_\-\s]\s*")


def phrase_from_name(stem):
    """Return (order, text) from a file name without extension."""
    order = None
    m = NUM_PREFIX.match(stem)
    if m:
        order = int(m.group(1))
        stem = stem[m.end():]
    text = re.sub(r"\s+", " ", stem.replace("_", " ")).strip()
    return order, text


def title_from_folder(name):
    t = re.sub(r"[-_]+", " ", name).strip()
    return t[:1].upper() + t[1:]


def url_for(*parts):
    return "/".join(quote(p, safe="") for p in parts)


def full_lecture(folder):
    """Audio files in <lecture>/full/ : the complete lecture, in one or several parts."""
    sub = next((p for p in folder.iterdir() if p.is_dir() and p.name.lower() == "full"), None)
    if not sub:
        return []
    parts = []
    for f in sub.iterdir():
        if f.is_file() and f.suffix.lower() in AUDIO and not f.name.startswith("._"):
            order, title = phrase_from_name(f.stem)
            parts.append((order if order is not None else 10**9, title.lower(), {
                "title": title or f.stem,
                "audio": url_for("assets", folder.name, sub.name, f.name),
            }))
    parts.sort(key=lambda x: (x[0], x[1]))
    return [p[2] for p in parts]


def build():
    warnings = []
    lectures = []
    if not ASSETS.is_dir():
        sys.exit("assets/ folder not found")

    folders = sorted(p for p in ASSETS.iterdir() if p.is_dir() and not p.name.startswith("."))
    loose = [p.name for p in ASSETS.iterdir() if p.is_file() and p.suffix.lower() in AUDIO]
    if loose:
        warnings.append(f"{len(loose)} audio file(s) lie directly in assets/ and are ignored; "
                        f"move them into a lecture folder: {', '.join(sorted(loose)[:5])}")

    for folder in folders:
        meta = {}
        meta_file = folder / "lecture.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                warnings.append(f"{meta_file.relative_to(ROOT)}: not valid JSON ({e}); defaults used")
        texts = meta.get("texts", {}) or {}

        items = []
        for f in folder.iterdir():
            if not f.is_file() or f.suffix.lower() not in AUDIO or f.name.startswith("._"):
                continue
            order, text = phrase_from_name(f.stem)
            text = texts.get(f.name, text)
            if not re.search(r"[A-Za-z]", text):
                warnings.append(f"{f.relative_to(ROOT)}: no English words in the name, skipped")
                continue
            items.append((order if order is not None else 10**9, text.lower(), {
                "text": text,
                "audio": url_for("assets", folder.name, f.name),
            }))
        if not items:
            continue
        items.sort(key=lambda x: (x[0], x[1]))

        lecture = {
            "id": folder.name,
            "title": meta.get("title") or title_from_folder(folder.name),
            "phrases": [x[2] for x in items],
        }
        full = full_lecture(folder)
        if full:
            lecture["full"] = full
        lectures.append(lecture)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"lectures": lectures}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    total = sum(len(l["phrases"]) for l in lectures)
    print(f"{len(lectures)} lecture(s), {total} phrase(s) -> {OUT.relative_to(ROOT)}")
    for l in lectures:
        full = f", full lecture in {len(l['full'])} part(s)" if l.get("full") else ""
        print(f"  {l['id']}: {len(l['phrases'])} phrase(s){full}")
    for w in warnings:
        print("WARNING:", w)


if __name__ == "__main__":
    build()
