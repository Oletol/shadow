#!/usr/bin/env python3
"""Build SCORM 1.2 packages for Moodle from data/lectures.json.

One package per lecture, plus one with all lectures:

    downloads/scorm/ben-crystal-original-pronunciation.zip
    downloads/scorm/all-lectures.zip
    downloads/scorm/index.html          list of the packages with download links

Each package holds only a small page (scorm/index.html) that opens the practice
from the published site (settings.json -> "publicUrl") and passes the results to
Moodle. So phrases added later appear in Moodle without uploading the package again;
a new package is needed only for a new lecture.

Run:  python3 tools/build_scorm.py            (GitHub Actions runs it on every push)
      python3 tools/build_scorm.py --url http://localhost:8000/   (for a local test)
Uses only the Python standard library.
"""
import argparse
import html
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "scorm" / "index.html"
OUT = ROOT / "downloads" / "scorm"
DEFAULT_URL = "https://oletol.github.io/shadow/"

MANIFEST = """<?xml version="1.0" encoding="UTF-8"?>
<manifest identifier="{ident}" version="1.0"
  xmlns="http://www.imsproject.org/xsd/imscp_rootv1p1p2"
  xmlns:adlcp="http://www.adlnet.org/xsd/adlcp_rootv1p2"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://www.imsproject.org/xsd/imscp_rootv1p1p2 imscp_rootv1p1p2.xsd http://www.imsglobal.org/xsd/imsmd_rootv1p2p1 imsmd_rootv1p2p1.xsd http://www.adlnet.org/xsd/adlcp_rootv1p2 adlcp_rootv1p2.xsd">
  <metadata>
    <schema>ADL SCORM</schema>
    <schemaversion>1.2</schemaversion>
  </metadata>
  <organizations default="org">
    <organization identifier="org">
      <title>{title}</title>
      <item identifier="item" identifierref="practice" isvisible="true">
        <title>{title}</title>
      </item>
    </organization>
  </organizations>
  <resources>
    <resource identifier="practice" type="webcontent" adlcp:scormtype="sco" href="index.html">
      <file href="index.html"/>
    </resource>
  </resources>
</manifest>
"""

LIST_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SCORM packages – {site}</title>
<style>
 body {{ margin: 0; background: #eef1f8; color: #505569; font: 16px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
 main {{ max-width: 720px; margin: 0 auto; padding: 32px 16px; }}
 h1 {{ color: #0a1e64; font-size: 1.5rem; margin: 0 0 8px; }}
 ul {{ padding: 0; list-style: none; }}
 li {{ background: #fff; border: 1px solid #d9deea; border-radius: 12px; padding: 14px 16px; margin: 10px 0; }}
 a {{ color: #0541f0; font-weight: 600; }}
 small {{ display: block; color: #505569; }}
</style></head>
<body><main>
<h1>SCORM packages for Moodle</h1>
<p>Download a package and add it to the course as the activity “SCORM package”. Built {built}.</p>
<ul>
{items}
</ul>
</main></body></html>
"""


def ident(s):
    """xs:ID: starts with a letter, then letters, digits, - _ ."""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-") or "lecture"
    return s if s[0].isalpha() else "x" + s


def build_one(template, url, lecture_id, title, out_path):
    query = {"scorm": "1"}
    if lecture_id:
        query["lecture"] = lecture_id
    app_url = url + ("&" if "?" in url else "?") + urlencode(query)
    page = (template
            .replace("__TITLE__", html.escape(title))
            .replace("__APP_URL__", json.dumps(app_url)))
    manifest = MANIFEST.format(ident="shadow-" + ident(lecture_id or "all"), title=html.escape(title))
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("imsmanifest.xml", manifest)
        z.writestr("index.html", page)
    return app_url


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="address of the published practice (default: publicUrl in data/settings.json)")
    args = ap.parse_args()

    settings = {}
    sf = ROOT / "data" / "settings.json"
    if sf.exists():
        settings = json.loads(sf.read_text(encoding="utf-8"))
    url = args.url or settings.get("publicUrl") or DEFAULT_URL
    if not re.match(r"^https?://", url):
        sys.exit(f"publicUrl must start with https:// – got {url!r}")
    if not url.endswith("/") and "?" not in url:
        url += "/"

    data = json.loads((ROOT / "data" / "lectures.json").read_text(encoding="utf-8"))
    lectures = [l for l in data.get("lectures", []) if l.get("phrases")]
    if not lectures:
        sys.exit("data/lectures.json has no lectures; run tools/build_index.py first")
    template = TEMPLATE.read_text(encoding="utf-8")

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.zip"):
        old.unlink()
    site_title = settings.get("title") or "Shadowing practice"
    made = []
    for l in lectures:
        title = l.get("title") or l["id"]
        name = ident(l["id"]) + ".zip"
        build_one(template, url, l["id"], title, OUT / name)
        made.append((name, title, len(l["phrases"])))
    if len(lectures) > 1:
        total = sum(len(l["phrases"]) for l in lectures)
        build_one(template, url, None, site_title, OUT / "all-lectures.zip")
        made.append(("all-lectures.zip", site_title + " – all lectures", total))

    items = "\n".join(
        f'<li><a href="{html.escape(n)}" download>{html.escape(t)}</a>'
        f'<small>{html.escape(n)} · {c} phrase{"s" if c != 1 else ""}</small></li>'
        for n, t, c in made)
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    (OUT / "index.html").write_text(
        LIST_PAGE.format(site=html.escape(site_title), built=built, items=items), encoding="utf-8")

    print(f"{len(made)} SCORM package(s) -> {OUT.relative_to(ROOT)} (practice at {url})")
    for n, t, c in made:
        print(f"  {n}: {t}, {c} phrase(s)")


if __name__ == "__main__":
    main()
