#!/usr/bin/env python3
"""Check that every relative link in the repo's markdown files resolves to an existing file.

Skips external links (http/https/mailto) and pure in-page anchors (#...).
Anchors on relative links are stripped before checking the target path.

Usage: python3 tools/check_links.py
Exit code 1 if any dead link is found.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")

dead = []
files = sorted(
    p for p in REPO.rglob("*.md")
    if ".ipynb_checkpoints" not in p.parts and ".git" not in p.parts
)
for f in files:
    in_fence = False
    for lineno, line in enumerate(f.read_text().splitlines(), 1):
        if re.match(r"\s*(```|~~~)", line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in LINK.finditer(line):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path = target.split("#", 1)[0]
            if not path:
                continue
            resolved = (f.parent / path).resolve()
            if not resolved.exists():
                dead.append((f.relative_to(REPO), lineno, target))

for rel, lineno, target in dead:
    print(f"DEAD {rel}:{lineno} -> {target}")
print(f"\n{'FAIL' if dead else 'OK'}: {len(dead)} dead link(s) across {len(files)} markdown file(s)")
sys.exit(1 if dead else 0)
