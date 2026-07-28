#!/usr/bin/env python3
"""Check that every $...$ / ```math block in this repo's markdown actually renders on GitHub.

GitHub's markdown pipeline runs emphasis parsing over inline math candidates, so a
`$...$` segment whose content contains a `_..._` or `*...*` pair (even paired across
two segments on the same line) is silently not rendered as math -- and the underscores
or asterisks are eaten. This script uses GitHub's own renderer as ground truth:
it POSTs markdown to `gh api /markdown` (mode=gfm) and reports any `$` that survives
outside a <math-renderer>/<pre>/<code> element, i.e. math GitHub failed to recognize.

Usage:
    python3 tools/check_gh_math.py                 # check all *.md (whole-file mode)
    python3 tools/check_gh_math.py path/a.md ...   # check specific files
    python3 tools/check_gh_math.py --lines [files] # also map failures to source lines
                                                   # (batched line-by-line rendering)

Requires an authenticated `gh` CLI. Exit code 1 if any unrendered math is found.

Known safe patterns (what fixes look like):
    inline:  $`\\mathbb{E}_{x \\sim p}[f(x)]`$        (dollar-backtick, emphasis-immune)
    display: a ```math fenced block, indented to sit inside its bullet
"""

import html
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SENTINEL = "XQZ9LINESPLIT9XQZ"

# GitHub's client-side MathJax applies a macro filter AFTER the markdown API
# recognizes the math, showing "The following macros are not allowed: <name>"
# in the browser. The API cannot catch this — so we scan for known-blocked
# macros locally. Extend this set when a new one is discovered on a real page.
# (\operatorname: github/markup#1688 — use \mathrm or \mathop{\text{...}}.)
DISALLOWED_MACROS = {"operatorname"}


def math_segments(text: str):
    segs = re.findall(r"\$`(.*?)`\$", text)
    segs += re.findall(r"(?<![`$])\$([^$`\n]+)\$(?!`)", text)
    segs += re.findall(r"```math\n(.*?)```", text, re.S)
    return segs


def disallowed_macro_hits(path: Path) -> list[str]:
    hits = []
    for seg in math_segments(path.read_text()):
        for m in re.findall(r"\\([a-zA-Z]+)", seg):
            if m in DISALLOWED_MACROS:
                hits.append(f"\\{m} in: {seg.strip()[:80]}")
    return hits


def render(text: str) -> str:
    r = subprocess.run(
        ["gh", "api", "/markdown", "-f", f"text={text}", "-f", "mode=gfm"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.exit(f"gh api /markdown failed: {r.stderr.strip()}")
    return r.stdout


def strip_rendered(out: str) -> str:
    """Remove content GitHub already handled (math, code); what's left with $ is broken."""
    out = re.sub(r"<math-renderer[^>]*>.*?</math-renderer>", "✓MATH", out, flags=re.S)
    out = re.sub(r"<pre>.*?</pre>", "✓PRE", out, flags=re.S)
    out = re.sub(r"<code>.*?</code>", "✓CODE", out, flags=re.S)
    return out


def leftover_dollar_lines(out: str) -> list[str]:
    cleaned = strip_rendered(out)
    bad = []
    for chunk in cleaned.split("\n"):
        if "$" in chunk:
            bad.append(html.unescape(re.sub(r"<[^>]+>", "", chunk)).strip())
    return bad


def candidate_lines(path: Path) -> list[tuple[int, str]]:
    """Lines containing $, excluding fenced-code content."""
    out, in_fence = [], False
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if re.match(r"\s*(```|~~~)", line):
            in_fence = not in_fence
            continue
        if not in_fence and "$" in line:
            out.append((i, line))
    return out


def check_file_whole(path: Path) -> list[str]:
    text = path.read_text()
    if "$" not in text:
        return []
    return leftover_dollar_lines(render(text))


def check_file_lines(path: Path, batch_size: int = 40) -> list[tuple[int, str]]:
    """Map breakage to source lines by rendering lines individually (batched)."""
    cands = candidate_lines(path)
    broken = []
    for start in range(0, len(cands), batch_size):
        batch = cands[start:start + batch_size]
        joined = f"\n\n{SENTINEL}\n\n".join(line for _, line in batch)
        chunks = render(joined).split(SENTINEL)
        if len(chunks) != len(batch):  # sentinel got mangled; fall back to singles
            chunks = [render(line) for _, line in batch]
        for (lineno, line), chunk in zip(batch, chunks):
            if leftover_dollar_lines(chunk):
                broken.append((lineno, line))
    return broken


def main() -> None:
    args = sys.argv[1:]
    line_mode = "--lines" in args
    args = [a for a in args if a != "--lines"]
    if args:
        files = [Path(a) for a in args]
    else:
        files = sorted(
            p for p in REPO.rglob("*.md")
            if ".ipynb_checkpoints" not in p.parts and ".git" not in p.parts
        )

    total = 0
    for f in files:
        rel = f.relative_to(REPO) if f.is_absolute() and f.is_relative_to(REPO) else f
        macro_hits = disallowed_macro_hits(f)
        if macro_hits:
            total += len(macro_hits)
            print(f"\n=== {rel} ({len(macro_hits)} disallowed macro use(s)) ===")
            for h in macro_hits[:10]:
                print(f"   {h}")
        bad = check_file_whole(f)
        if not bad:
            continue
        total += len(bad)
        print(f"\n=== {rel} ({len(bad)} unrendered) ===")
        if line_mode:
            for lineno, line in check_file_lines(f):
                print(f"  {lineno}: {line.strip()[:160]}")
        else:
            for b in bad[:20]:
                print(f"   {b[:160]}")
            if len(bad) > 20:
                print(f"   ... and {len(bad) - 20} more")

    print(f"\n{'FAIL' if total else 'OK'}: {total} unrendered math line(s) across {len(files)} file(s)")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
