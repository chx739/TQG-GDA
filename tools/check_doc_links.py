#!/usr/bin/env python3
"""Validate local Markdown file targets; external links and anchors are out of scope."""
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

root = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
errors = []
checked = 0
for name in paths:
    if not name.endswith('.md') or name.startswith('references/'):
        continue
    path = root / name
    text = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
    for link in re.findall(r'\]\(([^\s)]+)\)', text):
        if re.match(r'^[a-zA-Z][\w+.-]*:|^/|^#', link):
            continue
        target = unquote(link.split('#', 1)[0])
        if not target:
            continue
        checked += 1
        if not (path.parent / target).exists():
            errors.append(f'{name}: {link}')
for error in errors:
    print(error, file=sys.stderr)
print(f'Checked {checked} local Markdown targets; {len(errors)} missing. External links and anchors not checked.')
sys.exit(bool(errors))
