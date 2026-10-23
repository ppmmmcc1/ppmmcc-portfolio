"""Install a git pre-commit hook that lints changed strategy specs (spec 11, role 1).

    python scripts/install_hooks.py

The hook runs `nqlab spec lint --strict` on every staged file under nqlab/specs/ and
blocks the commit while any spec still has undecided or contradictory fields. Bypass a
single commit with `git commit --no-verify` (e.g. to save a deliberately incomplete
scaffold), which is a visible, deliberate act.
"""

from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path

HOOK = """#!/bin/sh
# installed by nqlab/scripts/install_hooks.py
files=$(git diff --cached --name-only --diff-filter=ACM | grep -E '(^|/)specs/[^/]+\\.ya?ml$')
[ -z "$files" ] && exit 0
top=$(git rev-parse --show-toplevel)
cd "$top/{rel}" || exit 1
PY=.venv/bin/python
[ -x "$PY" ] || PY=.venv/Scripts/python.exe
paths=""
for f in $files; do paths="$paths $top/$f"; done
"$PY" -m nqlab.cli spec lint --strict $paths
"""


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    top = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=here, text=True).strip())
    rel = here.relative_to(top).as_posix() or "."
    hook = top / ".git" / "hooks" / "pre-commit"
    hook.write_text(HOOK.replace("{rel}", rel), encoding="utf-8", newline="\n")
    hook.chmod(hook.stat().st_mode | stat.S_IEXEC)
    print(f"installed {hook}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
