"""Verification runner for individual phases (scripts/verify_phase.py N)."""

import sys
import subprocess
from pathlib import Path


def verify_phase(phase_num: int) -> int:
    phase_str = str(phase_num)
    test_dir = Path(f"tests/phase_{phase_str}")
    if not test_dir.exists():
        print(f"[FAIL] Test directory {test_dir} does not exist.")
        return 1

    print("=" * 65)
    print(f"VERIFYING PHASE {phase_str}: {test_dir}")
    print("=" * 65)

    python_exe = sys.executable
    cmd = [python_exe, "-m", "pytest", str(test_dir), "-v"]
    result = subprocess.run(cmd)

    print("-" * 65)
    if result.returncode == 0:
        print(f"RESULT: PHASE {phase_str} PASSED ALL TESTS [PASS]")
    else:
        print(f"RESULT: PHASE {phase_str} FAILED TESTS [FAIL]")
    print("=" * 65)

    return result.returncode


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_phase.py <PHASE_NUMBER>")
        sys.exit(1)
    phase = int(sys.argv[1])
    sys.exit(verify_phase(phase))
