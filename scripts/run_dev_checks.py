"""Developer script to run pytest test suite and verify SDK examples."""

import subprocess
import sys


def main():
    print("=== Running Pytest Test Suite ===")
    res = subprocess.run([sys.executable, "-m", "pytest", "-v", "tests/"])
    if res.returncode != 0:
        print("[FAIL] Pytest failed.")
        sys.exit(res.returncode)

    print("\n=== Running SDK Examples Verification ===")
    examples = [
        "examples/01_quickstart_sdk.py",
        "examples/02_custom_target_profile.py",
        "examples/03_sarif_pr_export.py",
    ]
    for example in examples:
        print(f"Executing {example}...")
        ex_res = subprocess.run([sys.executable, example])
        if ex_res.returncode != 0:
            print(f"[FAIL] Example {example} failed.")
            sys.exit(ex_res.returncode)

    print("\n[OK] All dev checks & examples passed successfully!")


if __name__ == "__main__":
    main()
