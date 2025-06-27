#!/usr/bin/env python3
"""Check commit message follows project conventions."""

import sys
import re


def check_commit_message(message):
    """Validate commit message format."""
    lines = message.strip().split("\n")

    if not lines:
        print("❌ Empty commit message")
        return False

    # Check first line format: <type>: <description> [<task-id>]
    first_line = lines[0]

    # Valid commit types
    valid_types = ["feat", "fix", "refactor", "test", "docs", "chore", "perf"]

    # Pattern for first line
    pattern = r"^(" + "|".join(valid_types) + r"):\s+.+\s+\[IMPL-D\d+-\d{3}\]$"

    if not re.match(pattern, first_line):
        print(f"❌ Invalid commit message format: {first_line}")
        print(f"✅ Expected: <type>: <description> [IMPL-D<day>-<number>]")
        print(f"✅ Example: feat: Add ONNX encoder class [IMPL-D2-001]")
        print(f"✅ Valid types: {', '.join(valid_types)}")
        return False

    # Check for Refs in body
    body = "\n".join(lines[1:])
    if "Refs: #" not in body:
        print("❌ Missing 'Refs: #' in commit body")
        print("✅ Add: Refs: #D2-001 (matching your task ID)")
        return False

    return True


def main():
    """Main entry point."""
    # Read commit message from file (provided by git)
    commit_msg_file = sys.argv[1] if len(sys.argv) > 1 else ".git/COMMIT_EDITMSG"

    try:
        with open(commit_msg_file, "r") as f:
            message = f.read()
    except FileNotFoundError:
        print(f"❌ Could not read commit message from {commit_msg_file}")
        sys.exit(1)

    # Check message
    if not check_commit_message(message):
        print("\n💡 Tip: Use task branches like: git checkout -b impl/D2-001-description")
        sys.exit(1)

    print("✅ Commit message format is valid!")


if __name__ == "__main__":
    main()
