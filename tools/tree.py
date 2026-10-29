#!/usr/bin/env python3
import os
import sys

def limited_tree(path, max_files=5, prefix=""):
    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        print(prefix + "└── [Permission denied]")
        return

    shown = entries[:max_files]
    hidden = len(entries) - max_files

    for i, name in enumerate(shown):
        full = os.path.join(path, name)
        is_last = (i == len(shown) - 1) and hidden <= 0

        connector = "└── " if is_last else "├── "
        print(prefix + connector + name)

        if os.path.isdir(full):
            extension = "    " if is_last else "│   "
            limited_tree(full, max_files, prefix + extension)

    if hidden > 0:
        print(prefix + f"└── ... ({hidden} more)")

if len(sys.argv) < 2:
    print("Usage: python3 limited_tree.py <path> [max_files]")
    sys.exit(1)

path = sys.argv[1]
max_files = int(sys.argv[2]) if len(sys.argv) > 2 else 10

if not os.path.isdir(path):
    print(f"Error: '{path}' is not a valid directory")
    sys.exit(1)

print(path)
limited_tree(path, max_files)