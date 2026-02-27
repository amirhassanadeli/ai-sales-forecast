from pathlib import Path

IGNORE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "migrations",
    "node_modules", "staticfiles", ".idea"
}
IGNORE_FILES = {".DS_Store"}


def print_tree(root: Path, prefix: str = "", max_depth: int = 6, depth: int = 0):
    if depth > max_depth:
        return

    entries = []
    for p in root.iterdir():
        if p.name in IGNORE_DIRS or p.name in IGNORE_FILES:
            continue
        # فقط پوشه‌ها و فایل‌های مهم
        entries.append(p)

    # مرتب‌سازی: اول پوشه‌ها بعد فایل‌ها
    entries.sort(key=lambda x: (x.is_file(), x.name.lower()))

    for i, p in enumerate(entries):
        is_last = (i == len(entries) - 1)
        branch = "└── " if is_last else "├── "
        print(prefix + branch + p.name)

        if p.is_dir():
            extension = "    " if is_last else "│   "
            print_tree(p, prefix + extension, max_depth, depth + 1)


if __name__ == "__main__":
    root = Path(".").resolve()
    print(root.name)
    print_tree(root, max_depth=7)
