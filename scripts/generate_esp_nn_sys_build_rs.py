#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CMAKE_FILE = ROOT / "crates" / "esp_nn_sys" / "vendor" / "esp-nn" / "CMakeLists.txt"
TEMPLATE_FILE = ROOT / "crates" / "esp_nn_sys" / "build-template.rs"
OUTPUT_FILE = ROOT / "crates" / "esp_nn_sys" / "build.rs"


PLACEHOLDERS = {
    "c_srcs": "{{C_SRCS}}",
    "s3_srcs": "{{ESP32S3_SRCS}}",
    "p4_srcs": "{{ESP32P4_SRCS}}",
}


def extract_cmake_list(content: str, name: str) -> list[str]:
    pattern = rf"set\s*\(\s*{re.escape(name)}\s*(.*?)\)"
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        raise RuntimeError(f"Cannot find set({name} ...) in {CMAKE_FILE}")

    body = match.group(1)

    files = re.findall(r'"([^"]+)"', body)

    if not files:
        raise RuntimeError(f"Found set({name} ...), but no quoted files were extracted")

    return files


def format_rust_array_items(files: list[str], tag: str) -> str:
    return f"{tag}\n" + "\n".join(f'    "{file}",' for file in files)


def generate() -> str:
    if not CMAKE_FILE.exists():
        raise FileNotFoundError(f"Missing CMake file: {CMAKE_FILE}")

    if not TEMPLATE_FILE.exists():
        raise FileNotFoundError(f"Missing template file: {TEMPLATE_FILE}")

    cmake_content = CMAKE_FILE.read_text(encoding="utf-8")
    template = TEMPLATE_FILE.read_text(encoding="utf-8")

    extracted = {
        name: extract_cmake_list(cmake_content, name)
        for name in PLACEHOLDERS
    }

    output = template

    for name, placeholder in PLACEHOLDERS.items():
        replacement = format_rust_array_items(extracted[name], name.upper())
        if placeholder not in output:
            raise RuntimeError(f"Template missing placeholder: {placeholder}")
        output = output.replace(placeholder, replacement)

    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate crates/esp_nn_sys/build.rs from ESP-NN CMakeLists.txt."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether build.rs is up to date without writing it.",
    )
    args = parser.parse_args()

    try:
        generated = generate()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.check:
        if not OUTPUT_FILE.exists():
            print(f"error: {OUTPUT_FILE} does not exist", file=sys.stderr)
            return 1

        current = OUTPUT_FILE.read_text(encoding="utf-8")

        if current != generated:
            print("error: build.rs is out of date. Run:", file=sys.stderr)
            print("  python scripts/generate_esp_nn_sys_build_rs.py", file=sys.stderr)
            return 1

        print("build.rs is up to date")
        return 0

    OUTPUT_FILE.write_text(generated, encoding="utf-8", newline="\n")
    print(f"generated {OUTPUT_FILE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
