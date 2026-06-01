#!/usr/bin/env python3
"""CLI for converting CUDA source files to UPTK API usage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from converter import CudaToUptkConverter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CUDA host-side API calls to UPTK equivalents."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input .cu/.cpp/.hip file or directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file or directory",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Modify files in place",
    )
    parser.add_argument(
        "--cmake",
        action="store_true",
        help="Also convert CMakeLists.txt in the input directory",
    )
    parser.add_argument(
        "--map",
        type=Path,
        help="Custom api_map.json path",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print errors",
    )
    return parser.parse_args()


def _collect_source_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    suffixes = {".cu", ".cpp", ".c", ".hip", ".cc", ".cxx"}
    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.suffix in suffixes and file_path.is_file()
    )


def _resolve_output_path(
    input_path: Path,
    root_input: Path,
    output_root: Path | None,
    in_place: bool,
) -> Path | None:
    if in_place:
        return input_path
    if output_root is None:
        return input_path.with_suffix(input_path.suffix + ".uptk")
    if root_input.is_file():
        return output_root
    relative = input_path.relative_to(root_input)
    return output_root / relative


def main() -> int:
    args = parse_args()
    converter = CudaToUptkConverter(args.map)

    input_path = args.input.resolve()
    source_files = _collect_source_files(input_path)
    if not source_files:
        print(f"No source files found under {input_path}", file=sys.stderr)
        return 1

    exit_code = 0
    for source_file in source_files:
        output_file = _resolve_output_path(source_file, input_path, args.output, args.in_place)
        report = converter.convert_file(source_file, output_file, in_place=args.in_place)
        if not args.quiet:
            print(f"Converted: {report.input_path} -> {report.output_path}")
            for item in report.replacements:
                print(f"  - {item}")
            for warning in report.warnings:
                print(f"  ! {warning}")

    if args.cmake:
        cmake_root = input_path if input_path.is_dir() else input_path.parent
        cmake_files = sorted(cmake_root.rglob("CMakeLists.txt"))
        for cmake_file in cmake_files:
            output_file = cmake_file if args.in_place else (
                args.output / cmake_file.relative_to(cmake_root)
                if args.output and cmake_root.is_dir()
                else cmake_file.with_name("CMakeLists.txt.uptk")
            )
            report = converter.convert_cmake(cmake_file, output_file, in_place=args.in_place)
            if not args.quiet:
                print(f"Converted CMake: {report.input_path} -> {report.output_path}")
                for item in report.replacements:
                    print(f"  - {item}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
