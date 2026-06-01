#!/usr/bin/env python3
"""Generate CUDA->UPTK API mappings from uptk source convert files."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_CONVERT = ROOT / "src" / "runtime" / "runtime_fun_convert.cpp"
DRIVER_CONVERT = ROOT / "src" / "driver" / "driver_fun_convert.cpp"
OUTPUT = Path(__file__).resolve().parent / "api_map.json"

RUNTIME_DEF = re.compile(
    r"^\s*(?:__host__\s+)?UPTKError(?:_t)?\s+(UPTK[A-Za-z0-9_]+)\s*\(",
    re.MULTILINE,
)
RUNTIME_CALL = re.compile(r"\bcuda([A-Za-z0-9_]+)\s*\(")

DRIVER_DEF = re.compile(
    r"^\s*UPTKError\s+(UP[A-Za-z0-9_]+)\s*\(",
    re.MULTILINE,
)
DRIVER_CALL = re.compile(r"\bcu([A-Za-z0-9_]+)\s*\(")

DRIVER_TYPE_RULES = {
    "CUresult": "UPTKError",
    "CUDA_SUCCESS": "UPTKSuccess",
    "CUdeviceptr": "UPTKdeviceptr",
    "CUdeviceptr_v2": "UPTKdeviceptr",
}


def _collect_pairs(source: Path, def_pattern: re.Pattern, call_pattern: re.Pattern, uptk_prefix: str, cuda_prefix: str) -> dict[str, str]:
    text = source.read_text(encoding="utf-8")
    mapping: dict[str, str] = {}
    fallbacks: dict[str, str] = {}

    for match in def_pattern.finditer(text):
        uptk_name = match.group(1)
        block_end = text.find("\n}", match.end())
        if block_end == -1:
            continue
        block = text[match.start() : block_end]
        calls = call_pattern.findall(block)
        if not calls:
            continue
        cuda_name = f"{cuda_prefix}{calls[0]}"
        canonical = f"{uptk_prefix}{cuda_name[len(cuda_prefix):]}"
        if uptk_name == canonical:
            mapping[cuda_name] = uptk_name
        else:
            fallbacks.setdefault(cuda_name, uptk_name)

    for cuda_name, uptk_name in fallbacks.items():
        mapping.setdefault(cuda_name, uptk_name)

    return mapping


HEADER_NAME_SUFFIXES = {
    "_runtime_api",
    "_runtime",
    "_driver_types",
    "_library_types",
}


def _is_header_name(cuda_token: str) -> bool:
    return any(cuda_token.endswith(suffix) for suffix in HEADER_NAME_SUFFIXES)


def _collect_runtime_types(runtime_functions: dict[str, str]) -> dict[str, str]:
    header = (ROOT / "include" / "UPTK_runtime_api.h").read_text(encoding="utf-8")
    mapping: dict[str, str] = {}
    for token in re.findall(r"\bUPTK[A-Za-z0-9_]+\b", header):
        if token.startswith("UPTK"):
            cuda_token = "cuda" + token[4:]
            if _is_header_name(cuda_token):
                continue
            if cuda_token in runtime_functions:
                continue
            mapping[cuda_token] = token
    return mapping


def _collect_driver_types() -> dict[str, str]:
    header = (ROOT / "include" / "UPTK.h").read_text(encoding="utf-8")
    mapping = dict(DRIVER_TYPE_RULES)
    for token in re.findall(r"\bUPTK[A-Za-z0-9_]+\b", header):
        if token.startswith("UPTK"):
            cuda_token = "CU" + token[4:]
            mapping[cuda_token] = token
    return mapping


def generate() -> dict:
    runtime_functions = _collect_pairs(RUNTIME_CONVERT, RUNTIME_DEF, RUNTIME_CALL, "UPTK", "cuda")
    driver_functions = _collect_pairs(DRIVER_CONVERT, DRIVER_DEF, DRIVER_CALL, "UP", "cu")

    # Ensure common runtime calls exist even if wrapper naming differs.
    for cuda_name in list(runtime_functions):
        simple = "UPTK" + cuda_name[4:]
        runtime_functions.setdefault(cuda_name, simple)

    for cu_name in list(driver_functions):
        simple = "UP" + cu_name[2:]
        driver_functions.setdefault(cu_name, simple)

    return {
        "runtime_functions": dict(sorted(runtime_functions.items(), key=lambda item: len(item[0]), reverse=True)),
        "driver_functions": dict(sorted(driver_functions.items(), key=lambda item: len(item[0]), reverse=True)),
        "runtime_types": dict(sorted(_collect_runtime_types(runtime_functions).items(), key=lambda item: len(item[0]), reverse=True)),
        "driver_types": dict(sorted(_collect_driver_types().items(), key=lambda item: len(item[0]), reverse=True)),
    }


def main() -> None:
    data = generate()
    OUTPUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(
        f"runtime_functions={len(data['runtime_functions'])}, "
        f"driver_functions={len(data['driver_functions'])}, "
        f"runtime_types={len(data['runtime_types'])}, "
        f"driver_types={len(data['driver_types'])}"
    )


if __name__ == "__main__":
    main()
