#!/usr/bin/env python3
"""Convert CUDA host-side API usage to UPTK equivalents."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_MAP = TOOL_DIR / "api_map.json"

RUNTIME_HEADER = "UPTK_runtime_api.h"
DRIVER_HEADER = "UPTK.h"

CUDA_RUNTIME_INCLUDES = (
    "cuda_runtime.h",
    "cuda_runtime_api.h",
    "driver_types.h",
    "cuda.h",
)

UPTK_RUNTIME_LIB = "UPTKrt"
UPTK_DRIVER_LIB = "UPTKdrt"


@dataclass
class ConversionReport:
    input_path: Path
    output_path: Path
    replacements: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class CudaToUptkConverter:
    def __init__(self, api_map_path: Path | None = None) -> None:
        map_path = api_map_path or DEFAULT_MAP
        if not map_path.exists():
            raise FileNotFoundError(
                f"API map not found: {map_path}. Run generate_map.py first."
            )
        data = json.loads(map_path.read_text(encoding="utf-8"))
        self.runtime_functions = data["runtime_functions"]
        self.driver_functions = data["driver_functions"]
        self.runtime_types = data["runtime_types"]
        self.driver_types = data["driver_types"]

    def convert_source(self, source: str, filename: str = "<input>") -> tuple[str, list[str], list[str]]:
        protected, restore = self._protect_include_lines(source)
        text = protected
        replacements: list[str] = []
        warnings: list[str] = []

        text, driver_fn_replacements = self._apply_mapping(text, self.driver_functions, "cu")
        replacements.extend(driver_fn_replacements)

        text, runtime_fn_replacements = self._apply_mapping(text, self.runtime_functions, "cuda")
        replacements.extend(runtime_fn_replacements)

        text, driver_type_replacements = self._apply_mapping(text, self.driver_types, "CU")
        replacements.extend(driver_type_replacements)

        text, runtime_type_replacements = self._apply_mapping(text, self.runtime_types, "cuda")
        replacements.extend(runtime_type_replacements)

        text = restore(text)
        text, header_replacements = self._ensure_headers(text, filename)
        replacements.extend(header_replacements)

        if "cudaMalloc" in source and "UPTKMalloc" not in text:
            warnings.append("Expected runtime API conversion did not occur.")

        return text, replacements, warnings

    def convert_file(
        self,
        input_path: Path,
        output_path: Path | None = None,
        in_place: bool = False,
    ) -> ConversionReport:
        input_path = input_path.resolve()
        if output_path is None:
            output_path = input_path if in_place else input_path.with_suffix(input_path.suffix + ".uptk")
        else:
            output_path = output_path.resolve()

        original = input_path.read_text(encoding="utf-8")
        converted, replacements, warnings = self.convert_source(original, input_path.name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(converted, encoding="utf-8")

        return ConversionReport(
            input_path=input_path,
            output_path=output_path,
            replacements=replacements,
            warnings=warnings,
        )

    def convert_cmake(
        self,
        input_path: Path,
        output_path: Path | None = None,
        in_place: bool = False,
    ) -> ConversionReport:
        input_path = input_path.resolve()
        if output_path is None:
            output_path = input_path if in_place else input_path.with_name(input_path.name + ".uptk")
        else:
            output_path = output_path.resolve()

        text = input_path.read_text(encoding="utf-8")
        replacements: list[str] = []
        warnings: list[str] = []

        if UPTK_RUNTIME_LIB not in text and "target_link_libraries" in text:
            new_text, count = re.subn(
                r"(target_link_libraries\s*\(\s*[^\)\n]*\bcudart\b)([^\)\n]*\))",
                rf"\1 {UPTK_RUNTIME_LIB}\2",
                text,
                count=1,
            )
            if count:
                text = new_text
                replacements.append(f"Added {UPTK_RUNTIME_LIB} to target_link_libraries")

        if "UPTKdrt" in text or "cuda.h" in text:
            if UPTK_DRIVER_LIB not in text and "target_link_libraries" in text:
                new_text, count = re.subn(
                    r"(target_link_libraries\s*\([^\)]*\))",
                    lambda match: (
                        match.group(1)[:-1] + f" {UPTK_DRIVER_LIB})"
                        if UPTK_DRIVER_LIB not in match.group(1)
                        else match.group(1)
                    ),
                    text,
                    count=1,
                )
                if count and new_text != text:
                    text = new_text
                    replacements.append(f"Added {UPTK_DRIVER_LIB} to target_link_libraries")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        return ConversionReport(
            input_path=input_path,
            output_path=output_path,
            replacements=replacements,
            warnings=warnings,
        )

    def _apply_mapping(
        self,
        text: str,
        mapping: dict[str, str],
        prefix_hint: str,
    ) -> tuple[str, list[str]]:
        replacements: list[str] = []
        for source, target in mapping.items():
            if source == target:
                continue
            pattern = re.compile(rf"\b{re.escape(source)}\b")
            new_text, count = pattern.subn(target, text)
            if count:
                replacements.append(f"{source} -> {target} ({count}x)")
                text = new_text

        # Prefix fallback for unmapped symbols.
        if prefix_hint == "cuda":
            fallback = re.compile(r"\bcuda([A-Z][A-Za-z0-9_]*)\b")

            def _runtime_fallback(match: re.Match[str]) -> str:
                return "UPTK" + match.group(1)

            new_text, count = fallback.subn(_runtime_fallback, text)
            if count:
                replacements.append(f"cuda* -> UPTK* fallback ({count}x)")
                text = new_text
        elif prefix_hint == "cu":
            fallback = re.compile(r"\bcu([A-Z][A-Za-z0-9_]*)\b")

            def _driver_fallback(match: re.Match[str]) -> str:
                return "UP" + match.group(1)

            new_text, count = fallback.subn(_driver_fallback, text)
            if count:
                replacements.append(f"cu* -> UP* fallback ({count}x)")
                text = new_text
        elif prefix_hint == "CU":
            fallback = re.compile(r"\bCU([A-Z][A-Za-z0-9_]*)\b")

            def _type_fallback(match: re.Match[str]) -> str:
                return "UPTK" + match.group(1)

            new_text, count = fallback.subn(_type_fallback, text)
            if count:
                replacements.append(f"CU* -> UPTK* fallback ({count}x)")
                text = new_text

        return text, replacements

    def _ensure_headers(self, text: str, filename: str) -> tuple[str, list[str]]:
        replacements: list[str] = []
        uses_runtime = any(token in text for token in ("UPTKMalloc", "UPTKMemcpy", "UPTKDeviceSynchronize", "UPTKError_t"))
        uses_driver = any(token in text for token in ("UPInit", "UPCtxCreate", "UPModuleLoad", "UPTKcontext"))

        if uses_runtime and RUNTIME_HEADER not in text:
            text = self._insert_after_cuda_include(text, f"#include <{RUNTIME_HEADER}>")
            replacements.append(f"Added #include <{RUNTIME_HEADER}>")

        if uses_driver and DRIVER_HEADER not in text:
            text = self._insert_after_cuda_include(text, f"#include <{DRIVER_HEADER}>")
            replacements.append(f"Added #include <{DRIVER_HEADER}>")

        if not replacements and filename.endswith((".cu", ".cpp", ".c", ".hip")):
            if "UPTK" in text and RUNTIME_HEADER not in text and uses_runtime:
                replacements.append("Runtime UPTK calls detected without runtime header")

        return text, replacements

    @staticmethod
    def _protect_include_lines(text: str) -> tuple[str, callable]:
        include_pattern = re.compile(r"^[ \t]*#include[^\n]*$", re.MULTILINE)
        placeholders: dict[str, str] = {}

        def _protect(match: re.Match[str]) -> str:
            key = f"__UPTK_INCLUDE_{len(placeholders)}__"
            placeholders[key] = match.group(0)
            return key

        protected = include_pattern.sub(_protect, text)

        def restore(value: str) -> str:
            restored = value
            for key, original in placeholders.items():
                restored = restored.replace(key, original)
            return restored

        return protected, restore

    @staticmethod
    def _insert_after_cuda_include(text: str, include_line: str) -> str:
        lines = text.splitlines(keepends=True)
        insert_at = 0
        for index, line in enumerate(lines):
            if "#include" in line and any(name in line for name in CUDA_RUNTIME_INCLUDES):
                insert_at = index + 1
        if insert_at == 0:
            for index, line in enumerate(lines):
                if line.strip().startswith("#include"):
                    insert_at = index + 1
        if include_line + "\n" in text or include_line + "\r\n" in text:
            return text
        if insert_at < len(lines) and lines[insert_at].strip() == "":
            lines[insert_at] = include_line + "\n"
        else:
            lines.insert(insert_at, include_line + "\n")
        return "".join(lines)
