#!/usr/bin/env python3
"""Tests for the CUDA -> UPTK converter."""

from __future__ import annotations

import difflib
import tempfile
import unittest
from pathlib import Path

from converter import CudaToUptkConverter

ROOT = Path(__file__).resolve().parents[2]
CUDA_DEMO = ROOT / "example_port" / "cuda_port_demo" / "MyDemo_CUDA" / "MyDemo.cu"
UPTK_DEMO = ROOT / "example_port" / "cuda_port_demo" / "MyDemo_UPTK" / "MyDemo.cu"
CUDA_CMAKE = ROOT / "example_port" / "cuda_port_demo" / "MyDemo_CUDA" / "CMakeLists.txt"
UPTK_CMAKE = ROOT / "example_port" / "cuda_port_demo" / "MyDemo_UPTK" / "CMakeLists.txt"


def _normalize(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").splitlines()]
    return "\n".join(lines).strip() + "\n"


class ConverterTests(unittest.TestCase):
    @classmethod
    setUpClass(cls) -> None:
        cls.converter = CudaToUptkConverter()

    def test_runtime_demo_matches_reference(self) -> None:
        source = CUDA_DEMO.read_text(encoding="utf-8")
        expected = _normalize(UPTK_DEMO.read_text(encoding="utf-8"))
        converted, replacements, warnings = self.converter.convert_source(
            source, CUDA_DEMO.name
        )
        actual = _normalize(converted)

        self.assertFalse(warnings, msg=f"Unexpected warnings: {warnings}")
        self.assertTrue(replacements, msg="Expected at least one replacement")
        if actual != expected:
            diff = "\n".join(
                difflib.unified_diff(
                    expected.splitlines(),
                    actual.splitlines(),
                    fromfile="expected",
                    tofile="actual",
                    lineterm="",
                )
            )
            self.fail(f"Converted output differs from reference:\n{diff}")

    def test_runtime_api_replacement(self) -> None:
        source = """
#include <cuda_runtime_api.h>
int main() {
    cudaMalloc(nullptr, 0);
    cudaMemcpy(nullptr, nullptr, 0, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaFree(nullptr);
    return 0;
}
"""
        converted, _, _ = self.converter.convert_source(source, "sample.cu")
        self.assertIn("#include <UPTK_runtime_api.h>", converted)
        self.assertIn("UPTKMalloc", converted)
        self.assertIn("UPTKMemcpyHostToDevice", converted)
        self.assertIn("UPTKDeviceSynchronize", converted)
        self.assertIn("UPTKFree", converted)
        self.assertNotIn("cudaMalloc", converted)

    def test_kernel_code_unchanged(self) -> None:
        source = """
__global__ void add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
"""
        converted, replacements, _ = self.converter.convert_source(source, "kernel.cu")
        self.assertEqual(source, converted)
        self.assertEqual(replacements, [])

    def test_cmake_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cmake_path = Path(tmp_dir) / "CMakeLists.txt"
            cmake_path.write_text(
                "target_link_libraries(MyDemo cudart)\n",
                encoding="utf-8",
            )
            report = self.converter.convert_cmake(cmake_path, in_place=True)
            text = cmake_path.read_text(encoding="utf-8")
            self.assertIn("UPTKrt", text)
            self.assertTrue(report.replacements)

    def test_reference_cmake_has_uptkrt(self) -> None:
        expected = UPTK_CMAKE.read_text(encoding="utf-8")
        self.assertIn("UPTKrt", expected)


def main() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ConverterTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
