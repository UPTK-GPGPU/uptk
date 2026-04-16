#!/usr/bin/env python3
import subprocess
import sys
import os

MODULES = ["device", "context", "memory", "module", "stream", "event", "graph", "launch"]
TEST_DIR = "../build/tests"

def run_and_parse(module):
    exe = os.path.join(TEST_DIR, module)
    if not os.path.exists(exe):
        print(f"错误：找不到可执行文件 {exe}")
        return 0, 0, 0

    print(f"\n=== 运行模块测试：{module} ===")
    print(f"路径：{exe}\n")

    result = subprocess.run(
        [exe],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='ignore'
    )
    output = result.stdout
    print(output)

    # =======================
    # 逐行解析 最稳 不会漏
    # =======================
    total = 0
    passed = 0

    lines = output.splitlines()
    for line in lines:
        line = line.strip()

        # 计数：只要看到 Test: 就算一个用例
        if "Test:" in line and "=====" in line or "====" in line:
            total += 1

        # 统计通过
        if "Result:" in line and "TEST PASSED" in line:
            passed += 1

    failed = total - passed
    return total, passed, failed

def main():
    if len(sys.argv) != 2:
        print(f"用法：python3 {sys.argv[0]} <module>")
        print("支持模块：" + " ".join(MODULES))
        sys.exit(1)

    module = sys.argv[1]
    if module not in MODULES:
        print("不支持的模块")
        sys.exit(1)

    total, passed, failed = run_and_parse(module)
    rate = passed / total * 100 if total > 0 else 0.0

    print("\n" + "="*50)
    print("           接口测试统计")
    print("="*50)
    print(f"模块：{module}")
    print(f"总接口数：{total}")
    print(f"通过：{passed}")
    print(f"失败：{failed}")
    print(f"通过率：{rate:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
