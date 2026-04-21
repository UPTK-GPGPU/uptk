#!/usr/bin/env python3
import subprocess
import sys
import os

MODULES = ["device", "context", "memory", "module", "stream", "event", "graph", "launch"]
TEST_DIR = "../build/test"

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

    total = 0
    passed = 0

    for line in output.splitlines():
        line = line.strip()

        # 严格统计：只有 Test: 才算用例
        if "=====" in line and "Test:" in line:
            total += 1

        # 严格统计：只要出现成功就计数
        if "Result:" in line and "✅ TEST PASSED" in line:
            passed += 1

    # 安全兜底
    passed = min(passed, total)
    failed = total - passed
    failed = max(failed, 0)

    return total, passed, failed

def main():
    if len(sys.argv) != 2:
        print(f"用法：python3 {sys.argv[0]} <module|all>")
        print("支持模块：" + " ".join(MODULES) + " all")
        sys.exit(1)

    arg = sys.argv[1]

    if arg in MODULES:
        total, passed, failed = run_and_parse(arg)
        rate = passed / total * 100 if total > 0 else 0.0

        print("\n" + "="*50)
        print("           接口测试统计")
        print("="*50)
        print(f"模块：{arg}")
        print(f"总用例：{total}")
        print(f"通过：{passed}")
        print(f"失败：{failed}")
        print(f"通过率：{rate:.2f}%")
        print("="*50)

    elif arg == "all":
        grand_total = 0
        grand_passed = 0
        grand_failed = 0
        module_results = []

        for module in MODULES:
            t, p, f = run_and_parse(module)
            module_results.append((module, t, p, f))
            grand_total += t
            grand_passed += p
            grand_failed += f

        print("\n" + "="*60)
        print("           所有模块测试汇总")
        print("="*60)
        for name, t, p, f in module_results:
            rate = p / t * 100 if t > 0 else 0
            print(f"{name:10s} | 总用例：{t:4d} | 通过：{p:4d} | 失败：{f:4d} | 通过率：{rate:6.2f}%")

        print("-" * 60)
        g_rate = grand_passed / grand_total * 100 if grand_total else 0
        print(f"总计        | 总用例：{grand_total:4d} | 通过：{grand_passed:4d} | 失败：{grand_failed:4d} | 通过率：{g_rate:6.2f}%")
        print("=" * 60)

    else:
        print("无效参数")
        sys.exit(1)

if __name__ == "__main__":
    main()
