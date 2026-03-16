#!/usr/bin/env python3
"""
HIP测试用例批量运行脚本
自动运行当前目录下所有测试用例，汇总测试结果
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class TestResult:
    """测试结果数据类"""
    name: str
    status: str  # "pass", "fail", "error", "skip"
    duration: float  # 运行时间（秒）
    output: str  # 输出日志
    return_code: int  # 返回码
    timestamp: str  # 时间戳

class HIPTestRunner:
    """HIP测试用例运行器"""
    
    def __init__(self, test_dirs: List[str] = None):
        """
        初始化测试运行器
        
        Args:
            test_dirs: 指定要运行的测试目录列表，None则自动发现
        """
        self.current_dir = Path.cwd()
        self.results: Dict[str, TestResult] = {}
        self.start_time = None
        self.end_time = None
        
        # 如果未指定测试目录，自动发现所有测试目录
        if test_dirs is None:
            self.test_dirs = self.discover_test_dirs()
        else:
            self.test_dirs = test_dirs
    
    def discover_test_dirs(self) -> List[str]:
        """自动发现所有测试目录"""
        test_dirs = []
        exclude_dirs = {'.git', '__pycache__', '.vscode', 'venv'}
        
        for item in os.listdir(self.current_dir):
            item_path = self.current_dir / item
            if item_path.is_dir() and item not in exclude_dirs:
                # 检查是否有Makefile
                makefile = item_path / "Makefile"
                if makefile.exists():
                    test_dirs.append(item)
                else:
                    # 如果有其他构建文件也认为是测试目录
                    build_files = list(item_path.glob("*.cpp")) + \
                                 list(item_path.glob("*.cu")) + \
                                 list(item_path.glob("*.hip"))
                    if build_files:
                        test_dirs.append(item)
        
        # 按字母顺序排序
        test_dirs.sort()
        return test_dirs
    
    def run_single_test(self, test_dir: str) -> TestResult:
        """运行单个测试用例"""
        test_path = self.current_dir / test_dir
        print(f"正在运行测试: {test_dir}")
        
        start_time = time.time()
        
        try:
            # 进入测试目录
            original_dir = os.getcwd()
            os.chdir(test_path)
            
            # 运行 make run 命令
            process = subprocess.run(
                ["make", "run"],
                capture_output=True,
                text=True,
                timeout=30,  # 30秒超时
                shell=False
            )
            
            # 计算运行时间
            duration = time.time() - start_time
            
            # 判断测试结果
            if process.returncode == 0:
                status = "pass"
            else:
                status = "fail"
            
            # 返回原始目录
            os.chdir(original_dir)
            
            result = TestResult(
                name=test_dir,
                status=status,
                duration=duration,
                output=process.stdout + process.stderr,
                return_code=process.returncode,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            print(f"  结果: {status.upper()} ({duration:.2f}s)")
            return result
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"  结果: TIMEOUT ({duration:.2f}s)")
            return TestResult(
                name=test_dir,
                status="error",
                duration=duration,
                output=f"测试超时 (超过30秒)",
                return_code=-1,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        except FileNotFoundError:
            duration = time.time() - start_time
            print(f"  结果: ERROR (make命令未找到)")
            return TestResult(
                name=test_dir,
                status="error",
                duration=duration,
                output=f"错误: Makefile不存在或make命令不可用",
                return_code=-1,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        except Exception as e:
            duration = time.time() - start_time
            print(f"  结果: ERROR ({str(e)})")
            return TestResult(
                name=test_dir,
                status="error",
                duration=duration,
                output=f"未知错误: {str(e)}",
                return_code=-1,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
    
    def run_all_tests(self) -> None:
        """运行所有测试用例"""
        print("=" * 60)
        print(f"开始运行HIP测试用例")
        print(f"发现 {len(self.test_dirs)} 个测试目录")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # 逐个运行测试
        for i, test_dir in enumerate(self.test_dirs, 1):
            print(f"\n[{i}/{len(self.test_dirs)}] ", end="")
            result = self.run_single_test(test_dir)
            self.results[test_dir] = result
        
        self.end_time = time.time()
        
        # 打印汇总报告
        self.print_summary()
    
    def print_summary(self) -> None:
        """打印测试结果汇总"""
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r.status == "pass")
        failed = sum(1 for r in self.results.values() if r.status == "fail")
        errors = sum(1 for r in self.results.values() if r.status == "error")
        total_time = self.end_time - self.start_time
        
        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)
        
        # 打印统计信息
        print(f"\n总计: {total} 个测试用例")
        print(f"通过: {passed} ({(passed/total*100 if total>0 else 0):.1f}%)")
        print(f"失败: {failed}")
        print(f"错误: {errors}")
        print(f"总运行时间: {total_time:.2f} 秒")
        
        # 打印详细结果表格
        print(f"\n详细结果:")
        print("-" * 80)
        print(f"{'测试用例':<20} {'状态':<10} {'耗时(秒)':<10} {'返回码':<8}")
        print("-" * 80)
        
        for test_name, result in self.results.items():
            status_display = {
                "pass": "✅ PASS",
                "fail": "❌ FAIL",
                "error": "⚠️ ERROR"
            }.get(result.status, result.status.upper())
            
            print(f"{test_name:<20} {status_display:<10} {result.duration:<10.2f} {result.return_code:<8}")
        
        print("-" * 80)
        
        # 如果有失败的测试，显示详细信息
        failed_tests = {k: v for k, v in self.results.items() if v.status in ["fail", "error"]}
        if failed_tests:
            print(f"\n失败的测试用例 ({len(failed_tests)} 个):")
            for test_name, result in failed_tests.items():
                print(f"\n{'='*40}")
                print(f"测试: {test_name}")
                print(f"状态: {result.status.upper()}")
                print(f"返回码: {result.return_code}")
                print(f"输出摘要:")
                print("-" * 40)
                # 只显示前200个字符的输出
                output_preview = result.output[:200] + ("..." if len(result.output) > 200 else "")
                print(output_preview)
    
    def save_results(self, output_file: str = "test_results.json") -> None:
        """保存测试结果到JSON文件"""
        results_dict = {
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results.values() if r.status == "pass"),
                "failed": sum(1 for r in self.results.values() if r.status == "fail"),
                "errors": sum(1 for r in self.results.values() if r.status == "error"),
                "total_time": self.end_time - self.start_time if self.end_time else 0,
                "timestamp": datetime.now().isoformat()
            },
            "results": {
                name: {
                    "status": result.status,
                    "duration": result.duration,
                    "return_code": result.return_code,
                    "timestamp": result.timestamp,
                    "output": result.output
                }
                for name, result in self.results.items()
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        
        print(f"\n详细结果已保存到: {output_file}")
    
    def generate_report(self, report_file: str = "test_report.txt") -> None:
        """生成文本报告"""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("HIP测试用例运行报告\n")
            f.write("=" * 60 + "\n\n")
            
            total = len(self.results)
            passed = sum(1 for r in self.results.values() if r.status == "pass")
            
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"运行目录: {self.current_dir}\n")
            f.write(f"总计: {total} 个测试用例\n")
            f.write(f"通过: {passed} ({(passed/total*100 if total>0 else 0):.1f}%)\n")
            f.write(f"失败: {total - passed} 个\n\n")
            
            f.write("详细结果:\n")
            f.write("-" * 60 + "\n")
            
            for test_name, result in self.results.items():
                status_icon = "✓" if result.status == "pass" else "✗"
                f.write(f"{status_icon} {test_name:<20} {result.duration:.2f}s\n")
            
            f.write("\n" + "=" * 60 + "\n")


def main():
    """主函数"""
    # 创建测试运行器
    runner = HIPTestRunner()
    
    if not runner.test_dirs:
        print("错误: 未找到任何测试目录！")
        print("确保当前目录包含有Makefile的测试用例目录")
        sys.exit(1)
    
    try:
        # 运行所有测试
        runner.run_all_tests()
        
        # 保存结果
        runner.save_results("hip_test_results.json")
        runner.generate_report("hip_test_report.txt")
        
        # 根据测试结果返回适当的退出码
        passed = sum(1 for r in runner.results.values() if r.status == "pass")
        total = len(runner.results)
        
        if passed == total:
            print(f"\n🎉 所有测试通过！")
            sys.exit(0)
        else:
            print(f"\n⚠️  有 {total - passed} 个测试失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
