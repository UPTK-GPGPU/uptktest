import subprocess
import sys
import os
import re

def run_ctest_for_type(lib_type):
    json_file = f"{lib_type}.json"
    test_dir = os.path.abspath(os.path.join(os.getcwd(), "build", f"{lib_type}_test"))

    if not os.path.exists(json_file):
        print(f"❌ 未找到函数列表：{json_file}")
        return
    if not os.path.exists(test_dir):
        print(f"❌ 测试目录不存在：{test_dir}")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        raw = f.read()

    func_list = re.findall(r'"([a-zA-Z0-9_]+)"', raw)
    func_list = list(filter(None, func_list))
    total_api = len(func_list)

    current_idx = 0
    found_cases = 0
    passed_cases = 0
    failed_cases = 0
    skipped_cases = 0

    print(f"\n🚀 开始测试 {lib_type.upper()} 接口")
    print(f"📂 测试目录：{test_dir}")
    print(f"📊 总接口数：{total_api}\n")

    for func in func_list:
        current_idx += 1
        test_name = f"test_{func}"

        cmd = ["ctest", "-R", f"^{test_name}$"]
        res = subprocess.run(
            cmd,
            cwd=test_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        output = res.stdout

        if "No tests" in output or "No such test" in output:
            print(f"[{current_idx}/{total_api}] ⚠️ {test_name} (不支持，不计入)")
            continue

        found_cases += 1
        if "Passed" in output or "100% tests passed" in output:
            passed_cases += 1
            print(f"[{current_idx}/{total_api}] ✅ {test_name} (通过)")
        elif "Failed" in output:
            failed_cases += 1
            print(f"[{current_idx}/{total_api}] ❌ {test_name} (失败)")
        elif "SKIP" in output or "Skip" in output:
            skipped_cases += 1
            print(f"[{current_idx}/{total_api}] ⏭️ {test_name} (跳过)")
        else:
            failed_cases += 1
            print(f"[{current_idx}/{total_api}] ❌ {test_name} (异常)")

    coverage = (found_cases / total_api * 100) if total_api > 0 else 0.0
    pass_rate = (passed_cases / found_cases * 100) if found_cases > 0 else 0.0

    print("\n" + "=" * 80)
    print(f"                          {lib_type.upper()} 测试报告")
    print("=" * 80)
    print(f"总接口数量        : {total_api}")
    print(f"实际找到用例      : {found_cases}")
    print(f"用例通过数量      : {passed_cases}")
    print(f"用例失败数量      : {failed_cases}")
    print(f"用例跳过数量      : {skipped_cases}")
    print(f"接口覆盖率        : {coverage:.2f}% (支持/总数)")
    print(f"用例通过率        : {pass_rate:.2f}% (通过/支持)")
    print("=" * 80)

def run_all():
    run_ctest_for_type("cuda")
    run_ctest_for_type("cublas")
    run_ctest_for_type("nccl")
    run_ctest_for_type("cufft")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  python test.py cuda")
        print("  python test.py cublas")
        print("  python test.py nccl")
        print("  python test.py cufft")
        print("  python test.py all")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "all":
        run_all()
    elif mode in ["cuda", "cublas", "nccl", "cufft"]:
        run_ctest_for_type(mode)
    else:
        print("不支持的类型！")
        sys.exit(1)
