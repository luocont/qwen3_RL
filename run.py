"""
快速启动脚本
简单易用的工作流启动器
"""

import os
import sys
from workflow import CaseAnalysisWorkflow
from config import DASHSCOPE_API_KEY, PATHS


def check_api_key():
    """检查API密钥配置"""
    api_key = os.getenv("DASHSCOPE_API_KEY") or DASHSCOPE_API_KEY

    if api_key == "sk-your-api-key-here" or not api_key:
        print("\n" + "=" * 50)
        print("错误：未配置API密钥")
        print("=" * 50)
        print("\n请选择以下方式之一配置API密钥:\n")
        print("1. 设置环境变量:")
        print("   export DASHSCOPE_API_KEY='your-api-key'")
        print("\n2. 修改 config.py 文件:")
        print("   DASHSCOPE_API_KEY = 'sk-your-actual-api-key'")
        print("\n3. 在运行时输入:")
        print("   请输入API密钥: ", end="")
        api_key = input().strip()
        if api_key:
            return api_key
        return None

    return api_key


def interactive_mode():
    """交互式对话模式"""
    print("\n" + "=" * 50)
    print("交互式对话模式")
    print("=" * 50)
    print("输入 'quit' 或 'exit' 退出\n")

    api_key = check_api_key()
    if not api_key:
        return

    workflow = CaseAnalysisWorkflow(api_key=api_key)

    while True:
        try:
            user_input = input("\n您: ").strip()

            if user_input.lower() in ['quit', 'exit', '退出']:
                print("\n感谢使用，再见！")
                break

            if not user_input:
                continue

            print("\n" + "=" * 50)
            case_analysis, thinking, reply = workflow.run_single_input(user_input)

            # 显示简化输出
            print("\n--- 咨询师回复 ---")
            print(reply)

        except KeyboardInterrupt:
            print("\n\n检测到中断，退出...")
            break
        except Exception as e:
            print(f"\n发生错误: {str(e)}")


def batch_mode():
    """批处理模式"""
    print("\n" + "=" * 50)
    print("批处理模式")
    print("=" * 50)

    api_key = check_api_key()
    if not api_key:
        return

    # 询问处理轮数
    print("\n请输入要处理的轮数（留空处理全部）: ", end="")
    max_rounds_input = input().strip()
    max_rounds = int(max_rounds_input) if max_rounds_input else None

    workflow = CaseAnalysisWorkflow(api_key=api_key)

    rl_json_path = PATHS['rl_json']
    output_path = PATHS['output']

    print(f"\n开始处理 {rl_json_path}...")
    workflow.run_workflow(rl_json_path, max_rounds=max_rounds)
    workflow.save_results(output_path)

    print(f"\n处理完成！结果已保存到 {output_path}")


def quick_test():
    """快速测试模式"""
    print("\n" + "=" * 50)
    print("快速测试模式")
    print("=" * 50)

    api_key = check_api_key()
    if not api_key:
        return

    workflow = CaseAnalysisWorkflow(api_key=api_key)

    test_message = "我最近感到很焦虑，晚上睡不着觉"
    print(f"\n测试输入: {test_message}")

    case_analysis, thinking, reply = workflow.run_single_input(test_message)

    print("\n" + "=" * 50)
    print("测试结果")
    print("=" * 50)
    print(f"\n5Ps病例:\n{case_analysis}")
    print(f"\n思考内容:\n{thinking}")
    print(f"\n回复内容:\n{reply}")

    workflow.save_results("quick_test_results.json")
    print("\n测试完成！结果已保存到 quick_test_results.json")


def main():
    print("\n" + "=" * 50)
    print("AI病例分析工作流")
    print("=" * 50)

    print("\n请选择模式:")
    print("1. 交互式对话 (实时对话)")
    print("2. 批处理 (处理RL.json)")
    print("3. 快速测试 (单次测试)")
    print("0. 退出")

    while True:
        try:
            choice = input("\n请输入选项 (0-3): ").strip()

            if choice == '0':
                print("\n再见！")
                break
            elif choice == '1':
                interactive_mode()
                break
            elif choice == '2':
                batch_mode()
                break
            elif choice == '3':
                quick_test()
                break
            else:
                print("无效选项，请重新输入")

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n发生错误: {str(e)}")


if __name__ == "__main__":
    main()
