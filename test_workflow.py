"""
工作流测试脚本
用于快速验证工作流功能
"""

import os
from workflow import CaseAnalysisWorkflow


def test_single_round():
    """测试单轮对话"""
    print("=" * 50)
    print("测试单轮对话")
    print("=" * 50)

    # 使用环境变量或手动设置API密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("警告：未设置API密钥，请设置 DASHSCOPE_API_KEY 环境变量")
        return

    workflow = CaseAnalysisWorkflow(api_key=api_key)

    # 测试用例
    test_input = "我最近感到很焦虑，晚上睡不着觉"
    print(f"测试输入: {test_input}\n")

    try:
        case_analysis, thinking, reply = workflow.run_single_input(test_input)

        print("\n" + "=" * 50)
        print("测试结果")
        print("=" * 50)
        print(f"5Ps病例:\n{case_analysis}\n")
        print(f"思考内容:\n{thinking}\n")
        print(f"回复内容:\n{reply}\n")

        # 保存结果
        workflow.save_results("test_results.json")
        print("测试完成！")

    except Exception as e:
        print(f"测试失败: {str(e)}")


def test_multiple_rounds():
    """测试多轮对话"""
    print("\n" + "=" * 50)
    print("测试多轮对话")
    print("=" * 50)

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("警告：未设置API密钥")
        return

    workflow = CaseAnalysisWorkflow(api_key=api_key)

    # 模拟多轮对话
    test_inputs = [
        "我最近感到很焦虑，晚上睡不着觉",
        "主要是工作压力太大，老板总是对我要求很高",
        "我觉得自己可能做不好这份工作"
    ]

    for idx, user_input in enumerate(test_inputs, 1):
        print(f"\n第 {idx} 轮对话")
        print("=" * 30)
        print(f"用户: {user_input}")

        try:
            case_analysis, thinking, reply = workflow.run_single_input(user_input)
            print(f"\n当前病例:\n{workflow.case_history}\n")

        except Exception as e:
            print(f"第 {idx} 轮对话失败: {str(e)}")
            break

    workflow.save_results("test_multiple_rounds_results.json")
    print("\n多轮对话测试完成！")


def test_json_extraction():
    """测试JSON数据提取"""
    print("\n" + "=" * 50)
    print("测试JSON数据提取")
    print("=" * 50)

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("警告：未设置API密钥")
        return

    workflow = CaseAnalysisWorkflow(api_key=api_key)

    try:
        # 提取用户输入
        rl_json_path = "e:/GitLoadWareHouse/work/智能体/RL.json"
        user_inputs = workflow.extract_user_inputs(rl_json_path)

        print(f"成功提取 {len(user_inputs)} 条用户输入")
        print("\n前5条用户输入:")
        for i, user_input in enumerate(user_inputs[:5], 1):
            print(f"{i}. {user_input}")

    except Exception as e:
        print(f"提取失败: {str(e)}")


if __name__ == "__main__":
    print("AI工作流测试")
    print("=" * 50)

    # 检查API密钥
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("\n请先设置环境变量:")
        print("export DASHSCOPE_API_KEY='your-api-key'")
        print("\n或在config.py中直接设置API密钥")
    else:
        # 运行测试
        print("开始测试...\n")

        # 测试1: 单轮对话
        test_single_round()

        # 测试2: JSON数据提取
        test_json_extraction()

        # 测试3: 多轮对话（需要较长时间，可选）
        # test_multiple_rounds()
