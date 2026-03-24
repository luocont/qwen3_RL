"""
使用示例脚本
展示工作流的各种使用方式
"""

from workflow import CaseAnalysisWorkflow
import os


def example_1_single_input():
    """示例1：处理单个用户输入"""
    print("=" * 50)
    print("示例1：处理单个用户输入")
    print("=" * 50)

    # 初始化工作流（确保设置了API密钥）
    api_key = os.getenv("DASHSCOPE_API_KEY")
    workflow = CaseAnalysisWorkflow(api_key=api_key)

    # 用户输入
    user_input = "我最近感到很焦虑，晚上睡不着觉"

    # 处理输入
    case_analysis, thinking, reply = workflow.run_single_input(user_input)

    # 查看结果
    print(f"\n5Ps病例:\n{case_analysis}")
    print(f"\n思考内容:\n{thinking}")
    print(f"\n回复内容:\n{reply}")

    # 保存结果
    workflow.save_results("example1_results.json")


def example_2_conversation():
    """示例2：模拟多轮对话"""
    print("\n" + "=" * 50)
    print("示例2：模拟多轮对话")
    print("=" * 50)

    api_key = os.getenv("DASHSCOPE_API_KEY")
    workflow = CaseAnalysisWorkflow(api_key=api_key)

    # 模拟对话
    conversation = [
        "我最近感到很焦虑，晚上睡不着觉",
        "主要是工作压力太大，老板总是对我要求很高",
        "我觉得自己可能做不好这份工作",
        "以前也遇到过类似的情况，但这次感觉特别严重"
    ]

    for idx, user_input in enumerate(conversation, 1):
        print(f"\n第 {idx} 轮")
        print(f"用户: {user_input}")

        # 处理输入
        case_analysis, thinking, reply = workflow.run_single_input(user_input)

        print(f"咨询师: {reply}")

        # 显示当前病例状态
        print(f"\n当前病例完善度:")
        case_lines = workflow.case_history.split('\n')
        for line in case_lines:
            if '待完善' not in line and line.strip():
                print(f"  ✓ {line.strip()}")
            elif line.strip():
                print(f"  ○ {line.strip()}")

    # 保存完整对话结果
    workflow.save_results("example2_conversation.json")


def example_3_custom_config():
    """示例3：自定义配置"""
    print("\n" + "=" * 50)
    print("示例3：自定义配置")
    print("=" * 50)

    # 可以自定义API密钥
    custom_api_key = "your-custom-api-key"

    workflow = CaseAnalysisWorkflow(api_key=custom_api_key)

    # 处理输入
    user_input = "我和家人的关系很紧张"
    case_analysis, thinking, reply = workflow.run_single_input(user_input)

    print(f"回复: {reply}")


def example_4_access_history():
    """示例4：访问历史记录"""
    print("\n" + "=" * 50)
    print("示例4：访问历史记录")
    print("=" * 50)

    api_key = os.getenv("DASHSCOPE_API_KEY")
    workflow = CaseAnalysisWorkflow(api_key=api_key)

    # 进行几轮对话
    inputs = [
        "我感到很沮丧",
        "主要是因为工作不顺心"
    ]

    for user_input in inputs:
        workflow.run_single_input(user_input)

    # 访问历史记录
    print("\n5Ps病例历史:")
    print(workflow.case_history)

    print("\n对话历史:")
    for msg in workflow.conversation_history:
        role = msg['role']
        content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
        print(f"{role}: {content}")


def example_5_error_handling():
    """示例5：错误处理"""
    print("\n" + "=" * 50)
    print("示例5：错误处理")
    print("=" * 50)

    try:
        # 使用无效的API密钥
        workflow = CaseAnalysisWorkflow(api_key="invalid-key")

        # 尝试处理输入
        workflow.run_single_input("测试输入")

    except Exception as e:
        print(f"捕获到错误: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("\n建议：检查API密钥是否正确")


def example_6_json_processing():
    """示例6：处理JSON数据文件"""
    print("\n" + "=" * 50)
    print("示例6：处理JSON数据文件")
    print("=" * 50)

    api_key = os.getenv("DASHSCOPE_API_KEY")
    workflow = CaseAnalysisWorkflow(api_key=api_key)

    # 提取用户输入
    rl_json_path = "e:/GitLoadWareHouse/work/智能体/RL.json"

    try:
        user_inputs = workflow.extract_user_inputs(rl_json_path)
        print(f"成功提取 {len(user_inputs)} 条用户输入")

        # 显示前5条
        print("\n前5条用户输入:")
        for i, user_input in enumerate(user_inputs[:5], 1):
            print(f"{i}. {user_input[:60]}...")

        # 处理前3条
        print("\n开始处理前3条...")
        for i, user_input in enumerate(user_inputs[:3], 1):
            print(f"\n处理第 {i} 条...")
            workflow.run_single_input(user_input)

        # 保存结果
        workflow.save_results("example6_json_processing.json")

    except FileNotFoundError:
        print("错误：找不到RL.json文件")
    except Exception as e:
        print(f"处理失败: {str(e)}")


def main():
    """运行所有示例"""
    print("AI病例分析工作流 - 使用示例")
    print("=" * 50)
    print("\n注意：运行示例前请确保已设置API密钥")
    print("可以通过环境变量设置: export DASHSCOPE_API_KEY='your-key'\n")

    examples = [
        ("单个用户输入处理", example_1_single_input),
        ("多轮对话模拟", example_2_conversation),
        ("自定义配置", example_3_custom_config),
        ("访问历史记录", example_4_access_history),
        ("错误处理", example_5_error_handling),
        ("JSON数据处理", example_6_json_processing)
    ]

    print("可用示例:")
    for idx, (name, _) in enumerate(examples, 1):
        print(f"{idx}. {name}")

    print("0. 退出")

    while True:
        try:
            choice = input("\n选择要运行的示例 (0-6): ").strip()

            if choice == '0':
                break
            elif choice.isdigit() and 1 <= int(choice) <= len(examples):
                name, func = examples[int(choice) - 1]
                print(f"\n运行示例: {name}")
                func()
                print("\n示例运行完成！")
                break
            else:
                print("无效选项，请重新输入")

        except KeyboardInterrupt:
            print("\n\n程序已中断")
            break
        except Exception as e:
            print(f"\n示例运行失败: {str(e)}")
            print("请确保API密钥已正确设置")


if __name__ == "__main__":
    main()
