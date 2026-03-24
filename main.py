"""
AI工作流主程序
支持多种运行模式：批处理、交互式、单次处理
"""

import argparse
import os
from workflow import CaseAnalysisWorkflow
from config import DASHSCOPE_API_KEY, PATHS, MODELS


def batch_mode(workflow, rl_json_path, max_rounds=None, output_path=None):
    """批处理模式：处理RL.json中的用户输入"""
    print("=" * 50)
    print("批处理模式")
    print("=" * 50)

    workflow.run_workflow(rl_json_path, max_rounds=max_rounds)

    if output_path:
        workflow.save_results(output_path)


def interactive_mode(workflow):
    """交互式模式：实时对话"""
    print("=" * 50)
    print("交互式模式（输入'quit'退出）")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n您: ").strip()
            if user_input.lower() in ['quit', 'exit', '退出']:
                break

            if not user_input:
                continue

            print("\n" + "=" * 50)
            workflow.run_single_input(user_input)

        except KeyboardInterrupt:
            print("\n检测到中断，退出...")
            break


def single_mode(workflow, user_input, output_path=None):
    """单次处理模式"""
    print("=" * 50)
    print("单次处理模式")
    print("=" * 50)

    case_analysis, thinking, reply = workflow.run_single_input(user_input)

    print("\n" + "=" * 20 + "总结" + "=" * 20)
    print(f"病例分析:\n{case_analysis}")
    print(f"\n思考内容:\n{thinking}")
    print(f"\n回复内容:\n{reply}")

    if output_path:
        workflow.save_results(output_path)


def main():
    parser = argparse.ArgumentParser(description='AI病例分析工作流')
    parser.add_argument(
        '--mode',
        choices=['batch', 'interactive', 'single'],
        default='batch',
        help='运行模式：batch(批处理), interactive(交互式), single(单次处理)'
    )
    parser.add_argument(
        '--input',
        default=PATHS['rl_json'],
        help='RL.json文件路径（批处理模式）'
    )
    parser.add_argument(
        '--output',
        default=PATHS['output'],
        help='输出文件路径'
    )
    parser.add_argument(
        '--max-rounds',
        type=int,
        default=None,
        help='批处理模式下处理的最大轮数'
    )
    parser.add_argument(
        '--message',
        type=str,
        help='单次处理模式下的用户输入'
    )
    parser.add_argument(
        '--api-key',
        default=None,
        help='阿里云百炼API密钥（默认使用环境变量）'
    )

    args = parser.parse_args()

    # 确定API密钥
    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY") or DASHSCOPE_API_KEY

    if api_key == "sk-your-api-key-here":
        print("错误：请设置API密钥（通过环境变量 DASHSCOPE_API_KEY 或 --api-key 参数）")
        return

    # 初始化工作流
    workflow = CaseAnalysisWorkflow(api_key=api_key)

    # 根据模式执行
    if args.mode == 'batch':
        batch_mode(workflow, args.input, args.max_rounds, args.output)

    elif args.mode == 'interactive':
        interactive_mode(workflow)

    elif args.mode == 'single':
        if not args.message:
            print("错误：单次处理模式需要提供 --message 参数")
            return
        single_mode(workflow, args.message, args.output)


if __name__ == "__main__":
    main()
