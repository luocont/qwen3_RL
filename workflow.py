"""
AI工作流：病例分析师 + 思考回复模型
工作流程：
1. 读取RL.json中的用户输入
2. 5Ps模型分析病例
3. 思考与回复模型处理并回复
4. 更新历史记录，进行下一轮
"""

import json
import os
from openai import OpenAI
from prompts.five_ps_teacher import FIVE_PS_TEACHER_SYSTEM_PROMPT
from prompts.think_and_reply_teacher import REPLY_TEACHER_SYSTEM_PROMPT


class CaseAnalysisWorkflow:
    def __init__(self, api_key=None):
        """初始化工作流"""
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.case_history = ""  # 5Ps病例历史
        self.conversation_history = []  # 对话历史

    def extract_user_inputs(self, rl_json_path):
        """从RL.json中提取所有role为user的content"""
        with open(rl_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        user_inputs = []
        for conversation in data:
            for message in conversation.get('messages', []):
                if message.get('role') == 'user':
                    user_inputs.append(message.get('content'))
        return user_inputs

    def analyze_case_with_5ps(self, user_input):
        """使用5Ps模型分析病例"""
        # 替换提示词中的变量
        system_prompt = FIVE_PS_TEACHER_SYSTEM_PROMPT.replace(
            "{最新的5Ps病例}",
            self.case_history if self.case_history else "（暂无病例记录）"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        completion = self.client.chat.completions.create(
            model="qwen-max",
            messages=messages,
            stream=True
        )

        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content

        # 提取5Ps内容
        if "<5Ps>" in response and "</5Ps>" in response:
            start = response.find("<5Ps>")
            end = response.find("</5Ps>") + len("</5Ps>")
            self.case_history = response[start:end]

        return response

    def think_and_reply(self, user_input):
        """使用思考与回复模型处理"""
        # 替换提示词中的变量
        system_prompt = REPLY_TEACHER_SYSTEM_PROMPT.replace(
            "{previous_5ps_case}",
            self.case_history if self.case_history else "（暂无病例记录）"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        completion = self.client.chat.completions.create(
            model="qwen3-max",
            messages=messages,
            stream=True
        )

        is_answering = False
        thinking_content = ""
        reply_content = ""

        print("\n" + "=" * 20 + "思考过程" + "=" * 20)
        for chunk in completion:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                if not is_answering:
                    print(delta.reasoning_content, end="", flush=True)
                    thinking_content += delta.reasoning_content
            if hasattr(delta, "content") and delta.content:
                if not is_answering:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20)
                    is_answering = True
                print(delta.content, end="", flush=True)
                reply_content += delta.content

        return thinking_content, reply_content

    def update_conversation_history(self, user_input, assistant_reply):
        """更新对话历史"""
        self.conversation_history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_reply}
        ])

    def run_workflow(self, rl_json_path, max_rounds=None):
        """运行完整工作流"""
        user_inputs = self.extract_user_inputs(rl_json_path)

        if max_rounds:
            user_inputs = user_inputs[:max_rounds]

        print(f"共找到 {len(user_inputs)} 条用户输入")
        print("=" * 50)

        for idx, user_input in enumerate(user_inputs, 1):
            print(f"\n\n{'='*20} 第 {idx} 轮对话 {'='*20}")
            print(f"用户输入: {user_input}")

            # Step 1: 5Ps模型分析
            print("\n--- 5Ps病例分析 ---")
            case_analysis = self.analyze_case_with_5ps(user_input)
            print("病例分析完成")

            # Step 2: 思考与回复
            print("\n--- 思考与回复 ---")
            thinking, reply = self.think_and_reply(user_input)

            # Step 3: 更新历史
            self.update_conversation_history(user_input, reply)

            print(f"\n当前5Ps病例:\n{self.case_history}")

    def run_single_input(self, user_input):
        """运行单个输入的工作流"""
        print(f"用户输入: {user_input}")

        # Step 1: 5Ps模型分析
        print("\n--- 5Ps病例分析 ---")
        case_analysis = self.analyze_case_with_5ps(user_input)
        print("病例分析完成")

        # Step 2: 思考与回复
        print("\n--- 思考与回复 ---")
        thinking, reply = self.think_and_reply(user_input)

        # Step 3: 更新历史
        self.update_conversation_history(user_input, reply)

        return case_analysis, thinking, reply

    def save_results(self, output_path):
        """保存结果到文件"""
        results = {
            "case_history": self.case_history,
            "conversation_history": self.conversation_history
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到 {output_path}")


def main():
    """主函数"""
    # 初始化工作流
    workflow = CaseAnalysisWorkflow()

    # 方式1：处理RL.json中的所有用户输入
    rl_json_path = "e:/GitLoadWareHouse/work/智能体/RL.json"

    # 处理前3轮对话（可以调整max_rounds）
    workflow.run_workflow(rl_json_path, max_rounds=3)

    # 保存结果
    output_path = "e:/GitLoadWareHouse/work/智能体/workflow_results.json"
    workflow.save_results(output_path)

    # 方式2：处理单个输入（实时交互）
    # while True:
    #     user_input = input("\n请输入您的问题（输入'quit'退出）: ")
    #     if user_input.lower() == 'quit':
    #         break
    #     workflow.run_single_input(user_input)


if __name__ == "__main__":
    main()
