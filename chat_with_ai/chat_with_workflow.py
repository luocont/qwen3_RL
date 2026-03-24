"""
使用AI工作流生成心理咨询对话

特点：
- 保留 Client Agent 的原有逻辑（使用框架预设提示词）
- 使用 AI 工作流（5Ps病例分析 + 思考回复模型）作为 Counselor Agent
- 支持自定义系统提示词
- 生成与原始格式一致的 session_*.json 文件

使用示例：
    python chat_with_workflow.py --api_key your-key --output_dir ../output-workflow
"""

import argparse
import json
import multiprocessing
import traceback
import os
import sys
from pathlib import Path

# 添加工作流目录到路径
workflow_dir = Path(__file__).parent.parent
sys.path.insert(0, str(workflow_dir))

from workflow import CaseAnalysisWorkflow


class PromptTemplate:
    """简单的提示词模板类"""
    def __init__(self, input_variables, template):
        self.input_variables = input_variables
        self.template = template

    def format(self, **kwargs):
        result = self.template
        for var in self.input_variables:
            if var in kwargs:
                result = result.replace(f"{{{var}}}", str(kwargs[var]))
        return result


# 导入通用 LLM 客户端（用于 Client Agent）
try:
    from llm_client import create_client_from_env
except ImportError:
    print("警告: 无法导入 llm_client，Client Agent 将无法工作")
    create_client_from_env = None


# ============================================
# 配置
# ============================================
# 数据文件路径（相对于脚本所在目录）
SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR / "data.json"
PROMPTS_DIR = SCRIPT_DIR  # 提示词文件在同一目录

# 进程级全局变量
_process_workflow = None
_process_client_llm = None


def load_env_file():
    """加载 .env 文件"""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        print(f"加载环境变量: {env_file}", flush=True)
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value
        print("环境变量加载完成", flush=True)
    else:
        print("警告: 未找到 .env 文件，Client Agent 将无法工作", flush=True)


def generate_with_api(prompt: str) -> str:
    """使用 API 生成响应（用于 Client Agent）"""
    global _process_client_llm
    if _process_client_llm is None:
        if create_client_from_env is None:
            raise Exception("llm_client 模块不可用")
        try:
            _process_client_llm = create_client_from_env()
        except Exception as e:
            print(f"错误: 无法创建 LLM 客户端: {e}")
            print("请确保 .env 文件配置正确")
            raise
    response = _process_client_llm.completion(prompt=prompt)
    return response.choices[0].message.content


def translate_to_chinese(text: str) -> str:
    """将英文文本翻译成中文"""
    import re
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    total_chars = len(text.strip())

    if total_chars == 0 or english_chars / total_chars < 0.3:
        return text

    try:
        prompt = f"""请将以下英文文本翻译成中文，保持对话的自然和口语化：

{text}

请只输出翻译结果，不要添加其他解释。"""
        translation = generate_with_api(prompt)
        return translation.strip()
    except Exception as e:
        print(f"[警告] 翻译失败，使用原文: {e}")
        return text


# ============================================
# Client Agent（保持不变）
# ============================================
class ClientAgent:
    """来访者智能体（使用框架原有逻辑和提示词）"""

    def __init__(self, example):
        self.example = example
        self._load_prompt()

    def _load_prompt(self):
        """加载 Client Agent 提示词"""
        prompt_path = Path(__file__).parent / "agent_client.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"找不到 Client Agent 提示词文件: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()

        self.attitude = (
            f"{self.example['AI_client']['attitude']}: "
            f"{self.example['AI_client']['attitude_instruction']}"
        )
        self.prompt_template = PromptTemplate(
            input_variables=["intake_form", "attitude", "history"],
            template=prompt_text
        )

    def generate(self, history):
        """生成来访者响应"""
        history_text = '\n'.join([
            f"{message['role'].capitalize()}: {message['message']}"
            for message in history
        ])

        prompt = self.prompt_template.format(
            intake_form=self.example,
            attitude=self.attitude,
            history=history_text
        )

        return generate_with_api(prompt)


# ============================================
# Workflow Counselor Agent（使用AI工作流）
# ============================================
class WorkflowCounselorAgent:
    """使用AI工作流的咨询师智能体"""

    def __init__(self, example, api_key=None):
        """
        初始化工作流咨询师

        Args:
            example: 数据样本
            api_key: 阿里云百炼API密钥
        """
        self.example = example
        # 为每个会话创建独立的工作流实例
        self.workflow = CaseAnalysisWorkflow(api_key=api_key)
        # 存储最近一次的5Ps分析
        self.last_case_analysis = None

    def get_case_history(self):
        """获取当前的5Ps病例分析历史"""
        return self.workflow.case_history

    def get_last_case_analysis(self):
        """获取最近一次的5Ps病例分析"""
        return self.last_case_analysis

    def generate_case_analysis(self, history):
        """
        生成当前轮次的5Ps病例分析（调用5Ps模型）

        Args:
            history: 对话历史列表

        Returns:
            5Ps病例分析字符串
        """
        # 从历史中提取最后一条client消息作为输入
        if not history:
            return "<5Ps>\nP1 主诉：初始咨询\nP2-P5：待进一步收集信息\n</5Ps>"

        # 获取最后一条client消息
        last_client_message = None
        for msg in reversed(history):
            if msg['role'] == 'client':
                last_client_message = msg['message']
                break

        if not last_client_message:
            return "<5Ps>\nP1 主诉：待确认\n</5Ps>"

        try:
            # 调用5Ps模型（只生成5Ps，不生成回复）
            case_analysis = self.workflow.analyze_case_with_5ps(last_client_message)

            # 存储最近一次的5Ps分析
            self.last_case_analysis = case_analysis

            # 打印5Ps病例分析
            print(f"\n    📊 5Ps病例分析:", flush=True)
            print(f"    {case_analysis}", flush=True)
            print(f"    [DEBUG] 5Ps已存储 (长度={len(case_analysis)})", flush=True)

            return case_analysis

        except Exception as e:
            print(f"[错误] 5Ps分析生成失败: {e}")
            return "<5Ps>\nP1 主诉：分析失败，请重试\n</5Ps>"

    def generate_reply(self, history):
        """
        基于5Ps和历史对话生成咨询师回复（调用思考及回复模型）

        Args:
            history: 对话历史列表（应包含当前轮次的5Ps）

        Returns:
            咨询师的回复内容
        """
        try:
            # 获取最后一条client消息作为输入
            if not history:
                return "你好，我是心理咨询师，很高兴能和你交流。请告诉我你最近遇到的问题。"

            # 获取最后一条client消息
            last_client_message = None
            for msg in reversed(history):
                if msg['role'] == 'client':
                    last_client_message = msg['message']
                    break

            if not last_client_message:
                return "请告诉我你最近遇到的问题。"

            # 调用思考及回复模型（会使用当前5Ps，因为已经在workflow中更新了case_history）
            thinking, reply = self.workflow.think_and_reply(last_client_message)
            return reply

        except Exception as e:
            print(f"[错误] 回复生成失败: {e}")
            return "我理解你的感受。能告诉我更多关于这个情况的细节吗？"

    def generate(self, history):
        """
        生成咨询师响应（兼容旧接口，内部会先生成5Ps再生成回复）

        Args:
            history: 对话历史列表

        Returns:
            咨询师的回复内容
        """
        # 先生成5Ps
        self.generate_case_analysis(history)
        # 再生成回复
        return self.generate_reply(history)


# ============================================
# Therapy Session（使用工作流）
# ============================================
class WorkflowTherapySession:
    """使用AI工作流的咨询会话"""

    def __init__(
        self,
        example,
        max_turns: int,
        counselor_agent,
        client_agent,
        output_file: str = None
    ):
        """
        初始化会话

        Args:
            example: 数据样本
            max_turns: 最大对话轮数
            counselor_agent: WorkflowCounselorAgent 实例
            client_agent: ClientAgent 实例
            output_file: 输出文件路径（如果提供，每轮对话后会自动保存）
        """
        self.example = example
        self.max_turns = max_turns
        self.history = []
        self.counselor_agent = counselor_agent
        self.client_agent = client_agent
        self.output_file = output_file

    def _add_to_history(self, role: str, message: str, five_ps: str = None):
        """添加消息到历史"""
        history_entry = {"role": role, "message": message}
        if five_ps is not None:
            history_entry["5Ps"] = five_ps
            print(f"    [DEBUG] 添加5Ps到历史记录 (role={role}, 5Ps长度={len(five_ps)})", flush=True)
        self.history.append(history_entry)

    def _save_to_file(self, turn: int = None):
        """保存当前会话状态到文件"""
        if not self.output_file:
            return

        try:
            print(f"    [DEBUG] 开始保存文件...", flush=True)

            # 创建可序列化的example副本
            serializable_example = {}
            for key, value in self.example.items():
                if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                    serializable_example[key] = value
                else:
                    # 如果是不可序列化的对象，转换为字符串
                    serializable_example[key] = str(value)

            output_data = {
                "example": serializable_example,
                "cbt_technique": "AI Workflow (5Ps Case Analysis + Think & Reply)",
                "cbt_plan": "AI工作流: 基于阿里云百炼API，使用5Ps病例分析模型和思考回复模型",
                "cost": 0,
                "history": self.history
            }

            print(f"    [DEBUG] 准备写入文件: {self.output_file}", flush=True)

            # 直接写入文件（使用更可靠的方式）
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
                f.flush()  # 强制刷新缓冲区
                import os
                os.fsync(f.fileno())  # 强制写入磁盘

            print(f"    [DEBUG] 文件写入完成", flush=True)
            # 验证文件是否真的写入成功
            if not Path(self.output_file).exists():
                print(f"    [错误] 文件写入验证失败！", flush=True)

            if turn is not None:
                print(f"    💾 已保存到文件 (轮次 {turn})", flush=True)
            else:
                print(f"    💾 已保存到文件", flush=True)

        except Exception as e:
            print(f"    [错误] 保存文件失败: {e}", flush=True)
            import traceback
            traceback.print_exc()

    def _initialize_session(self):
        """初始化会话"""
        example_cbt = self.example['AI_counselor']['CBT']

        # 翻译前两句历史对话（英文 -> 中文）
        counselor_init = example_cbt['init_history_counselor']
        client_init = example_cbt['init_history_client']

        print("\n[翻译] 正在翻译初始对话历史...", flush=True)
        counselor_init_cn = translate_to_chinese(counselor_init)
        client_init_cn = translate_to_chinese(client_init)

        if counselor_init_cn != counselor_init:
            print(f"[翻译] 咨询师开场白: {counselor_init[:50]}... -> {counselor_init_cn[:50]}...", flush=True)
        if client_init_cn != client_init:
            print(f"[翻译] 来访者回复: {client_init[:50]}... -> {client_init_cn[:50]}...", flush=True)

        # 初始对话没有5Ps分析
        self._add_to_history("counselor", counselor_init_cn)
        self._add_to_history("client", client_init_cn)

        # 保存初始状态
        self._save_to_file(turn=0)

    def _check_goodbye(self, text: str) -> bool:
        """检测是否包含道别词汇"""
        goodbye_keywords = ['再见', '拜拜', '下次见', '再会', '回见']
        return any(keyword in text for keyword in goodbye_keywords)

    def _exchange_statements(self):
        """交替生成对话"""
        for turn in range(self.max_turns):
            print(f"\n    轮次 {turn + 1}/{self.max_turns}", flush=True)
            print("=" * 60, flush=True)

            # 第一步：先生成本轮的5Ps病例分析（基于历史）
            current_5ps = self.counselor_agent.generate_case_analysis(self.history)

            # 第二步：将5Ps添加到历史（这样咨询师生成回复时能看到）
            # 注意：此时还没有咨询师的回复，我们添加一个占位符
            temp_history_entry = {"role": "counselor", "message": "", "5Ps": current_5ps}
            self.history.append(temp_history_entry)

            # 第三步：咨询师基于包含当前5Ps的历史生成回复
            counselor_response = self.counselor_agent.generate_reply(self.history)

            # 更新历史记录中的咨询师回复
            self.history[-1]["message"] = counselor_response

            print(f"    📋 咨询师:\n{counselor_response}", flush=True)

            # 来访者回应（使用框架原有 Client Agent）
            try:
                client_response = self.client_agent.generate(self.history)
                client_response = client_response.replace('Client: ', '')
            except Exception as e:
                print(f"[错误] Client Agent 生成失败: {e}")
                client_response = "我明白了，谢谢你的建议。"

            # 检测Client Agent的结束标志（[/END]或道别词汇）
            # 如果Client Agent输出了这些标志，说明对话应该结束
            has_end_marker = '[/END]' in client_response
            has_goodbye = self._check_goodbye(client_response)

            # 移除[/END]标记（保留道别词汇）
            client_response = client_response.replace('[/END]', '')

            # 来访者不需要5Ps分析，传入None
            self._add_to_history("client", client_response, None)
            print(f"\n    📋 来访者:\n{client_response}", flush=True)

            # 保存当前轮次的对话到文件
            self._save_to_file(turn=turn + 1)

            # 如果Client Agent输出了结束标志，结束对话
            if has_end_marker or has_goodbye:
                print("\n    ✓ Client Agent输出结束标志，会话结束", flush=True)
                print("=" * 60, flush=True)
                break

    def run_session(self):
        """运行完整会话"""
        self._initialize_session()
        self._exchange_statements()

        # 最终保存一次
        self._save_to_file()

        return {
            "example": self.example,
            "cbt_technique": "AI Workflow (5Ps Case Analysis + Think & Reply)",
            "cbt_plan": "AI工作流: 基于阿里云百炼API，使用5Ps病例分析模型和思考回复模型",
            "cost": 0,
            "history": self.history
        }


# ============================================
# 处理单个任务
# ============================================
def process_task(
    index: int,
    example: dict,
    api_key: str,
    max_turns: int,
    output_dir: Path,
    total: int
):
    """处理单个任务"""
    file_number = index + 1
    try:
        print(f"\n[{file_number}/{total}] 开始处理", flush=True)

        # 创建 Counselor Agent（使用工作流）
        counselor_agent = WorkflowCounselorAgent(example=example, api_key=api_key)

        # 创建 Client Agent
        client_agent = ClientAgent(example=example)

        # 创建会话
        file_name = f"session_{file_number}.json"
        file_path = output_dir / file_name

        session = WorkflowTherapySession(
            example=example,
            max_turns=max_turns,
            counselor_agent=counselor_agent,
            client_agent=client_agent,
            output_file=str(file_path)  # 传入输出文件路径
        )

        session.run_session()

        print(f"[{file_number}/{total}] 完成，保存到 {file_name}", flush=True)

    except Exception as e:
        error_file_name = f"error_{file_number}.txt"
        error_file_path = output_dir / error_file_name

        with open(error_file_path, "w", encoding="utf-8") as f:
            f.write(f"Error: {e}\n\n")
            f.write(traceback.format_exc())

        print(f"[{file_number}/{total}] 失败，错误已保存到 {error_file_name}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="使用AI工作流生成心理咨询对话",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 使用配置文件中的API密钥
  python chat_with_workflow.py

  # 指定API密钥
  python chat_with_workflow.py --api_key your-key

  # 限制处理样本数量
  python chat_with_workflow.py --num_samples 3
        """
    )

    parser.add_argument("--api_key", type=str, help="阿里云百炼API密钥")
    parser.add_argument("--output_dir", type=str, default="../output-workflow", help="输出目录")
    parser.add_argument("--max_turns", type=int, default=20, help="最大对话轮数")
    parser.add_argument("--num_samples", type=int, help="处理样本数量（默认处理全部）")

    args = parser.parse_args()

    # 加载环境变量
    load_env_file()

    # 确定API密钥
    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 请设置API密钥")
        print("  方式1: --api_key your-key")
        print("  方式2: 设置环境变量 DASHSCOPE_API_KEY")
        print("  方式3: 在 .env 文件中配置")
        return

    # 加载数据
    data_file = Path(DATA_FILE)
    if not data_file.exists():
        print(f"错误: 数据文件不存在: {data_file}")
        return

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 限制样本数量
    if args.num_samples:
        data = data[:args.num_samples]
        print(f"处理前 {args.num_samples} 个样本", flush=True)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打印配置信息
    print("\n" + "=" * 60, flush=True)
    print("配置信息:", flush=True)
    print("=" * 60, flush=True)
    print(f"  咨询师模型: AI工作流 (5Ps + 思考回复)", flush=True)
    print(f"  最大轮数: {args.max_turns}", flush=True)
    print(f"  输出目录: {output_dir}", flush=True)
    print(f"  数据样本数: {len(data)}", flush=True)
    print("=" * 60 + "\n", flush=True)

    # 处理所有任务
    total = len(data)
    for index, example in enumerate(data):
        process_task(
            index=index,
            example=example,
            api_key=api_key,
            max_turns=args.max_turns,
            output_dir=output_dir,
            total=total
        )

    print("\n" + "=" * 60, flush=True)
    print(f"完成！共处理 {len(data)} 个会话", flush=True)
    print(f"结果保存在: {output_dir}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
