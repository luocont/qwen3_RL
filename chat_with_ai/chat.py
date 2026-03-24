"""
使用三联动模型生成心理咨询对话

特点：
- 保留 Client Agent 的原有逻辑（使用框架预设提示词）
- 使用三联动模型（情感分析+顾问模型+主模型）作为 Counselor Agent
- 支持自定义系统提示词
- 支持多 GPU 并行推理（每个 GPU 加载一次模型，持久化处理）
- 生成与原始格式一致的 session_*.json 文件

使用示例：
    # 使用配置文件
    python inference-triple-custom.py --config ../config_triple.json

    # 使用命令行参数
    python inference-triple-custom.py --sentiment_model /root/autodl-tmp/xinliyisheng/AruoraCounsel-Sense/new_mlp.bin --primary_model /root/autodl-tmp/xinliyisheng/AuroraCounsel-Guide --consultant_model /root/autodl-tmp/xinliyisheng/AuroraCounsel-Reflect

    # 使用多 GPU 并行（每个 GPU 一个进程）
    python inference-triple-custom.py --num_processes 6 --device_ids "0,1,2,3,4,5" ...
"""

import argparse
import json
import multiprocessing
import traceback
import os
from pathlib import Path


class PromptTemplate:
    """简单的提示词模板类（替代 langchain.prompts.PromptTemplate）"""

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
from llm_client import create_client_from_env

# 导入三联动模型 Agent
from triple_model_agent import (
    TripleModelAgent,
    get_triple_model_preset_prompt,
    preload_models_for_device
)


# ============================================
# 配置
# ============================================
DATA_FILE = "../dataset/data.json"
PROMPTS_DIR = "../prompts/cn/"

# 进程级全局变量（每个进程独立一份）
# 这些变量在每个进程中只初始化一次
_process_counselor_agent = None
_process_client_llm = None
_process_device_id = None


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
        try:
            _process_client_llm = create_client_from_env()
        except Exception as e:
            print(f"错误: 无法创建 LLM 客户端: {e}")
            print("请确保 .env 文件配置正确")
            raise
    response = _process_client_llm.completion(prompt=prompt)
    return response.choices[0].message.content


def translate_to_chinese(text: str) -> str:
    """
    将英文文本翻译成中文

    简单检测：如果文本主要是英文，则返回翻译后的中文
    如果已经是中文，直接返回
    """
    # 简单检测是否为英文（统计英文字母比例）
    import re
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    total_chars = len(text.strip())

    # 如果英文字母比例低于 30%，认为已经是中文
    if total_chars == 0 or english_chars / total_chars < 0.3:
        return text

    # 使用 API 翻译
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
# 原有 Client Agent（保持不变）
# ============================================
class ClientAgent:
    """来访者智能体（使用框架原有逻辑和提示词）"""

    def __init__(self, example):
        self.example = example
        self._load_prompt()

    def _load_prompt(self):
        """加载 Client Agent 提示词"""
        prompt_path = Path(PROMPTS_DIR) / "agent_client.txt"
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
# Therapy Session（使用持久化 Agent）
# ============================================
class TripleModelTherapySession:
    """使用三联动模型的咨询会话（使用进程内持久化 Agent）"""

    def __init__(
        self,
        example,
        max_turns: int,
        counselor_agent,
        client_agent
    ):
        """
        初始化会话（使用已加载的 Agent）

        Args:
            example: 数据样本
            max_turns: 最大对话轮数
            counselor_agent: 已初始化的 TripleModelAgent
            client_agent: 已初始化的 ClientAgent
        """
        self.example = example
        self.max_turns = max_turns
        self.history = []
        self.counselor_agent = counselor_agent
        self.client_agent = client_agent

    def _add_to_history(self, role: str, message: str):
        """添加消息到历史"""
        self.history.append({"role": role, "message": message})

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

        self._add_to_history("counselor", counselor_init_cn)
        self._add_to_history("client", client_init_cn)

    def _check_goodbye(self, text: str) -> bool:
        """检测是否包含道别词汇"""
        goodbye_keywords = ['再见', '拜拜', '下次见', '再会', '回见']
        return any(keyword in text for keyword in goodbye_keywords)

    def _clean_counselor_response(self, response: str) -> str:
        """从咨询师回复中提取纯文本（去除思考标签）"""
        import re
        clean_response = re.sub(r"<think>.*?</think>\n?", "", response, flags=re.DOTALL).strip()
        return clean_response

    def _exchange_statements(self):
        """交替生成对话"""
        for turn in range(self.max_turns):
            print(f"\n    轮次 {turn + 1}/{self.max_turns}", flush=True)
            print("=" * 60, flush=True)

            # 咨询师回应（使用三联动模型）
            counselor_response = self.counselor_agent.generate(self.history)

            # 打印完整的咨询师回复（包含思考过程）
            print(f"    📋 咨询师:\n{counselor_response}", flush=True)

            # 添加到历史（完整内容）
            self._add_to_history("counselor", counselor_response)

            # 第9轮后检测咨询师是否说了再见
            clean_counselor = self._clean_counselor_response(counselor_response)
            if turn >= 9 and self._check_goodbye(clean_counselor):
                print("\n    ✓ 检测到咨询师道别，会话结束", flush=True)
                print("=" * 60, flush=True)
                break

            # 来访者回应（使用框架原有 Client Agent）
            client_response = self.client_agent.generate(self.history)
            client_response = client_response.replace('Client: ', '')

            # 第9轮后检测来访者是否说了再见
            if turn >= 9 and self._check_goodbye(client_response):
                # 移除 [/END] 标记（如果有）
                client_response = client_response.replace('[/END]', '')
                self._add_to_history("client", client_response)
                print(f"\n    📋 来访者:\n{client_response}", flush=True)
                print("\n    ✓ 检测到来访者道别，会话结束", flush=True)
                print("=" * 60, flush=True)
                break

            # 前10轮：移除 [/END] 标记，不中断对话
            # 10轮后：检测 [/END] 标记，如果存在则结束对话
            if turn < 10:
                # 移除 [/END] 标记
                client_response = client_response.replace('[/END]', '')
            else:
                # 检测是否有结束标记
                if '[/END]' in client_response:
                    # 移除标记并添加到历史
                    client_response = client_response.replace('[/END]', '')
                    self._add_to_history("client", client_response)
                    print(f"\n    📋 来访者:\n{client_response}", flush=True)
                    print("\n    ✓ 检测到结束标记，会话结束", flush=True)
                    print("=" * 60, flush=True)
                    break

            self._add_to_history("client", client_response)
            print(f"\n    📋 来访者:\n{client_response}", flush=True)

    def run_session(self):
        """运行完整会话"""
        self._initialize_session()
        self._exchange_statements()

        return {
            "example": self.example,
            "cbt_technique": "Triple Model (Sentiment + Consultant + Primary)",
            "cbt_plan": f"三联动模型: {self.counselor_agent.primary_interactor.model if hasattr(self.counselor_agent, 'primary_interactor') else 'TripleModel'}",
            "cost": 0,
            "history": self.history  # 保留完整的对话历史（包含 <thinking> 标签）
        }


# ============================================
# 简化的持久化方案：每个进程处理一组任务
# ============================================
def process_worker(
    worker_id: int,
    device_id: int,
    task_chunk: list,
    sentiment_model_path: str,
    primary_model_path: str,
    consultant_model_path: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    n_sentiment_classes: int,
    output_dir: Path,
    max_turns: int,
    total: int
):
    """
    工作进程函数：加载一次模型，然后处理多个任务（完全对齐 inference-rl-custom.py）

    Args:
        worker_id: 工作进程 ID
        device_id: GPU 设备 ID
        task_chunk: 这个进程要处理的任务列表 [(index, example), ...]
        ...: 其他配置参数
    """
    global _process_counselor_agent
    global _process_device_id

    process_name = f"Worker-{worker_id}"
    _process_device_id = device_id

    print(f"\n[{process_name}] 启动 (GPU {device_id})，处理 {len(task_chunk)} 个任务", flush=True)

    # 设置当前进程使用的 GPU
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            print(f"[{process_name}] 设置 GPU: {device_id}", flush=True)
    except Exception as e:
        print(f"[{process_name}] 设置 GPU {device_id} 时警告: {e}", flush=True)

    # 加载环境变量（用于 Client Agent）
    load_env_file()

    # 创建一个临时 example 用于初始化
    temp_example = {"AI_client": {"attitude": "", "attitude_instruction": ""}}

    # 预加载所有三个模型（只加载一次！完全对齐 inference-rl-custom.py）
    print(f"[{process_name}] 正在加载模型到 GPU {device_id}...", flush=True)
    preload_models_for_device(
        sentiment_model_path=sentiment_model_path,
        primary_model_path=primary_model_path,
        consultant_model_path=consultant_model_path,
        bert_model_name='bert-base-chinese',
        n_sentiment_classes=n_sentiment_classes,
        device_id=device_id
    )

    # 创建 Counselor Agent（使用缓存的模型）
    _process_counselor_agent = TripleModelAgent(
        example=temp_example,
        sentiment_model_path=sentiment_model_path,
        primary_model_path=primary_model_path,
        consultant_model_path=consultant_model_path,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        n_sentiment_classes=n_sentiment_classes,
        device_id=device_id,
        use_cache=True
    )

    print(f"[{process_name}] 模型加载完成，开始处理任务", flush=True)

    # 处理分配给这个进程的所有任务
    for index, example in task_chunk:
        file_number = index + 1
        try:
            print(f"\n[{file_number}/{total}] {process_name} 开始处理 (GPU {device_id})", flush=True)

            # 更新 agent 的 example
            _process_counselor_agent.example = example

            # 创建 Client Agent
            client_agent = ClientAgent(example=example)

            # 创建会话
            session = TripleModelTherapySession(
                example=example,
                max_turns=max_turns,
                counselor_agent=_process_counselor_agent,
                client_agent=client_agent
            )

            session_data = session.run_session()

            # 保存结果
            file_name = f"session_{file_number}.json"
            file_path = output_dir / file_name

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=4)

            print(f"[{file_number}/{total}] {process_name} 完成，保存到 {file_name}", flush=True)

        except Exception as e:
            error_file_name = f"error_{file_number}.txt"
            error_file_path = output_dir / error_file_name

            with open(error_file_path, "w", encoding="utf-8") as f:
                f.write(f"Error: {e}\n\n")
                f.write(traceback.format_exc())

            print(f"[{file_number}/{total}] {process_name} 失败，错误已保存到 {error_file_name}", flush=True)

        finally:
            # 只清理生成时的缓存，不卸载模型！（完全对齐 inference-rl-custom.py）
            try:
                import torch
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    with torch.cuda.device(device_id):
                        torch.cuda.empty_cache()
            except Exception:
                pass

    print(f"\n[{process_name}] 完成所有任务", flush=True)


def main():
    # 首先保存默认值
    DEFAULTS = {
        "sentiment_model": "/root/autodl-tmp/xinliyisheng/AruoraCounsel-Sense/new_mlp.bin",
        "primary_model": "/root/autodl-tmp/xinliyisheng/AuroraCounsel-Guide",
        "consultant_model": "/root/autodl-tmp/xinliyisheng/AuroraCounsel-Reflect",
        "system_prompt": "你是一位精通理情行为疗法（Rational Emotive Behavior Therapy，简称REBT）的心理咨询师，能够合理地采用理情行为疗法给来访者提供专业地指导和支持，缓解来访者的负面情绪和行为反应，帮助他们实现个人成长和心理健康。理情行为治疗主要包括以下几个阶段，下面是对话阶段列表，并简要描述了各个阶段的重点。（1）**检查非理性信念和自我挫败式思维**：理情行为疗法把认知干预视为治疗的'生命'，因此，几乎从治疗一开始，在问题探索阶段，咨询师就以积极的、说服教导式的态度帮助来访者探查隐藏在情绪困扰后面的原因，包括来访者理解事件的思维逻辑，产生情绪的前因后果，借此来明确问题的所在。咨询师坚定地激励来访者去反省自己在遭遇刺激事件后，在感到焦虑、抑郁或愤怒前对自己'说'了些什么。（2）**与非理性信念辩论**：咨询师运用多种技术（主要是认知技术）帮助来访者向非理性信念和思维质疑发难，证明它们的不现实、不合理之处，认识它们的危害进而产生放弃这些不合理信念的愿望和行为。（3）**得出合理信念，学会理性思维**：在识别并驳倒非理性信念的基础上，咨询师进一步诱导、帮助来访者找出对于刺激情境和事件的适宜的、理性的反应，找出理性的信念和实事求是的、指向问题解决的思维陈述，以此来替代非理性信念和自我挫bail式思维。为了巩固理性信念，咨询师要向来访者反复教导，证明为什么理性信念是合情合理的，它与非理性信念有什么不同，为什么非理性信念导致情绪失调，而理性信念导致较积极、健康的结果。（4）**迁移应用治疗收获**：积极鼓励来访者把在治疗中所学到的客观现实的态度，科学合理的思维方式内化成个人的生活态度，并在以后的生活中坚持不懈地按理情行为疗法的教导来解决新的问题。你需要一步一步来，你一次最多问一个问题。需要富有同情心的回复用户的问题，并且当交流一段过程了解用户的具体情况后应该不要再问问题而是及时给出建议。",
        "output_dir": "../output-triple-custom",
        "max_turns": 20,
        "max_new_tokens": 2048,
        "temperature": 0.7,
        "num_processes": 1,
        "n_sentiment_classes": 2
    }

    parser = argparse.ArgumentParser(
        description="使用三联动模型生成心理咨询对话",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 使用配置文件
  python inference-triple-custom.py --config ../config_triple.json

  # 使用命令行参数
  python inference-triple-custom.py --sentiment_model /path/to/sentiment.bin --primary_model /path/to/primary --consultant_model /path/to/consultant

  # 使用预设系统提示词
  python inference-triple-custom.py --sentiment_model /path/to/sentiment.bin --primary_model /path/to/primary --consultant_model /path/to/consultant --preset_prompt cbt

  # 使用多 GPU 并行（每个 GPU 一个进程）
  python inference-triple-custom.py --num_processes 6 --device_ids "0,1,2,3,4,5" ...
        """
    )
    parser.add_argument("--config", type=str, help="配置文件路径 (JSON格式)")
    parser.add_argument("--sentiment_model", type=str, default=DEFAULTS["sentiment_model"], help="情感分类模型路径 (.bin文件)")
    parser.add_argument("--primary_model", type=str, default=DEFAULTS["primary_model"], help="主模型路径")
    parser.add_argument("--consultant_model", type=str, default=DEFAULTS["consultant_model"], help="顾问模型路径")
    parser.add_argument("--system_prompt", type=str, default=DEFAULTS["system_prompt"],help="系统提示词（直接输入）")
    parser.add_argument("--preset_prompt", type=str, choices=["cbt", "person_centered", "brief"],
                        help="使用预设系统提示词")
    parser.add_argument("--system_prompt_file", type=str, help="从文件读取系统提示词")
    parser.add_argument("--output_dir", type=str, default=DEFAULTS["output_dir"], help="输出目录")
    parser.add_argument("--max_turns", type=int, default=DEFAULTS["max_turns"], help="最大对话轮数")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULTS["max_new_tokens"], help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=DEFAULTS["temperature"], help="采样温度")
    parser.add_argument("--num_processes", type=int, default=DEFAULTS["num_processes"], help="并行进程数")
    parser.add_argument("--num_samples", type=int, help="处理样本数量（默认处理全部）")
    parser.add_argument("--n_sentiment_classes", type=int, default=DEFAULTS["n_sentiment_classes"], choices=[2, 3],
                        help="情感分类数量: 2=消极/积极, 3=消极/中性/积极")
    parser.add_argument("--device_ids", type=str, default="0,1", help="指定使用的 GPU 设备 ID，例如 '0,1' 或 '0,1,2,3,4,5'")

    args = parser.parse_args()

    # 加载环境变量
    load_env_file()

    # 加载配置文件
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"加载配置文件: {args.config}", flush=True)

    # 合并配置（命令行参数优先）
    # 逻辑：如果命令行参数是默认值且配置文件中有值，则使用配置文件的值
    def get_config_value(arg_name, config_key, default):
        arg_value = getattr(args, arg_name)
        if arg_value == DEFAULTS.get(arg_name, default) and config_key in config:
            return config[config_key]
        return arg_value

    sentiment_model_path = get_config_value("sentiment_model", "sentiment_model_path", DEFAULTS["sentiment_model"])
    primary_model_path = get_config_value("primary_model", "primary_model_path", DEFAULTS["primary_model"])
    consultant_model_path = get_config_value("consultant_model", "consultant_model_path", DEFAULTS["consultant_model"])
    max_turns = get_config_value("max_turns", "max_turns", DEFAULTS["max_turns"])
    max_new_tokens = get_config_value("max_new_tokens", "max_new_tokens", DEFAULTS["max_new_tokens"])
    temperature = get_config_value("temperature", "temperature", DEFAULTS["temperature"])
    n_sentiment_classes = get_config_value("n_sentiment_classes", "n_sentiment_classes", DEFAULTS["n_sentiment_classes"])

    # output_dir 特殊处理，检查配置文件
    output_dir = args.output_dir
    if output_dir == DEFAULTS["output_dir"] and "output_dir" in config:
        output_dir = config["output_dir"]

    # 系统提示词处理
    system_prompt = None
    if args.system_prompt:
        system_prompt = args.system_prompt
    elif args.preset_prompt:
        system_prompt = get_triple_model_preset_prompt(args.preset_prompt)
        print(f"使用预设提示词: {args.preset_prompt}", flush=True)
    elif args.system_prompt_file:
        prompt_file = Path(args.system_prompt_file)
        if prompt_file.exists():
            with open(prompt_file, "r", encoding="utf-8") as f:
                system_prompt = f.read()
        else:
            print(f"警告: 提示词文件不存在: {args.system_prompt_file}", flush=True)
    elif config.get("system_prompt"):
        system_prompt = config["system_prompt"]

    # 验证必需参数
    if not sentiment_model_path:
        print("错误: 请指定情感分类模型路径", flush=True)
        print("  方式1: --sentiment_model /path/to/sentiment.bin", flush=True)
        print("  方式2: 在配置文件中设置 sentiment_model_path", flush=True)
        return

    if not primary_model_path:
        print("错误: 请指定主模型路径", flush=True)
        print("  方式1: --primary_model /path/to/primary", flush=True)
        print("  方式2: 在配置文件中设置 primary_model_path", flush=True)
        return

    if not consultant_model_path:
        print("错误: 请指定顾问模型路径", flush=True)
        print("  方式1: --consultant_model /path/to/consultant", flush=True)
        print("  方式2: 在配置文件中设置 consultant_model_path", flush=True)
        return

    # 加载数据
    data_file = Path(DATA_FILE)
    if not data_file.exists():
        print(f"错误: 数据文件不存在: {data_file}", flush=True)
        print(f"请确保 {DATA_FILE} 文件存在", flush=True)
        return

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 限制样本数量
    if args.num_samples:
        data = data[:args.num_samples]
        print(f"处理前 {args.num_samples} 个样本", flush=True)

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 解析 GPU 设备 ID
    device_ids = [int(x.strip()) for x in args.device_ids.split(',')] if args.device_ids else [0]

    # 打印配置信息
    print("\n" + "=" * 60, flush=True)
    print("配置信息:", flush=True)
    print("=" * 60, flush=True)
    print(f"  情感分类模型: {sentiment_model_path}", flush=True)
    print(f"  情感分类数量: {n_sentiment_classes} ({'消极/积极' if n_sentiment_classes == 2 else '消极/中性/积极'})", flush=True)
    print(f"  主模型: {primary_model_path}", flush=True)
    print(f"  顾问模型: {consultant_model_path}", flush=True)
    print(f"  系统提示词: {system_prompt[:80] if system_prompt else '默认'}{'...' if system_prompt and len(system_prompt) > 80 else ''}", flush=True)
    print(f"  最大轮数: {max_turns}", flush=True)
    print(f"  最大tokens: {max_new_tokens}", flush=True)
    print(f"  采样温度: {temperature}", flush=True)
    print(f"  输出目录: {output_dir}", flush=True)
    print(f"  数据样本数: {len(data)}", flush=True)
    print(f"  并行进程数: {args.num_processes}", flush=True)
    print(f"  GPU 设备: {device_ids}", flush=True)
    print("=" * 60 + "\n", flush=True)

    # 准备所有任务
    all_tasks = [(index, example) for index, example in enumerate(data)]
    total = len(data)

    # 运行会话
    if args.num_processes > 1:
        # 多进程模式
        # 确保进程数不超过 GPU 数
        actual_num_processes = min(args.num_processes, len(device_ids))
        if actual_num_processes != args.num_processes:
            print(f"警告: 进程数 {args.num_processes} 超过 GPU 数 {len(device_ids)}，使用 {actual_num_processes} 个进程", flush=True)

        # 将任务分配给每个 GPU
        # 按轮询分配：第1个任务→GPU0，第2个→GPU1，第3个→GPU0，...
        task_chunks = [[] for _ in range(actual_num_processes)]
        for i, task in enumerate(all_tasks):
            task_chunks[i % actual_num_processes].append(task)

        for i, chunk in enumerate(task_chunks):
            print(f"GPU {device_ids[i]} 分配到 {len(chunk)} 个任务", flush=True)

        # 使用 spawn 方法避免 CUDA 初始化问题
        ctx = multiprocessing.get_context('spawn')

        # 启动工作进程
        processes = []
        for worker_id in range(actual_num_processes):
            p = ctx.Process(
                target=process_worker,
                args=(
                    worker_id,
                    device_ids[worker_id],
                    task_chunks[worker_id],
                    sentiment_model_path,
                    primary_model_path,
                    consultant_model_path,
                    system_prompt,
                    max_new_tokens,
                    temperature,
                    n_sentiment_classes,
                    output_dir,
                    max_turns,
                    total
                )
            )
            processes.append(p)
            p.start()

        # 等待所有进程完成
        for p in processes:
            p.join()

    else:
        # 单进程模式
        print("单进程模式\n", flush=True)
        process_worker(
            0,
            device_ids[0],
            all_tasks,
            sentiment_model_path,
            primary_model_path,
            consultant_model_path,
            system_prompt,
            max_new_tokens,
            temperature,
            n_sentiment_classes,
            output_dir,
            max_turns,
            total
        )

    print("\n" + "=" * 60, flush=True)
    print(f"完成！共处理 {len(data)} 个会话", flush=True)
    print(f"结果保存在: {output_dir}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
