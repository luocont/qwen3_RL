# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass, field


@dataclass
class AgentAPIConfig:
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = os.environ.get("DASHSCOPE_API_KEY", "your-api-key-here")
    model: str = "qwen-plus"


@dataclass
class TrainConfig:
    model_name: str = "/path/to/qwen3-sft-checkpoint"
    output_dir: str = "./outputs/grpo_counselor"
    rl_json: str = "./data/rl_data.json"

    group_size: int = 4
    max_new_tokens: int = 768
    temperature: float = 0.7
    top_p: float = 0.9

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    max_steps: int = 500
    warmup_steps: int = 10
    logging_steps: int = 1
    save_steps: int = 20
    save_total_limit: int = 5
    bf16: bool = True
    fp16: bool = False

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

    kl_coef: float = 0.03
    kl_plot_every_n_steps: int = 10

    local_rank: int = -1
    deepspeed: str | None = None
    fsdp: str | None = None
    ddp_find_unused_parameters: bool = False

    wandb_project: str = "grpo-counselor-rl"


@dataclass
class DataGenConfig:
    client_api: AgentAPIConfig = field(default_factory=AgentAPIConfig)
    case_api: AgentAPIConfig = field(default_factory=AgentAPIConfig)
    panas_api: AgentAPIConfig = field(default_factory=AgentAPIConfig)
    counselor_api: AgentAPIConfig = field(default_factory=AgentAPIConfig)

    num_sessions: int = 50
    max_turns_per_session: int = 12
    output_path: str = "./data/rl_data.json"

    intake_forms: list = field(default_factory=lambda: [
        "来访者是一名25岁的女性，因工作压力大、与男友关系紧张而感到焦虑和抑郁，常常失眠，觉得自己一无是处。",
        "来访者是一名40岁的男性，最近因公司裁员失业，感到迷茫和自我怀疑，对未来充满恐惧，与家人关系也变得紧张。",
        "来访者是一名18岁的高中生，面临高考压力，父母期望很高，感到喘不过气，有时会有逃避的念头。",
        "来访者是一名32岁的女性，经历了一段失败的婚姻，离婚后感到孤独和自我否定，对新的感情关系充满恐惧。",
        "来访者是一名55岁的男性，退休后感到生活失去意义，与子女关系疏远，常常回忆过去，感到空虚和无助。",
    ])
