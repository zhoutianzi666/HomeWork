from fastdeploy import LLM, SamplingParams

prompts = [
    "你好，请问你是谁",
]


# 采样参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)

model = "/root/paddlejob/workspace/env_run/output/EB45T"
# 加载模型
llm = LLM(model=model, tensor_parallel_size=1, data_parallel_size=8, max_model_len=8192, num_gpu_blocks_override=102, engine_worker_queue_port=9981, enable_expert_parallel=True)

# 批量进行推理（llm内部基于资源情况进行请求排队、动态插入处理）
outputs = llm.generate(prompts, sampling_params)

# 输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
    print("generated_text", generated_text)


