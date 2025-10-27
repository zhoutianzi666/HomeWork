import openai
import multiprocessing
from functools import partial
import json
import copy
import os
import time
import requests
import yaml

ip = "10.94.130.153"
#service_http_port = "9811"


api_urls = []
for ip in ["10.94.130.153", "10.94.130.151"]:
    for service_http_port in [9811,1822,1833,1844,1855,1866,1877,1888]:
        api_url = f"http://{ip}:{service_http_port}/v1/chat/completions"
        api_urls.append(api_url)

max_concurrency = 500  # 设置最大并发进程数


def process_request(api_url, task_queue, result_dict_mp):
    """处理单个请求的进程函数"""
    # client = openai.Client(base_url=f"http://{ip}:{port}/v1", api_key="EMPTY_API_KEY")
    count = 0
    
    while not task_queue.empty():

        try:
            # 从队列中获取任务（互斥访问）
            task = task_queue.get(timeout=2.0)
        except:
            # 处理获取任务超时的情况
            print('task_queue.get() timeout')
            continue
        
        idx, json_data = task
        task_id = json_data["task_id"]
        pload = json_data["data"]
        result_dict = {}
        pload.update(hyper_parameters)
        headers = {"User-Agent": "Benchmark Client"}
        response = requests.post(url=api_url, headers=headers, json=pload, stream=True)
        text = ""
        if response.status_code == 200:
            
            for chunk_bytes in response.iter_content(chunk_size=1000000):
                chunk = chunk_bytes.decode("utf-8").removeprefix(
                    "data: ")
                if "DONE" not in chunk:
                    chunk = json.loads(chunk)
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        text += chunk["choices"][0]["delta"]["content"]
            # 将结果存入共享字典
            print(text)
            result_dict[task_id] = {
                "idx": idx,
                "text": str(text),
                "status": "success"
            }
        else:
            # 处理非200状态码的情况
            text = f'response.status_code = {response.status_code}, pload = {pload}'
            result_dict[task_id] = {
                    "idx": idx,
                    "text": text,
                    "status": "success"
                }
        output_file_name = f"/root/paddlejob/workspace/env_run/output/zkk/{task_id}.json"
        with open(output_file_name, 'w', encoding='utf-8') as f:
            # 将结果写入文件
            json.dump(dict(result_dict), f, indent=2)
        print(f'Done {task_id}')
        count+=1
    print(f'Task queue empty, done {count}.')

def get_rollout(json_data_list: list, n=1):
    """为每个数据生成n个副本并添加唯一任务ID"""
    result_list = []
    for idx, json_data in enumerate(json_data_list):
        for i in range(n):
            new_json_data = {
                "rollout_n": i,
                "task_id": f"prompt{idx}_rollout{i}",  # 添加唯一任务ID
                "data": copy.deepcopy(json_data)
                }
            result_list.append((idx, new_json_data))  # 存储元组(索引, 数据)
    return result_list

if __name__ == "__main__":
    
    hyper_parameters = {}
    with open("/root/paddlejob/workspace/env_run/output/zkk/HomeWork/fd_script/request.yaml", "r") as f:
        hyper_parameters = yaml.safe_load(f)
        print(f'hyper_parameters = {hyper_parameters}')
    
    # 读取数据
    data_path = "/root/paddlejob/workspace/env_run/output/zkk/HomeWork/fd_script/math_set_test_fd.jsonl_271"

    data_list = []
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                try:
                    data_list.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    # 准备任务
    num_seqs = len(data_list)
    num_seqs = 1000
    rollout_n = 1
    prompt_list = data_list[:num_seqs]
    rollout_reqs = get_rollout(prompt_list, n=rollout_n)
    print(f'Total tasks: {len(rollout_reqs)}')
    
    # 创建任务队列和结果字典
    task_queue = multiprocessing.Queue()
    
    # 将任务放入队列
    for task in rollout_reqs:
        task_queue.put(task)
    
    # 创建并启动进程
    processes = []
    for tt in range(max_concurrency):
        p = multiprocessing.Process(
            target=process_request,
            args=(api_urls[0], task_queue, hyper_parameters)
        )
        processes.append(p)
        p.start()
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    

