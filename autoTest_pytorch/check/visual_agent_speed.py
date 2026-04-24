import time
import torch
import numpy as np

# from visual_discrete_agent import VisualDiscreteAgent

import sys
import os

from visual_discrete_agent import VisualDiscreteAgent

def profile_visual_agent(num_iterations=100, batch_size=1):
    print(f"Profiling VisualDiscreteAgent...")
    print(f"Iterations: {num_iterations}, Batch Size: {batch_size}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize the agent (fake screen region)
    agent = VisualDiscreteAgent(screen_region=(0, 0, 100, 100))
    agent._set_runtime_modes()
    
    # Create fake inputs
    dummy_input = torch.randn(batch_size, 3, 640, 640).to(device)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            for _ in range(10):
                features = agent.backbone.feature_extractor(dummy_input)
                tokens = features.permute(0, 2, 3, 1).reshape(batch_size, agent.backbone.num_memory_tokens, 128)
                x = agent.backbone.token_adapter(tokens)
                x = x + agent.backbone.memory_position().unsqueeze(0)
                memory = agent.backbone.core.encode(x)
                queries = agent.backbone._build_queries(batch_size)
                out_features = agent.backbone.core.decode(queries, memory)
                q_out = agent.q_network(out_features)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    print("Starting profiling...")
    
    times = {
        "YOLO Backbone": [],
        "Token Adapter (Linear)": [],
        "Transformer Encoder": [],
        "Transformer Decoder": [],
        "FQF Head (Q-Network)": [],
        "Total": []
    }
    
    for _ in range(num_iterations):
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                if device.type == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                
                # 1. YOLO Backbone
                features = agent.backbone.feature_extractor(dummy_input)
                if device.type == "cuda": torch.cuda.synchronize()
                t1 = time.perf_counter()
                
                # 2. Token Adapter
                tokens = features.permute(0, 2, 3, 1).reshape(batch_size, agent.backbone.num_memory_tokens, 128)
                x = agent.backbone.token_adapter(tokens)
                x = x + agent.backbone.memory_position().unsqueeze(0)
                if device.type == "cuda": torch.cuda.synchronize()
                t2 = time.perf_counter()
                
                # 3. Transformer Encoder
                memory = agent.backbone.core.encode(x)
                if device.type == "cuda": torch.cuda.synchronize()
                t3 = time.perf_counter()
                
                # 4. Transformer Decoder
                queries = agent.backbone._build_queries(batch_size)
                out_features = agent.backbone.core.decode(queries, memory)
                if device.type == "cuda": torch.cuda.synchronize()
                t4 = time.perf_counter()
                
                # 5. FQF Head
                q_out = agent.q_network(out_features)
                if device.type == "cuda": torch.cuda.synchronize()
                t5 = time.perf_counter()
                
                times["YOLO Backbone"].append((t1 - t0) * 1000)
                times["Token Adapter (Linear)"].append((t2 - t1) * 1000)
                times["Transformer Encoder"].append((t3 - t2) * 1000)
                times["Transformer Decoder"].append((t4 - t3) * 1000)
                times["FQF Head (Q-Network)"].append((t5 - t4) * 1000)
                times["Total"].append((t5 - t0) * 1000)

    print("\n--- Profiling Results (ms per iteration) ---")
    
    avg_times = {k: np.mean(v) for k, v in times.items()}
    total_avg_time = avg_times["Total"]
    
    print(f"{'Component':<25} | {'Time (ms)':<10} | {'Percentage':<10}")
    print("-" * 52)
    for k, v in avg_times.items():
        if k == "Total":
            continue
        percentage = (v / total_avg_time) * 100
        print(f"{k:<25} | {v:>8.2f} ms | {percentage:>8.2f} %")
    
    print("-" * 52)
    print(f"{'Total':<25} | {total_avg_time:>8.2f} ms | {'100.00 %':>10}")
    print("\n==================================")
    fps = 1000.0 / total_avg_time
    print(f"Estimated Inference FPS: {fps:.2f}")

if __name__ == "__main__":
    profile_visual_agent(num_iterations=100, batch_size=1)
    # You can also profile train_step batch size
    # profile_visual_agent(num_iterations=50, batch_size=32)
