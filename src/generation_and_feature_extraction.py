# This code is based on the original repository by Junyi Ye (MIT License, 2024)
# Source: https://github.com/JunyiYe/CreativeMath

import argparse
import logging
import os
import torch
from datetime import datetime
from tqdm import tqdm
from logger import setup_logger
from models import ModelWrapper
from prompts import load_novel_solution_generation_prompt
from utils import load_json, save_json


model_version = {
    "Deepseek-math-7b-rl": "deepseek-ai/deepseek-math-7b-rl",
    "Qwen-2.5-math-7B": "Qwen/Qwen2.5-Math-7B-Instruct",
    "Mathstral-7B": "mistralai/Mathstral-7b-v0.1",
    "OpenMath2-Llama3.1-8B": "nvidia/OpenMath2-Llama3.1-8B",
    "OREAL-7B": "internlm/OREAL-7B"
}


def floatify(obj):
    """
    np.float32 / torch.float32 -> python float
    """
    import numpy as np
    import torch

    if isinstance(obj, dict):
        return {k: floatify(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [floatify(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(floatify(x) for x in obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return float(obj.item())
        else:
            return obj.tolist()
    elif isinstance(obj, float):
        return obj
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(description="Run Step 1. Generation and Step 2. Feature Extraction")
    parser.add_argument("--seed", type=int, default=42, help="randoom seed setting")
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--model_name", type=str, default="Mathstral-7B", help="The model used to generate novel solutions.")
    parser.add_argument("--n_generation", type=int, default=20, help="Number of generation samples for each problem")
    parser.add_argument("--dataset_name", type=str, default="REF", help="REF, TEST, AIME, AHSME, AMC")
    parser.add_argument("--do_sample", type=bool, default=True, help="do sample")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature setting")
    parser.add_argument("--top_p", type=float, default=1.0, help="LLM top_p setting")
    parser.add_argument("--top_k", type=int, default=50, help="LLM top_k setting")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="maximum new generation tokens")
    parser.add_argument("--log_dir", type=str, default="logs", help="logging log dir")
    parser.add_argument("--log_level", type=str, default="INFO", help="logging log level")
    parser.add_argument("--output_dir", type=str, default="output", help="output dir")
    parser.add_argument("--data_dir", type=str, default="data", help="dataset dir")

    
    args = parser.parse_args()
    args.model_id = model_version[args.model_name]

    log_dir = os.path.join(args.log_dir, args.dataset_name, 'generation')
    os.makedirs(log_dir, exist_ok=True)

    output_dir = os.path.join(args.output_dir, args.dataset_name, 'generation')
    os.makedirs(output_dir, exist_ok=True)


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{args.model_name}_{timestamp}.log")
    logger = setup_logger(log_file, args.log_level)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting the novel solution generation on {args.dataset_name} using {args.model_name}")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Logs saved to {os.path.abspath(log_file)}")

    
    model = ModelWrapper(args)

    if args.dataset_name in ["REF", "TEST", "AMC", "AIME", "AHSME"]:
        data_file = os.path.join(args.data_dir, f"{args.dataset_name}.json")
    else:
        msg = f"Invalid dataset name '{args.dataset_name}'. Please choose from ['REF', 'TEST', 'AMC', 'AIME', 'AHSME']."
        logger.error(msg)
        raise ValueError(msg)

    output_file = os.path.join(output_dir, f"{args.model_name}.json")
    
    dataset = load_json(data_file)

    results = []
    cnt = 0

    for idx, sample in enumerate(tqdm(dataset)):
        competition_id = sample["competition_id"]
        problem_id = sample["problem_id"]
        problem = sample["problem"]
        solutions = list(sample["solutions"].values())
        n = len(solutions)

        for k in range(1, n+1):
            for m in range(1, args.n_generation+1):
                prompt = load_novel_solution_generation_prompt(problem, solutions, k)
                response, attentions, ppl, _ = model.generate_response(prompt)
                if not response:
                    continue

                results.append(
                    {"competition_id": competition_id, 
                     "problem_id": problem_id,
                     "k": k,  # number of reference solutions used in the input prompt
                     "n": n,  # total number of available reference solutions
                     "idx": m, 
                     "response": response,
                     "perplexity": ppl
                     }
                )
                cnt += 1

                if cnt % args.save_interval == 0:
                    save_json(floatify(results), output_file)
            
    save_json(floatify(results), output_file)
    logger.info(f"All results saved to {os.path.abspath(output_file)}")
            
if __name__ == "__main__":
    main()