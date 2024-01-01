import functools
import glob

import pandas as pd

import t5
import t5.data.mixtures
import t5.models
import torch
import transformers

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

import kbqat_data.tasks_kbqat

import argparse

# UNIFIEDQA_PATH = "/data/xiahan/github/text-to-text-transfer-transformer/unifiedqa-v2-t5-base-1363200/"
# HF_PATH = "allenai/unifiedqa-t5-base"
# UNIFIEDQA_PATH = "/data/xiahan/github/text-to-text-transfer-transformer/unifiedqa-v2-t5-small-1251000/"
# HF_PATH = "allenai/unifiedqa-t5-small"

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_finetune(rank: int, world_size: int, use_cpu: bool, unifiedqa_path: str, model_type: str, subtask: str, start_step: int):
    # subtask = "kr4"
    print(f"Running DDP with model parallel example on rank {rank}.")
    if not use_cpu and world_size != 1:
        setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")


    model = t5.models.HfPyTorchModel(
        unifiedqa_path,  # t5-small
        f"./checkpoints_{model_type}_{subtask}",
        device
    )
    model._model = model._model.half()
    if not use_cpu and world_size != 1:
        model._model = DDP(model._model)

    START_STEP = start_step
    # predict
    if rank == 0:
        print("Before predict.....")
    df = pd.read_csv(f'data/{subtask}/test.tsv', sep='\t', lineterminator='\n')
    column_data = df.iloc[:, 0].tolist()
    with torch.inference_mode():
        model.predict(
            column_data,
            sequence_length={"inputs": 64},
            batch_size=4,
            output_file=f"./data/{subtask}/predict-{START_STEP}.txt",
        )

    # Run 1000 steps of fine-tuning
    if rank == 0:
        print("Train.....")
    model.finetune(
        mixture_or_task_name=f"{subtask}_mixture",
        finetune_steps=10000,
        pretrained_model_dir=unifiedqa_path,
        pretrained_checkpoint_step=START_STEP,
        save_steps=1000,
        sequence_length={"inputs": 64, "targets": 16},
        split="train",
        batch_size=1,
        optimizer=functools.partial(transformers.AdamW, lr=1e-5),
    )

    # predict
    if rank == 0:
        print("After predict.....")
    df = pd.read_csv(f'data/{subtask}/test.tsv', sep='\t', lineterminator='\n')
    column_data = df.iloc[:, 0].tolist()

    checkpoints = glob.glob(os.path.join(f"./checkpoints_{model_type}_{subtask}", "model.*"))
    for checkpoint in checkpoints:
        step = int(checkpoint.split("-")[-1])
        model.load_checkpoint(step, model_dir=f"./checkpoints_{model_type}_{subtask}")
        with torch.inference_mode():
            model.predict(
                column_data,
                sequence_length={"inputs": 64},
                batch_size=4,
                output_file=f"./checkpoints_{model_type}_{subtask}/predict-{step}.txt",
            )

    if not use_cpu and world_size != 1:
        cleanup()

def run_demo(demo_fn, world_size, use_cpu):
    mp.spawn(demo_fn,
             args=(world_size, use_cpu),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetuning unifiedqa")
    parser.add_argument('--unifiedqa_path', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--subtask', type=str)
    parser.add_argument('--start_step', type=int)

    args = parser.parse_args()

    use_cpu = False
    single_gpu = True
    if not use_cpu:
        if single_gpu:
            world_size = 1
        else:
            n_gpus = torch.cuda.device_count()
            assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
            world_size = n_gpus
    else:
        world_size = 1
    
    if not use_cpu and world_size != 1:
        run_demo(demo_finetune, world_size, use_cpu)
    else:
        demo_finetune(0, 1, use_cpu, args.unifiedqa_path, args.model_type, args.subtask, args.start_step)