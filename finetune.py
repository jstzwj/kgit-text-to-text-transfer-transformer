import functools

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

import tasks_kbqat
UNIFIEDQA_PATH = "gs://unifiedqa/models/large"

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_finetune(rank: int, world_size: int):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")


    model = t5.models.HfPyTorchModel(
        "t5-large", 
        "./checkpoints",
        device
    )
    model._model = DDP(model._model)

    # Evaluate the pre-trained checkpoint, before further fine-tuning
    if rank == 0:
        print("Eval.....")
    model.eval(
        "kr1_mixture",
        sequence_length={"inputs": 64, "targets": 64},
        batch_size=1,
    )

    # Run 1000 steps of fine-tuning
    if rank == 0:
        print("Train.....")
    model.finetune(
        mixture_or_task_name="kr1_mixture",
        finetune_steps=1000,
        pretrained_model_dir=UNIFIEDQA_PATH,
        pretrained_checkpoint_step=1363200,
        steps=1000,
        save_steps=100,
        sequence_length={"inputs": 64, "targets": 64},
        split="train",
        batch_size=1,
        optimizer=functools.partial(transformers.AdamW, lr=1e-4),
    )

    # Evaluate after fine-tuning
    if rank == 0:
        print("Eval after.....")
    model.eval(
        "kr1_mixture",
        checkpoint_steps="all",
        sequence_length={"inputs": 64, "targets": 4},
        batch_size=4,
    )

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_finetune, world_size)