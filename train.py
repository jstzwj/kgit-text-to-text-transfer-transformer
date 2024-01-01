import functools

import t5
import t5.data.mixtures
import t5.models
import torch
import transformers

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

import tasks_kbqat
model = t5.models.HfPyTorchModel(
    "t5-large", 
    "/data/xiahan/github/kbqat/models/unifiedqa-v2-t5-large-1363200/", 
    device
)

# Evaluate the pre-trained checkpoint, before further fine-tuning
print("Eval.....")
model.eval(
    "kr1_mixture",
    sequence_length={"inputs": 64, "targets": 4},
    batch_size=4,
)

# Run 1000 steps of fine-tuning
print("Train.....")
model.train(
    mixture_or_task_name="kr1_mixture",
    steps=1000,
    save_steps=100,
    sequence_length={"inputs": 64, "targets": 4},
    split="train",
    batch_size=1,
    optimizer=functools.partial(transformers.AdamW, lr=1e-4),
)

# Evaluate after fine-tuning
print("Eval after.....")
model.eval(
    "kr1_mixture",
    checkpoint_steps="all",
    sequence_length={"inputs": 64, "targets": 4},
    batch_size=4,
)

# Generate some predictions
inputs = [
    "cola sentence: This is a totally valid sentence.",
    "cola sentence: A doggy detail was walking famously.",
]
model.predict(
    inputs,
    sequence_length={"inputs": 32},
    batch_size=2,
    output_file="./tmp/hft5/example_predictions.txt",
)