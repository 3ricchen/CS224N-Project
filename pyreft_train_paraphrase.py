import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from my_datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation_reft import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model

from optimizer import AdamW
import transformers

import pyreft
from types import SimpleNamespace

import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpt2 = transformers.AutoModelForCausalLM.from_pretrained('gpt2-xl').to(device)
gpt2_tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2-xl', device = 'cuda')
gpt2_tokenizer.pad_token = gpt2_tokenizer.unk_token
EOS_TOKEN=gpt2_tokenizer.eos_token


# print(gpt2.config.attn_pdrop)
# print(gpt2.config.resid_pdrop)
# print(gpt2.config.embd_pdrop)

# gpt2.config.attn_pdrop = 0.3  # increase attention dropout
# gpt2.config.resid_pdrop = 0.3  # increase residual dropout
# gpt2.config.embd_pdrop = 0.3   # increase embedding dropout

reft_config = pyreft.ReftConfig(representations=[{
    "layer": l, "component": "block_output",
    "low_rank_dimension": 8,
    "intervention": pyreft.LoreftIntervention(embed_dim=gpt2.config.hidden_size,
    low_rank_dimension=8)} for l in [19, 24, 29, 35]])
reft_model = pyreft.get_reft_model(gpt2, reft_config)
reft_model.set_device("cuda")
reft_model = reft_model.float()
reft_model.print_trainable_parameters()

args = SimpleNamespace(
    para_train="data/quora-train.csv",
    para_dev="data/quora-dev.csv",
    para_test="data/quora-test-student.csv",
    para_dev_out="predictions/para-dev-output.csv",
    para_test_out="predictions/para-test-output.csv",
    seed=11711,
    epochs=10,
    use_gpu=True,  # change to True if you want GPU usage
    batch_size=32,
    lr=1e-5,
    model_size="gpt2"
)

para_train_data = load_paraphrase_data(args.para_train)
para_dev_data = load_paraphrase_data(args.para_dev)

para_train_data = ParaphraseDetectionDataset(para_train_data, args, tokenizer = gpt2_tokenizer)
para_dev_data = ParaphraseDetectionDataset(para_dev_data, args, tokenizer = gpt2_tokenizer)

para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=para_train_data.collate_fn)
para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                collate_fn=para_dev_data.collate_fn)

inputs = [f'<|user|>:Tell me if these questions are asking the same thing.\nQuestion 1: {p[0]}\nQuestion 2: {p[1]}\nAre these questions asking the same thing?</s>\n<|assistant|>:' for p in para_train_data]
outputs = [('yes' if p[2] == 1 else 'no') for p in para_train_data]
print('DATA LOADED')


positions = 'l3'
data_module = pyreft.make_multiple_position_supervised_data_module(
    gpt2_tokenizer, gpt2, inputs, outputs,
    positions = positions, num_interventions = len(reft_config.representations))

training_args = transformers.TrainingArguments(
    num_train_epochs=1, output_dir="./tmp", per_device_train_batch_size=10, 
    learning_rate=5e-4, logging_steps=25,
    lr_scheduler_type=transformers.SchedulerType.LINEAR,
    report_to = [], # disable logging
    warmup_steps=100,
    weight_decay = 0.01,
    ) 
start_time = time.time()
trainer = pyreft.ReftTrainerForCausalLM(
    model=reft_model, tokenizer=gpt2_tokenizer, args=training_args, **data_module)
_ = trainer.train()

print('*' * 50)
print(f"Training time: {time.time() - start_time}")
print('*' * 50)
import os
import shutil

save_dir = "./reft_gpt_large_PARAPHRASE_LARGE."
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)  # Remove the existing directory

reft_model.set_device("cpu")  # Move model to CPU before saving
reft_model.save(save_directory=save_dir)