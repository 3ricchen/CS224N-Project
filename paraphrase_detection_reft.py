'''
Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu`
trains and evaluates your ParaphraseGPT model and writes the required submission files.
'''

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

TQDM_DISABLE = False

# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

# class ParaphraseGPT(nn.Module):
#   """Your GPT-2 Model designed for paraphrase detection."""

#   def __init__(self, args):
#     super().__init__()
#     # gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
#     self.gpt = transformers.GPT2Model.from_pretrained(args.model_size)
#     self.gpttokenizer = transformers.GPT2Tokenizer.from_pretrained(args.model_size)

#     self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection has two outputs: 1 (yes) or 0 (no).

#     # By default, fine-tune the full model.
#     for param in gpt.parameters():
#       # param.requires_grad = True
#       param.requires_grad = False
    
#     reft_config = pyreft.ReftConfig(representations={
#       "layer": 10,
#       "component": "block_output",
#       "low_rank_dimension": 4,
#       "intervention": pyreft.LoreftIntervention(embed_dim=args.d, low_rank_dimension=4)
#     })

#     self.reft_model = pyreft.get_reft_model(gpt, reft_config)
#     self.reft_model.set_device('cuda')
#     self.reft_model.print_trainable_parameters()
#     # Classification head: maps the final token's representation to 2 classes (paraphrase or not).
#     self.paraphrase_detection_head = nn.Linear(args.d, 2)

#   def forward(self, input_ids, attention_mask):
#     """
#     TODO: Predict the label of the token using the paraphrase_detection_head Linear layer.

#     We structure the input as:

#       'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

#     So you want to find the prediction for the next token at the end of this sentence. Optimistically, it will be the
#     token "yes" (byte pair encoding index of 8505) for examples that are paraphrases or "no" (byte pair encoding index
#      of 3919) for examples that are not paraphrases.
#     """

#     'Takes a batch of sentences and produces embeddings for them.'
#     ### YOUR CODE HERE
#     # hidden = self.gpt(input_ids, attention_mask)['last_hidden_state']
#     # hidden = self.intervention(hidden)
#     outputs = self.reft_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
#     if hasattr(outputs, "hidden_states"):
#       hidden = outputs.hidden_states[-1]
#     else:
#       hidden = outputs["last_hidden_state"]
#     logits = self.paraphrase_detection_head(hidden[:, -1, :])
#     return logits



def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

  args = add_arguments(args)
  # model = ParaphraseGPT(args)
  # model = model.to(device)
  gpt = transformers.GPT2LMHeadModel.from_pretrained(args.model_size)
  gpt.config.return_dict = True
  gpttokenizer = transformers.GPT2Tokenizer.from_pretrained(args.model_size)
  gpttokenizer.pad_token = gpttokenizer.unk_token

  reft_config = pyreft.ReftConfig(representations={
    "layer": 10,
    "component": "block_output",
    "low_rank_dimension": 4,
    "intervention": pyreft.LoreftIntervention(embed_dim=args.d, low_rank_dimension=4)
  })

  reft_model = pyreft.get_reft_model(gpt, reft_config)
  reft_model = reft_model.to(torch.float32)
  reft_model.set_device('cuda')

  lr = args.lr
  # optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)
  best_dev_acc = 0

  ans_mapping = {0: 3919, 1: 8505}
  
  inputs = [f'Question 1: {p[0]}\nQuestion 2: {p[1]}\nAre these questions asking the same thing?\n' for p in para_train_data]
  outputs = [('yes' if p[3] == 1 else 'no') for p in para_train_data]

  inputs = inputs[:10]
  outputs = outputs[:10]
  data_module = pyreft.make_last_position_supervised_data_module(
    gpttokenizer, gpt, inputs, outputs
  )
  print('LOADED DATA MODULE')
  training_args = transformers.TrainingArguments(
    num_train_epochs = 10, output_dir = 'output', per_device_train_batch_size = 8, learning_rate = lr, logging_steps = 40, report_to = []
  )
  trainer = pyreft.ReftTrainerForCausalLM(
    model = reft_model, tokenizer = gpttokenizer, args = training_args, **data_module
  )
  _ = trainer.train()

  dev_acc, _, *_ = model_eval_paraphrase(para_dev_dataloader, reft_model, device, gpttokenizer)
  if dev_acc > best_dev_acc:
    best_dev_acc = dev_acc
    save_model(model, optimizer, args, args.filepath)


@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.filepath)

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=para_test_data.collate_fn)

  dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      f.write(f"{p}, {s} \n")

  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      f.write(f"{p}, {s} \n")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default= 32)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-reft-paraphrase.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  test(args)
