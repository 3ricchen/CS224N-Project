'''
Sonnet generation starter code with LoRA injection, topK sampling, LoRA dropout, and hyperparameter grid search.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import random
import torch
import math
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from my_datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False

########################################
# LoRA Implementation with Dropout
########################################

class LoRALinear(nn.Module):
    """
    A linear layer augmented with LoRA (Low-Rank Adaptation) and optional dropout.
    Given a frozen base weight, we learn two low-rank matrices A and B.
    The effective weight is: W_eff = W + (B @ A) * scaling.
    """
    def __init__(self, in_features, out_features, r=4, lora_alpha=16, bias=True, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        # scaling factor
        self.scaling = self.lora_alpha / self.r if self.r > 0 else 1

        # The original (frozen) weight.
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        # LoRA parameters (initialized to zero for B and using a standard init for A)
        if self.r > 0:
            self.A = nn.Parameter(torch.zeros(self.r, in_features))
            self.B = nn.Parameter(torch.zeros(out_features, self.r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        else:
            self.A = None
            self.B = None
        # Dropout for LoRA adjustment.
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.r > 0:
            lora_adjustment = (self.B @ self.A) * self.scaling
            lora_adjustment = self.lora_dropout(lora_adjustment)
            effective_weight = self.weight + lora_adjustment
        else:
            effective_weight = self.weight
        return F.linear(x, effective_weight, self.bias)


def get_parent_module(model, module_name):
    """Given a dot-separated module name, return the parent module and the attribute name."""
    names = module_name.split('.')
    parent = model
    for n in names[:-1]:
        parent = getattr(parent, n)
    return parent, names[-1]


def inject_lora_in_gpt2(gpt_model, lora_r, lora_alpha, lora_dropout=0.0):
    """
    Recursively inject LoRA modules into the target linear layers.
    For demonstration, we replace linear layers whose names include "attn.c_attn" or "attn.c_proj".
    Adjust the target strings as needed based on your GPT2Model structure.
    """
    for name, module in gpt_model.named_modules():
        # Check if the module is a linear layer and if its name suggests it belongs to attention.
        if isinstance(module, nn.Linear) and ('attn.c_attn' in name or 'attn.c_proj' in name):
            parent, attr = get_parent_module(gpt_model, name)
            # Create a LoRA-injected layer with the same dimensions.
            lora_linear = LoRALinear(
                module.in_features,
                module.out_features,
                r=lora_r,
                lora_alpha=lora_alpha,
                bias=(module.bias is not None),
                dropout=lora_dropout
            )
            # Copy the pretrained weights into the base weight.
            with torch.no_grad():
                lora_linear.weight.copy_(module.weight)
                if module.bias is not None:
                    lora_linear.bias.copy_(module.bias)
            setattr(parent, attr, lora_linear)
            print(f"Injected LoRA into module: {name}")

########################################
# SonnetGPT Model Definition
########################################

class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for sonnet generation, now with optional LoRA injection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # Optionally freeze parts of the model if not using LoRA.
    for param in self.gpt.parameters():
      param.requires_grad = True

    # If using LoRA, inject it into the target submodules.
    if getattr(args, 'use_lora', False):
      inject_lora_in_gpt2(self.gpt, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

  def forward(self, input_ids, attention_mask):
    """
    Forward pass for SonnetGPT.
    Given input token IDs and attention mask, returns sequence logits over the vocabulary.
    """
    return self.gpt.hidden_state_to_token(self.gpt(input_ids, attention_mask)['last_hidden_state'])

  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, top_k=0, max_length=128):
    """
    Generation method supporting both nucleus (top-p) and top-k sampling.
    - top_k: if greater than 0, only the top_k logits are considered.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    for _ in range(max_length):
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Apply top-k filtering if requested.
      if top_k > 0:
          kth_values = torch.topk(logits_last_token, top_k)[0][:, -1].unsqueeze(-1)
          logits_last_token = torch.where(
              logits_last_token < kth_values,
              torch.full_like(logits_last_token, -float('Inf')),
              logits_last_token
          )

      # Compute probabilities.
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Apply nucleus (top-p) filtering if top_p < 1.0.
      if top_p < 1.0:
          sorted_probs, sorted_indices = torch.sort(probs, descending=True)
          cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
          top_p_mask = cumulative_probs <= top_p
          top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
          top_p_mask[..., 0] = True  # Always include the highest probability token
          filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
          filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize
          sampled_index = torch.multinomial(filtered_probs, 1)
          sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)
      else:
          sampled_token = torch.multinomial(probs, 1)

      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat([attention_mask, torch.ones((token_ids.shape[0], 1), dtype=torch.int64).to(self.get_device())], dim=1)

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output

########################################
# Utility Functions
########################################

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
  print(f"Saved the model to {filepath}")

def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

########################################
# Training and Generation Functions
########################################

def train(args):
  """Train GPT-2 for sonnet generation with optional LoRA injection."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
      labels = b_ids[:, 1:].contiguous().flatten()
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    print('Generating several output sonnets...')
    model.eval()
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      # Pass top_k from args to generation.
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
      print(f'{batch[1]}{output[1]}\n\n')

    # Update save path to include hyperparameter settings.
    current_filepath = f'{epoch}_{args.epochs}-{args.lr}-lora{getattr(args, "lora_r", "noLORA")}-{getattr(args, "lora_alpha", "noLORA")}-sonnet.pt'
    save_model(model, optimizer, args, current_filepath)

  return train_loss

@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(f'{args.epochs-1}_{args.filepath}', map_location=device, weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    # Pass top_k from args to generation.
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))
    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])

########################################
# Grid Search Functionality
########################################

def grid_search(args):
  """
  Run a grid search over specified hyperparameters.
  Here we search over learning rates and (if using LoRA) lora_r and lora_alpha.
  For each combination we train for the specified number of epochs and print the final loss.
  """
  import itertools

  best_loss = float('inf')
  best_params = None

  # Get grid lists from args.
  lr_list = args.lr_grid
  if args.use_lora:
    lora_r_list = args.lora_r_grid
    lora_alpha_list = args.lora_alpha_grid
  else:
    lora_r_list = [None]
    lora_alpha_list = [None]

  for lr, lora_r, lora_alpha in itertools.product(lr_list, lora_r_list, lora_alpha_list):
    # Update args with the current hyperparameter combination.
    args.lr = lr
    if args.use_lora:
      args.lora_r = lora_r
      args.lora_alpha = lora_alpha
    # Update filepath to avoid collisions.
    args.filepath = f'{args.epochs}-{lr}-lora{lora_r}-{lora_alpha}-sonnet.pt'
    print(f"\n=== Training with lr={lr}, lora_r={lora_r}, lora_alpha={lora_alpha} ===")
    final_loss = train(args)
    print(f"Final training loss: {final_loss:.3f}")
    if final_loss < best_loss:
      best_loss = final_loss
      best_params = (lr, lora_r, lora_alpha)

  print("\nGrid search complete.")
  print(f"Best hyperparameters: lr={best_params[0]}, lora_r={best_params[1]}, lora_alpha={best_params[2]} with loss {best_loss:.3f}")

########################################
# Argument Parsing
########################################

def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability for nucleus sampling.", default=0.9)
  parser.add_argument("--top_k", type=int, help="Top-k filtering for generation (0 for no top_k filtering).", default=0)

  parser.add_argument("--batch_size", help='Training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="Learning rate", default=5e-5)
  parser.add_argument("--model_size", type=str, help="Model size as specified on Hugging Face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2-large')

  # LoRA-specific arguments.
  parser.add_argument("--use_lora", action="store_true", default = True, help="Whether to inject LoRA into the model")
  parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
  parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
  parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers")

  # Grid search arguments.
  parser.add_argument("--grid_search", action="store_true", help="Perform grid search over hyperparameters")
  parser.add_argument("--lr_grid", type=float, nargs='+', default=[5e-6], help="List of learning rates to try")
  parser.add_argument("--lora_r_grid", type=int, nargs='+', default=[4], help="List of LoRA rank values to try")
  parser.add_argument("--lora_alpha_grid", type=int, nargs='+', default=[16], help="List of LoRA alpha values to try")

  parser.add_argument("--weight_decay", type=float, help="Weight decay coefficient", default=0.0)

  args = parser.parse_args()
  return args

def add_arguments(args):
  """Add deterministic arguments based on the model size."""
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
  elif args.model_size == 'gpt2-xl':
    # You can add parameters for gpt2-xl if supported.
    args.d = 1600
    args.l = 48
    args.num_heads = 25
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args

########################################
# Main
########################################

if __name__ == "__main__":
  args = get_args()
  # Update filepath to include hyperparameter settings (this will be overridden in grid search if needed).
  args.filepath = f'{args.epochs}-{args.lr}-lora{getattr(args, "lora_r", "noLORA")}-{getattr(args, "lora_alpha", "noLORA")}-sonnet.pt'
  seed_everything(args.seed)
  
  if args.grid_search:
    grid_search(args)
  else:
    train(args)
    generate_submission_sonnets(args)
