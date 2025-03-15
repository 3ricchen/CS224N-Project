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
from einops import rearrange

from peft import PrefixEncoder, PrefixTuningConfig

from my_datasets import (
  SonnetsDataset
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model
from transformers import GPT2Tokenizer

from optimizer import AdamW

TQDM_DISABLE = False

def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

class SonnetGPTWithPrefix(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prefix_length = args.prefix_length
        
        self.prefix = nn.Parameter(torch.randn(1, args.prefix_length, args.d) * 0.005)
        self.dropout = nn.Dropout(args.dropout)

        # Freeze GPT-2 parameters, but not prefix
        for param in self.gpt.parameters():
            param.requires_grad = False
        self.prefix.requires_grad = True

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        word_embeddings = self.gpt.embed(input_ids)
        prefix_embeddings = self.prefix.expand(batch_size, -1, -1)
        combined_embeddings = torch.cat([prefix_embeddings, word_embeddings], dim=1)
        prefix_attention_mask = torch.ones(batch_size, self.prefix_length, device=attention_mask.device)
        attention_mask_with_prefix = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        
        sequence_output = self.gpt.encode(combined_embeddings, attention_mask_with_prefix)
        sequence_output = self.gpt.final_layer_norm(sequence_output)
        
        sequence_output = self.dropout(sequence_output)
        logits = self.gpt.hidden_state_to_token(sequence_output)
        return logits
    def get_device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def generate(self, input_ids, temperature=0.7, top_p=0.9, max_length=128):
        # Top-P sampling
        device = self.get_device()
        token_ids = input_ids.to(device)
        attention_mask = torch.ones(token_ids.shape, dtype=torch.int64, device=device)
        
        for _ in range(max_length):
            logits = self.forward(token_ids, attention_mask)
            # Reshape logits to be 2D
            logits_last = logits[:, -1, :] / temperature
            probs = F.softmax(logits_last, dim=-1)
            
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            top_p_mask = cumulative_probs <= top_p
            top_p_mask[..., 0] = True
            filtered_probs = sorted_probs * top_p_mask
            filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)
            
            sampled_idx = torch.multinomial(filtered_probs, 1)
            sampled_token = sorted_indices.gather(dim=-1, index=sampled_idx)
            
            if sampled_token.item() == self.tokenizer.eos_token_id:
                break
            
            token_ids = torch.cat([token_ids, sampled_token], dim=1)
            new_mask = torch.ones((token_ids.shape[0], 1), dtype=torch.int64, device=device)
            attention_mask = torch.cat([attention_mask, new_mask], dim=1)
        
        # Decode the generated tokens.
        generated_text = self.tokenizer.decode(token_ids[0].cpu().tolist())
        return token_ids, generated_text
    
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
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    sonnet_dataset = SonnetsDataset(args.sonnet_path)
    sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                   collate_fn=sonnet_dataset.collate_fn)

    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

    args = add_arguments(args)
    model = SonnetGPTWithPrefix(args)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

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
            input_ids = logits[:, args.prefix_length:-1, :]  # Predictions start after prefix
            target_ids = b_ids[:, 1:]                        # Targets in original sequence are shifted right
            
            # Reshape for cross entropy
            input_ids = input_ids.reshape(-1, input_ids.size(-1))
            target_ids = target_ids.reshape(-1)                
        
            loss = F.cross_entropy(input_ids, target_ids, reduction='mean')
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
            output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
            print(f'{batch[1]}{output[1]}\n\n')

        save_model(model, optimizer, args, f'{epoch}_{args.filepath}')

@torch.no_grad()
def generate_submission_sonnets(args):
     device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
     saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)
 
     model = SonnetGPTWithPrefix(saved['args'])
     model.load_state_dict(saved['model'])
     model = model.to(device)
     model.eval()
 
     held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)
     generated_sonnets = []
     for batch in held_out_sonnet_dataset:
         sonnet_id = batch[0]
         encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
         output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
         decoded_output = model.tokenizer.decode(output)
         full_sonnet = f'{decoded_output}\n\n'
         generated_sonnets.append((sonnet_id, full_sonnet))
         print(f'{decoded_output}\n\n')
 
     with open(args.sonnet_out, "w+") as f:
         f.write(f"--Generated Sonnets-- \n\n")
         for sonnet in generated_sonnets:
             f.write(f"\n{sonnet[0]}\n")
             f.write(sonnet[1])

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
    parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt")
    parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets_dev.txt")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action='store_true')

    # Generation parameters.
    parser.add_argument("--temperature", type=float, default=1.2, help="softmax temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Cumulative probability for nucleus sampling.")

    parser.add_argument("--batch_size", type=int, default=16, help='The training batch size.')
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    
    # L2 regularization (weight decay)
    parser.add_argument("--weight_decay", type=float, default=0.2, help="Weight decay (L2 regularization) coefficient.")
    
    # Dropout probability for the model.
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")

    parser.add_argument("--model_size", type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        default='gpt2-large', help="The model size as specified on Hugging Face.")

    # LoRA-specific arguments.
    parser.add_argument("--lora", action='store_true', help="Enable LoRA injection for fine-tuning.")
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank (low-dimensional bottleneck).")
    parser.add_argument("--lora_alpha", type=float, default=16, help="LoRA scaling factor alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.3, help="Dropout probability for LoRA update branch.")

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
    elif args.model_size == 'gpt2-xl':
        raise Exception('gpt2-xl is not supported in this example.')
    else:
        raise Exception(f'{args.model_size} is not supported.')
    args.prefix_length = 1
    return args

if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-prefixtuning-sonnet.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  generate_submission_sonnets(args)
  # test(args)