'''
Sonnet generation starter code with LoRA injection and dropout.

Running:
  `python sonnet_generation.py --use_gpu --lora`

trains your SonnetGPT model (with LoRA injected into attention layers)
and writes the required submission files.
'''

import argparse
import random
import torch
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

# LoRA injection

class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, r: int, alpha: float, lora_dropout: float):
        super().__init__()
        self.orig_linear = orig_linear
        self.r = r
        self.alpha = alpha
        
        # Freeze original weights.
        for param in self.orig_linear.parameters():
            param.requires_grad = False
        
        # Initialize LoRA parameters: A is random; B is zeros so that initially output is unchanged.
        self.lora_A = nn.Parameter(torch.randn(r, orig_linear.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(orig_linear.out_features, r))
        self.scaling = self.alpha / self.r
        
        # Dropout applied to the LoRA update branch
        self.lora_dropout = nn.Dropout(lora_dropout)

    def forward(self, x):
        # Standard forward using frozen linear layer plus the LoRA update.
        lora_update = x @ self.lora_A.t() @ self.lora_B.t()
        lora_update = self.lora_dropout(lora_update)
        return self.orig_linear(x) + lora_update * self.scaling


def inject_lora(module: nn.Module, r: int, alpha: float, lora_dropout: float):
    """
    Recursively traverse the given module and replace any nn.Linear submodule
    whose qualified name contains 'attn' with a LoRALinear version.
    """
    for name, child in module.named_children():
        # Recursively inject LoRA into children.
        inject_lora(child, r, alpha, lora_dropout)
        # Check if this child is a linear layer and is part of an attention block.
        # (Here we use a simple heuristic: if the child's name contains "attn".)
        if isinstance(child, nn.Linear) and "attn" in name:
            # Replace with LoRALinear
            setattr(module, name, LoRALinear(child, r, alpha, lora_dropout))


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
    """Your GPT-2 Model designed for sonnet generation, optionally with LoRA injection and dropout."""

    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # By default, fine-tune the full model.
        for param in self.gpt.parameters():
            param.requires_grad = True

        # Initialize dropout layer.
        self.dropout = nn.Dropout(args.dropout)

        # If LoRA is enabled, inject it into appropriate linear layers.
        if args.lora:
            inject_lora(self.gpt, args.lora_rank, args.lora_alpha, args.lora_dropout)
            print(f"LoRA injected with rank {args.lora_rank}, alpha {args.lora_alpha}, and dropout {args.lora_dropout}.")

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for SonnetGPT.
        Given input token IDs and attention mask, returns a sequence of logits
        (one per token) that predict the distribution over the vocabulary.
        """
        output = self.gpt(input_ids, attention_mask)
        hidden_state = output['last_hidden_state']
        # Apply dropout to the hidden states.
        hidden_state = self.dropout(hidden_state)
        return self.gpt.hidden_state_to_token(hidden_state)

    def get_device(self):
        for param in self.gpt.parameters():
            return param.device

    @torch.no_grad()
    def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
        token_ids = encoding.to(self.get_device())
        attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

        for _ in range(max_length):
            # Forward pass to get logits
            logits_sequence = self.forward(token_ids, attention_mask)
            logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

            # Top-p (nucleus) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            top_p_mask = cumulative_probs <= top_p
            top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
            top_p_mask[..., 0] = True  # Always include the highest probability token
            filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
            filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

            # Sample from filtered distribution
            sampled_index = torch.multinomial(filtered_probs, 1)
            sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

            # Stop if end-of-sequence token is reached
            if sampled_token.item() == self.tokenizer.eos_token_id:
                break

            # Append sampled token
            token_ids = torch.cat([token_ids, sampled_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
            )

        generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
        return token_ids, generated_output

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


def train(args):
    """Train GPT-2 for sonnet generation."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sonnet_dataset = SonnetsDataset(args.sonnet_path)
    sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                   collate_fn=sonnet_dataset.collate_fn)

    # Create the held-out dataset: these only have the first 3 lines.
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

    args = add_arguments(args)
    model = SonnetGPT(args)
    model = model.to(device)

    lr = args.lr
    # Include weight decay for L2 regularization.
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            # Get the input and move it to the GPU.
            b_ids, b_mask = batch['token_ids'], batch['attention_mask']
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore last prediction.
            labels = b_ids[:, 1:].contiguous().flatten()  # Shift labels.
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
            output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
            # print(f'{batch[1]}{output[1]}\n\n')

        # TODO: consider a stopping condition.
        save_model(model, optimizer, args, f'{epoch}_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)

    model = SonnetGPT(saved['args'])
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
    parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
    parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

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
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability.")

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
        # You can add the parameters for gpt2-xl if supported.
        raise Exception('gpt2-xl is not supported in this example.')
    else:
        raise Exception(f'{args.model_size} is not supported.')
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train(args)
    generate_submission_sonnets(args)
