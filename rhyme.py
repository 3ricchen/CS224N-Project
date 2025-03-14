"""
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

Trains your SonnetGPT model and writes the required submission files.
"""

import argparse
import random
import re
import string
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from my_datasets import SonnetsDataset
from models.gpt2 import GPT2Model
from optimizer import AdamW

TQDM_DISABLE = False


# ---------------- Helper Functions ----------------

def count_syllables(word):
    """
    A simple heuristic to count syllables by counting contiguous groups of vowels.
    Not perfect—consider using a pronunciation dictionary for better accuracy.
    """
    word = word.lower()
    word = re.sub(r'[^a-z]', '', word)
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        if ch in vowels:
            if not prev_vowel:
                count += 1
                prev_vowel = True
        else:
            prev_vowel = False
    if word.endswith("e") and count > 1:
        count -= 1
    return count if count > 0 else 1

def get_last_word(line):
    """
    Return the last word in the line (stripped of punctuation).
    """
    words = line.strip().split()
    return words[-1].strip(string.punctuation).lower() if words else ""

def rhymes(word1, word2):
    """
    A naive rhyme function: considers two words to rhyme if their last two letters match.
    """
    if len(word1) < 2 or len(word2) < 2:
        return False
    return word1[-2:] == word2[-2:]


# ---------------- Main Model Definition ----------------

class SonnetGPT(nn.Module):
    """GPT-2 Model modified for sonnet generation."""

    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(
            model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Fine-tune the full model by default.
        for param in self.gpt.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        """
        Forward pass: returns logits for each token.
        """
        hidden_states = self.gpt(input_ids, attention_mask)["last_hidden_state"]
        return self.gpt.hidden_state_to_token(hidden_states)

    def get_device(self):
        for param in self.gpt.parameters():
            return param.device

    @torch.no_grad()
    def generate(self, encoding, temperature=0.7, top_p=0.9, top_k=0, max_length=128):
        """
        Generate text token-by-token.
        If top_k > 0, use top-k sampling; otherwise, use top-p (nucleus) sampling.
        """
        device = self.get_device()
        token_ids = encoding.to(device)
        attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(device)

        for _ in range(max_length):
            logits_sequence = self.forward(token_ids, attention_mask)
            logits_last_token = logits_sequence[:, -1, :] / temperature

            if top_k > 0:
                # Apply top-k filtering:
                values, indices = torch.topk(logits_last_token, top_k)
                filtered_logits = torch.full_like(logits_last_token, -float('Inf'))
                filtered_logits.scatter_(1, indices, values)
                probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
            else:
                # Top-p (nucleus) sampling as before.
                probs = torch.nn.functional.softmax(logits_last_token, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                top_p_mask = cumulative_probs <= top_p
                top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # shift mask right
                top_p_mask[..., 0] = True  # always include the top token
                filtered_probs = sorted_probs * top_p_mask  # zero out tokens beyond top-p
                filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # renormalize
                probs = filtered_probs

            sampled_index = torch.multinomial(probs, 1)
            # sampled_index is already the token id since we've filtered logits.
            if sampled_index.item() == self.tokenizer.eos_token_id:
                break

            token_ids = torch.cat([token_ids, sampled_index], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(device)], dim=1
            )

        generated_output = self.tokenizer.decode(
            token_ids[0].cpu().numpy().tolist()
        )[3:]
        return token_ids, generated_output



# ---------------- Utility Functions ----------------

def save_model(model, optimizer, args, filepath):
    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "args": args,
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }
    torch.save(save_info, filepath)
    print(f"Saved the model to {filepath}")


def train(args):
    """Train the SonnetGPT model."""
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    sonnet_dataset = SonnetsDataset(args.sonnet_path)
    sonnet_dataloader = DataLoader(
        sonnet_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=sonnet_dataset.collate_fn,
    )
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

    args = add_arguments(args)
    model = SonnetGPT(args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for batch in tqdm(sonnet_dataloader, desc=f"train-{epoch}", disable=TQDM_DISABLE):
            b_ids = batch["token_ids"].to(device)
            b_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            logits = rearrange(logits[:, :-1].contiguous(), "b t d -> (b t) d")
            labels = b_ids[:, 1:].contiguous().flatten()
            loss = F.cross_entropy(logits, labels, reduction="mean")
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches
        print(f"Epoch {epoch}: train loss :: {train_loss:.3f}.")

        print("Generating several output sonnets...")
        model.eval()
        for batch in held_out_sonnet_dataset:
            encoding = model.tokenizer(
                batch[1], return_tensors="pt", padding=True, truncation=True
            ).to(device)
            output = model.generate(
                encoding["input_ids"],
                temperature=args.temperature,
                top_p=args.top_p,
            )[0][0]
            decoded_output = model.tokenizer.decode(output)
            # Post-process: enforce exactly 14 lines.
            lines = [line.strip() for line in decoded_output.split("\n") if line.strip()]
            if len(lines) >= 14:
                final_sonnet = "\n".join(lines[:14])
            else:
                final_sonnet = "\n".join(lines + [""] * (14 - len(lines)))
            print(f"{batch[1]}\n{final_sonnet}\n\n")

        save_model(model, optimizer, args, f"{epoch}_{args.filepath}")


@torch.no_grad()
def generate_submission_sonnets(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    saved = torch.load(f"{args.epochs-1}_{args.filepath}", weights_only=False)

    model = SonnetGPT(saved["args"])
    model.load_state_dict(saved["model"])
    model = model.to(device)
    model.eval()

    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)
    generated_sonnets = []

    for batch in held_out_sonnet_dataset:
        sonnet_id = batch[0]
        encoding = model.tokenizer(
            batch[1], return_tensors="pt", padding=False, truncation=True
        ).to(device)
        output = model.generate(
            encoding["input_ids"], temperature=args.temperature, top_p=args.top_p
        )[0][0]
        decoded_output = model.tokenizer.decode(output)
        lines = [line.strip() for line in decoded_output.split("\n") if line.strip()]
        if len(lines) >= 14:
            final_sonnet = "\n".join(lines[:14])
        else:
            final_sonnet = "\n".join(lines + [""] * (14 - len(lines)))
        generated_sonnets.append((sonnet_id, final_sonnet))
        print(f"{final_sonnet}\n\n")

    with open(args.sonnet_out, "w+") as f:
        f.write("--Generated Sonnets--\n\n")
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
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.2, help="softmax temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Cumulative probability for nucleus sampling.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        default="gpt2-large",
        help="The model size as specified on Hugging Face.",
    )
    return parser.parse_args()


def add_arguments(args):
    """Add arguments that are deterministic based on the model size."""
    if args.model_size == "gpt2":
        args.d = 768
        args.l = 12
        args.num_heads = 12
    elif args.model_size == "gpt2-medium":
        args.d = 1024
        args.l = 24
        args.num_heads = 16
    elif args.model_size == "gpt2-large":
        args.d = 1280
        args.l = 36
        args.num_heads = 20
    else:
        raise Exception(f"{args.model_size} is not supported.")
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f"{args.epochs}-{args.lr}-sonnet.pt"  # Save path.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
    generate_submission_sonnets(args)
