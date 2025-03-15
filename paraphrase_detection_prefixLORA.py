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
from evaluation import model_eval_paraphrase, model_test_paraphrase
# from models.gpt2 import GPT2Model
from models.gpt2 import GPT2Model
from optimizer import AdamW

import math
from torch import nn

from peft import get_peft_model, LoraConfig, TaskType
import time

TQDM_DISABLE = False

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

from peft import get_peft_model, LoraConfig, TaskType
class LoRALinear(nn.Module):
    """LoRA layer implementation"""
    def __init__(self, in_features, out_features, rank=8, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.linear = nn.Linear(in_features, out_features)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

class ParaphraseGPTLoRAWithPrefix(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(
            model=args.model_size,
            d=args.d,
            l=args.l,
            num_heads=args.num_heads
        )
        
        #  Prefix tuning
        self.prefix_length = args.prefix_length
        self.prefix = nn.Parameter(torch.randn(1, args.prefix_length, args.d) * 0.01)
        
        if args.use_lora:
            print("Using LoRA for fine-tuning")
            for layer in self.gpt.gpt_layers:
                # Update attention layers with a LoRA version
                d = args.d
                layer.self_attention.query = LoRALinear(d, d, args.lora_r, args.lora_alpha)
                layer.self_attention.key = LoRALinear(d, d, args.lora_r, args.lora_alpha)
                layer.self_attention.value = LoRALinear(d, d, args.lora_r, args.lora_alpha)
                layer.self_attention.attention_dense = LoRALinear(d, d, args.lora_r, args.lora_alpha)
        
        self.paraphrase_detection_head = nn.Linear(args.d, 2)
        
        for param in self.gpt.parameters():
            if not isinstance(param, nn.Parameter) or not param.requires_grad:
                param.requires_grad = False
        
        self.prefix.requires_grad = True
        for param in self.paraphrase_detection_head.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        # print(f"Input shape: {input_ids.shape}")
        word_embeddings = self.gpt.embed(input_ids)
        # print(f"Word embeddings shape: {word_embeddings.shape}")
        prefix_embeddings = self.prefix.expand(batch_size, -1, -1)
        # print(f"Prefix shape: {prefix_embeddings.shape}")
        combined_embeddings = torch.cat([prefix_embeddings, word_embeddings], dim=1)
        # print(f"Combined shape: {combined_embeddings.shape}")
        prefix_attention_mask = torch.ones(batch_size, self.prefix_length, device=attention_mask.device)
        attention_mask_with_prefix = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        
        sequence_output = self.gpt.encode(combined_embeddings, attention_mask_with_prefix)
        sequence_output = self.gpt.final_layer_norm(sequence_output)
        
        last_non_pad_idx = attention_mask_with_prefix.sum(dim=1) - 1
        batch_indices = torch.arange(batch_size, device=sequence_output.device).long()
        last_non_pad_idx = last_non_pad_idx.long()
        last_hidden = sequence_output[batch_indices, last_non_pad_idx]
        
        logits = self.paraphrase_detection_head(last_hidden)
        return logits
    
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
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    para_train_data = load_paraphrase_data(args.para_train)
    para_dev_data = load_paraphrase_data(args.para_dev)

    para_train_data = ParaphraseDetectionDataset(para_train_data, args)
    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(
        para_train_data, shuffle=True, batch_size=args.batch_size,
        collate_fn=para_train_data.collate_fn
    )
    para_dev_dataloader = DataLoader(
        para_dev_data, shuffle=False, batch_size=args.batch_size,
        collate_fn=para_dev_data.collate_fn
    )

    args = add_arguments(args)
    model = ParaphraseGPTLoRAWithPrefix(args)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
    best_dev_acc = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            labels = labels.to(device)

            # Convert token IDs to binary labels:
            # Assuming 8505 corresponds to "yes" (paraphrase) and 3919 corresponds to "no" (not a paraphrase)
            binary_labels = torch.where(labels == 8505,
                                        torch.tensor(1, device=labels.device),
                                        torch.tensor(0, device=labels.device))
            
            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, binary_labels, reduction='mean')
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / num_batches
        dev_acc, _, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss:.3f}, dev acc :: {dev_acc:.3f}")

@torch.no_grad()
def test(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(args.filepath, weights_only = False)

    model = ParaphraseGPTLoRAWithPrefix(saved['args'])
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()
    print(f"Loaded model to test from {args.filepath}")

    para_dev_data = load_paraphrase_data(args.para_dev)
    para_test_data = load_paraphrase_data(args.para_test, split='test')

    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
    para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

    para_dev_dataloader = DataLoader(
        para_dev_data, shuffle=False, batch_size=args.batch_size,
        collate_fn=para_dev_data.collate_fn
    )
    para_test_dataloader = DataLoader(
        para_test_data, shuffle=True, batch_size=args.batch_size,
        collate_fn=para_test_data.collate_fn
    )

    dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
    print(f"Dev paraphrase acc :: {dev_para_acc:.3f}")
    test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

    with open(args.para_dev_out, "w+") as f:
        f.write("id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
            f.write(f"{p}, {s} \n")

    with open(args.para_test_out, "w+") as f:
        f.write("id \t Predicted_Is_Paraphrase \n")
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

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--model_size", type=str,
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2-xl')

    # New arguments for LoRA.
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")

    parser.add_argument("--prefix_length", type=int, default=4, help="Length of the learnable prefix")

    args = parser.parse_args()
    return args

def add_arguments(args):
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
        args.d = 1600
        args.l = 48
        args.num_heads = 25
    else:
        raise Exception(f'{args.model_size} is not supported.')
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.epochs}-{args.lr}-paraphrase_LP.pt'
    seed_everything(args.seed)
    start_time = time.time()
    train(args)
    print('*' * 50)
    print('Training time elapsed:', time.time() - start_time)
    print('*' * 50)
    test(args)