# !/usr/bin/env python3

"""
Evaluation code for Quora paraphrase detection.

model_eval_paraphrase is suitable for the dev (and train) dataloaders where the label information is available.
model_test_paraphrase is suitable for the test dataloader where label information is not available.
"""

import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np
from sacrebleu.metrics import CHRF
from my_datasets import (
  SonnetsDataset,
)

TQDM_DISABLE = False


@torch.no_grad()
def model_eval_paraphrase(dataloader, model, device, tokenizer):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  # help(model.forward)
  # exit()
  y_true, y_pred, sent_ids = [], [], []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sent_ids, labels = batch['token_ids'], batch['attention_mask'], batch['sent_ids'], batch[
      'labels'].flatten()

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)
    inputs = {'input_ids' : b_ids, 'attention_mask' : b_mask}
    base_unit_location = prompt["input_ids"].shape[-1] - 1  # last position
    outputs = model.generate(
      **inputs, unit_locations=[b_ids.size(1)-1], max_length=b_ids.size(1)+1, num_beams=1, do_sample=False
    ).cpu().numpy()
    logits = outputs[0]
    # logits = model(b_ids, b_mask).cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    labels = (labels == 8505).long()
    
    y_true.extend(labels)
    y_pred.extend(preds)
    sent_ids.extend(b_sent_ids)

  f1 = f1_score(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true, y_pred)

  return acc, f1, y_pred, y_true, sent_ids


@torch.no_grad()
def model_eval_paraphrase_intervenable(dataloader, model, device, tokenizer, TQDM_DISABLE = False):
    model.eval()  # Turn off dropout and other randomness.
    y_true, y_pred, sent_ids = [], [], []
    
    yes_token_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    # step, batch = next(enumerate(tqdm(dataloader, desc='eval', disable=TQDM_DISABLE)))
    # Decode batch with tokenizer
    # base_unit_location = batch["input_ids"].shape[-1] - 1
    # print(batch)
    for step, batch in enumerate(tqdm(dataloader, desc='eval', disable=TQDM_DISABLE)):
        b_ids = batch['token_ids'].to(device)
        b_mask = batch['attention_mask'].to(device)
        b_sent_ids = batch['sent_ids']
        labels = batch['labels'].flatten()
        
        
        # Compute the actual length (number of non-padded tokens) for each example.
        # Assuming that b_mask contains 1s for tokens and 0s for padding.
        lengths = b_mask.sum(dim=1)  # shape: [batch_size]
        
        # For each example, the base unit is the last non-padded token.
        # Create a nested list (one per sample) in the expected format.
        total_length = b_ids.shape[1]
        unit_locations_batch = [[[total_length - 1]] for _ in range(b_ids.shape[0])]

        # for i in range(b_ids.shape[0]):
        #     tot_length =  b_ids.shape[1]
        #     intervention_idx = tot_length - 1
        #     token_id = b_ids[i, intervention_idx].item()
        #     token_str = tokenizer.decode([token_id])
        #     full_prompt = tokenizer.decode(b_ids[i], skip_special_tokens=False)
        #     print('*' * 50)
        #     print(f"Sample {i}:")
        #     print(f"  Full prompt: {full_prompt}")
        #     print(f"  Intervention token (at index {intervention_idx}): {token_str}\n")
        #     if token_id in expected_tokens:
        #         print("  Intervention token matches expected marker.")
        #     else:
        #         print("  Intervention token does NOT match expected marker.")
        #     print('*' * 50)
        
        
        # Prepare the input dictionary for the base prompt.
        prompt_batch = {"input_ids": b_ids, "attention_mask": b_mask}

        _, reft_response = reft_model.generate(
            prompt_batch,
            unit_locations={"sources->base": (None, unit_locations_batch)},
            intervene_on_prompt=True,
            max_new_tokens=512,
            do_sample=True, 
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
        
        # Process each generated output.
        first_generated = reft_response[:, total_length]
        pred_batch = (first_generated == yes_token_id).long()
        true_batch = (labels.cpu() == yes_token_id).long()
        
        y_pred.extend(pred_batch.cpu().numpy().tolist())
        y_true.extend(true_batch.numpy().tolist())
        sent_ids.extend(b_sent_ids)
    
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    
    return acc, f1, y_pred, y_true, sent_ids

@torch.no_grad()
def model_test_paraphrase(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true, y_pred, sent_ids = [], [], []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sent_ids = batch['token_ids'], batch['attention_mask'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask).cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_pred.extend(preds)
    sent_ids.extend(b_sent_ids)

  return y_pred, sent_ids


def test_sonnet(
    test_path='predictions/generated_sonnets.txt',
    gold_path='data/TRUE_sonnets_held_out.txt'
):
    chrf = CHRF()

    # get the sonnets
    generated_sonnets = [x[1] for x in SonnetsDataset(test_path)]
    true_sonnets = [x[1] for x in SonnetsDataset(gold_path)]
    max_len = min(len(true_sonnets), len(generated_sonnets))
    true_sonnets = true_sonnets[:max_len]
    generated_sonnets = generated_sonnets[:max_len]

    # compute chrf
    chrf_score = chrf.corpus_score(generated_sonnets, [true_sonnets])
    return float(chrf_score.score)