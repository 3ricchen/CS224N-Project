import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Model, GPT2Tokenizer
from torch.optim import AdamW
from my_datasets import ParaphraseDetectionDataset, load_paraphrase_data
from types import SimpleNamespace
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import time

# Define the model using GPT2-XL
class GPT2ParaphraseClassifier(nn.Module):
    def __init__(self, hidden_size=1280, linear_h = 256, num_labels=2, predrop_prob = 0.3, dropout_prob = 0.3, freeze = 0.98, pooling_type = 'mean', size = 'gpt2-large'):
        super(GPT2ParaphraseClassifier, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained(size)
        self.pooling_type = pooling_type
        self.hidden_size = hidden_size
        self.freeze = freeze
        num_layers = len(self.gpt2.h)
        num_freeze = int(num_layers * freeze)  # freeze first half

        for i in range(num_freeze):
            for param in self.gpt2.h[i].parameters():
                param.requires_grad = False
        print(f"Froze {num_freeze} out of {num_layers} GPT2 layers.")
        self.pre_drop = nn.Dropout(dropout_prob)
        self.dropout_prob = dropout_prob
        # Classifier that takes the concatenated embeddings from s1 and s2.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, linear_h),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(linear_h, num_labels)
        )
        print('Model initialized with dropout_prob =', dropout_prob)
    def mean_pooling(self, hidden_states, attention_mask):
        """
        Applies mean pooling to the hidden states.
        Only considers non-masked (i.e., valid) tokens.
        """
        # Expand attention_mask dimensions for broadcasting.
        mask = attention_mask.unsqueeze(-1).float()
        # Compute weighted sum and then normalize.
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        return pooled
    
    def max_pooling(self, hidden_states, attention_mask):
        """
        Applies max pooling to the hidden states.
        Masked positions are set to a very negative value to ignore them.
        """
        mask = attention_mask.unsqueeze(-1)
        # Replace masked positions with a large negative value.
        masked_hidden_states = hidden_states.masked_fill(mask == 0, -1e9)
        # Take the maximum value over the sequence length.
        pooled, _ = masked_hidden_states.max(dim=1)
        return pooled
    
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        # Process first sentence.
        outputs1 = self.gpt2(input_ids=input_ids_1, attention_mask=attention_mask_1)
        hidden_states1 = outputs1.last_hidden_state  # shape: (batch, seq_len, hidden_size)
        # Mean pooling (accounting for attention mask)
        # mask1 = attention_mask_1.unsqueeze(-1).float()
        # pooled1 = (hidden_states1 * mask1).sum(dim=1) / mask1.sum(dim=1)
        if self.pooling_type == 'mean':
            pooled1 = self.mean_pooling(hidden_states1, attention_mask_1)
        elif self.pooling_type == 'max':
            pooled1 = self.max_pooling(hidden_states1, attention_mask_1)
        
        # Process second sentence.
        outputs2 = self.gpt2(input_ids=input_ids_2, attention_mask=attention_mask_2)
        hidden_states2 = outputs2.last_hidden_state
        if self.pooling_type == 'mean':
            pooled2 = self.mean_pooling(hidden_states2, attention_mask_2)
        elif self.pooling_type == 'max':
            pooled2 = self.max_pooling(hidden_states2, attention_mask_2)
        
        # Concatenate both pooled representations.
        combined = torch.cat((pooled1, pooled2), dim=1)  # shape: (batch, hidden_size*2)
        combined = self.pre_drop(combined)
        logits = self.classifier(combined)
        return logits

# Example training loop using your dataset and model.
def train_model(model, dataloader, eval_train_dataloader, dev_dataloader, num_epochs=3, lr=1e-5, weight_decay = 0.02, 
                check_every = 2, device="cuda" if torch.cuda.is_available() else "cpu",):
    print('TRAINING MODEL WITH HYPERPARAMS (num_epochs, lr, weight_decay) =', num_epochs, lr, weight_decay)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)


    criterion = nn.CrossEntropyLoss()
    
    best_dev_f1 = -np.inf
    patience = 3  # Stop if no improvement for 3 evals
    patience_counter = 0

    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in iterator:
            # Move inputs to the device.
            input_ids_1 = batch['input_ids1'].to(device)
            attention_mask_1 = batch['attention_mask1'].to(device)
            input_ids_2 = batch['input_ids2'].to(device)
            attention_mask_2 = batch['attention_mask2'].to(device)
            labels = batch['int_labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {total_loss/len(dataloader):.4f}")
        if (epoch + 1) % check_every == 0:
            train_acc, train_f1, _, _ = evaluate_model(model, eval_train_dataloader, device)
            print(f"Epoch {epoch+1} - Train Accuracy: {train_acc:.4f} | Train F1: {train_f1:.4f}")
            dev_acc, dev_f1, _, _ = evaluate_model(model, dev_dataloader, device)
            print(f"Epoch {epoch+1} - Dev Accuracy: {dev_acc:.4f} | Dev F1: {dev_f1:.4f}")

            scheduler.step(dev_f1)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate adjusted to: {current_lr}")

            if dev_f1 > best_dev_f1:
                try:
                    if torch.cuda.device_count() > 1:
                        save_name = f"frankenstein/best_gpt2_paraphrase_classifier_{lr}-{weight_decay}-{model.module.dropout_prob}-{model.module.pre_drop}_{model.module.freeze}_{epoch}_{args.model_size}.pt"
                    else:
                        save_name = f"frankenstein/best_gpt2_paraphrase_classifier_{lr}-{weight_decay}-{model.dropout_prob}-{model.pre_drop}_{model.freeze}_{epoch}_{args.model_size}.pt"
                except:
                    save_name = f'frankenstein/best_gpt2_xl_paraphrase.pt'
                    print('youre retarded. this is the save name:', save_name)
                best_dev_f1 = dev_f1
                patience_counter = 0
                try:
                    torch.save(model.state_dict(), save_name)
                except:
                    torch.save(model.module.state_dict(), save_name)
                print("New best model saved.", save_name)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("My patience has been exhausted. Fuck this. Early stopping triggered.")
                    break


def evaluate_model(model, dataloader, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        # Wrap the dataloader with tqdm for progress monitoring.
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            # Extract and move batch data to device.
            input_ids_1 = batch['input_ids1'].to(device)
            attention_mask_1 = batch['attention_mask1'].to(device)
            input_ids_2 = batch['input_ids2'].to(device)
            attention_mask_2 = batch['attention_mask2'].to(device)
            labels = batch['int_labels'].to(device)
            
            # Get model predictions.
            logits = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            preds = torch.argmax(logits, dim=1)
            
            # Save predictions and true labels.
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute evaluation metrics.
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, f1, all_preds, all_labels

def save_model(model, path):
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state_dict, path)
    print(f"Model saved to {path}")



if __name__ == "__main__":
    args = SimpleNamespace(
        para_train="data/quora-train.csv",
        para_dev="data/quora-dev.csv",
        para_test="data/quora-test-student.csv",
        para_dev_out="predictions/para-dev-output.csv",
        para_test_out="predictions/para-test-output.csv",
        seed=11711,
        epochs=10,
        use_gpu=False,  # change to True if you want GPU usage
        batch_size=64,
        lr=1e-4,
        weight_decay = 0.2,
        model_size="gpt2-xl"
    )

    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(args.model_size)
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    NUM_TRAIN_EXMAPLES = 5000
    para_train_data = load_paraphrase_data(args.para_train)[:NUM_TRAIN_EXMAPLES]
    para_dev_data = load_paraphrase_data(args.para_dev)[:NUM_TRAIN_EXMAPLES]

    para_train_data = ParaphraseDetectionDataset(para_train_data, args, tokenizer = gpt2_tokenizer)
    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args, tokenizer = gpt2_tokenizer)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=128,
                                    collate_fn=para_dev_data.collate_fn)
    
    ## USED FOR ON-THE-FLY EVALS
    eval_train_data = load_paraphrase_data(args.para_train)[:1000]
    eval_train_loader = DataLoader(eval_train_data, shuffle=False, batch_size=128,
                                    collate_fn=para_train_data.collate_fn)
    eval_dev_data = load_paraphrase_data(args.para_dev)[:1000]
    eval_dev_loader = DataLoader(eval_dev_data, shuffle=False, batch_size=128,
                                    collate_fn=para_dev_data.collate_fn)
    # Create DataLoader with your custom collate function.
    
    # Initialize model.
    model = GPT2ParaphraseClassifier(hidden_size=1600, linear_h = 128, num_labels=2, predrop_prob = 0.1, dropout_prob = 0.6, freeze = 0.99, pooling_type = 'mean', size = args.model_size)
    
    ## --- MUTLIPLE GPU SETUP --- ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Training on", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    

    # Train the model.
    train_model(model, para_train_dataloader, eval_train_loader, eval_dev_loader, num_epochs=20, lr = 1e-4, weight_decay = 0.2, check_every = 1)

    # Evaluate the model.
    train_accuracy, train_f1, train_pred_labels, train_true_labels = evaluate_model(model, para_train_dataloader)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Train F1 Score: {train_f1:.4f}")

    dev_accuracy, dev_f1, dev_pred_labels, dev_true_labels = evaluate_model(model, para_dev_dataloader)
    print(f"Dev Accuracy: {dev_accuracy:.4f}")
    print(f"Dev F1 Score: {dev_f1:.4f}")

    with open(args.para_dev_out, "w+") as f:
        f.write("id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
            f.write(f"{p}, {s} \n")

    with open(args.para_test_out, "w+") as f:
        f.write("id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(test_para_sent_ids, test_para_y_pred):
            f.write(f"{p}, {s} \n")

    save_model(model, "frankenstein/gpt2_paraphrase_classifier.pt")