# dpo.py

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ==== Config ====
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
DATA_PATH = "data/dpo_dummy_data.json"
EPOCHS = 10
BATCH_SIZE = 2
LR = 1e-5

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Load Model ====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# ==== Utilities ====

def load_dataset_from_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def format_pair(prompt, response):
    return tokenizer(prompt + " " + response, return_tensors="pt", padding=True, truncation=True).to(device)

def compute_logprob(input_ids, attention_mask):
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    chosen_token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
    token_mask = attention_mask[:, 1:]
    return (chosen_token_log_probs * token_mask).sum(dim=1)

def collate_fn(batch):
    rejected_inputs = tokenizer(
        [ex["prompt"] + " " + ex["chosen"] for ex in batch],
        return_tensors="pt", padding=True, truncation=True
    ).to(device)

    chosen_inputs = tokenizer(
        [ex["prompt"] + " " + ex["rejected"] for ex in batch],
        return_tensors="pt", padding=True, truncation=True
    ).to(device)

    return chosen_inputs, rejected_inputs

# ==== Training ====

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

dataset = load_dataset_from_json(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

BETA = 0.1

for epoch in range(EPOCHS):
    total_loss = 0
    for step, (chosen_input, rejected_input) in enumerate(dataloader):
        optimizer.zero_grad()

        logprob_chosen = compute_logprob(**chosen_input)
        logprob_rejected = compute_logprob(**rejected_input)

        dpo_loss = -F.logsigmoid(BETA * (logprob_chosen - logprob_rejected)).mean()
        dpo_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += dpo_loss.item()
        print(f"[Epoch {epoch+1}, Step {step+1}] Loss: {dpo_loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f">>> Epoch {epoch+1} complete — Avg Loss: {avg_loss:.4f}")

# ==== Save model and tokenizer ====
SAVE_PATH = "models/smollm_dpo"
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")
