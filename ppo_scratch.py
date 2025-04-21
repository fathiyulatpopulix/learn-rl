import torch
import json
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ==== Config ====
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
DATA_PATH = "data/ppo_critic_data.json"
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-5
CLIP_EPS = 0.2
VF_COEFF = 0.5
ENTROPY_COEFF = 0.01

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Load model & tokenizer ====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)

# ==== Load dataset ====
with open(DATA_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# ==== Optimizer ====
optimizer = AdamW(model.parameters(), lr=LR)

# ==== PPO Step Function ====
def ppo_step(query, response, reward):
    model.eval()
    ref_model.eval()

    # Tokenize
    input_text = query + " " + response
    tokens = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
    input_ids = tokens.input_ids
    attention_mask = tokens.attention_mask

    # Get log_probs from model and reference
    with torch.no_grad():
        ref_logits = ref_model(input_ids, attention_mask=attention_mask).logits
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
    ref_log_probs = torch.gather(ref_log_probs[:, :-1, :], 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    model.train()
    logits = model(input_ids, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs[:, :-1, :], 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    # Calculate ratios and loss
    advantages = reward - reward.mean()  # simple advantage
    ratio = torch.exp(log_probs.sum(1) - ref_log_probs.sum(1))

    policy_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages).mean()
    entropy = -(log_probs.exp() * log_probs).sum(-1).mean()

    total_loss = policy_loss - ENTROPY_COEFF * entropy
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    return total_loss.item(), ratio.mean().item(), entropy.item()

# ==== Training Loop ====
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}")
    epoch_loss = 0
    for example in tqdm(dataset):
        loss, ratio, entropy = ppo_step(example["prompt"], example["response"], torch.tensor([example["reward"]], device=device))
        epoch_loss += loss
    print(f"Avg Loss: {epoch_loss / len(dataset):.4f}")

# ==== Save model ====
SAVE_PATH = "models/smollm_ppo"
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")
