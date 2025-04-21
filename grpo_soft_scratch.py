import torch
import json
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ==== Config ====
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
DATA_PATH = "data/grpo_dummy_data.json"
EPOCHS = 3
LR = 1e-5
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Load model and tokenizer ====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
model.train()

# ==== Load dataset ====
with open(DATA_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# ==== Optimizer ====
optimizer = AdamW(model.parameters(), lr=LR)

# ==== Training loop ====
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}")
    total_loss = 0

    for entry in tqdm(dataset):
        optimizer.zero_grad()

        prompt = entry["prompt"]
        responses = entry["responses"]
        preference_dist = torch.tensor(entry["preference_distribution"], dtype=torch.float32, device=device)

        logprobs = []
        for resp in responses:
            text = prompt + " " + resp
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = F.log_softmax(logits, dim=-1)
            labels = input_ids[:, 1:]
            log_probs = torch.gather(probs[:, :-1, :], 2, labels.unsqueeze(-1)).squeeze(-1)
            mask = attention_mask[:, 1:]
            summed_logprob = (log_probs * mask).sum(dim=1).squeeze()
            logprobs.append(summed_logprob)

        logprobs_tensor = torch.stack(logprobs)  # shape: (num_responses,)
        log_softmax_outputs = F.log_softmax(logprobs_tensor, dim=0)  # Normalize across responses
        loss = F.kl_div(log_softmax_outputs, preference_dist, reduction="batchmean", log_target=True)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    print(f"Avg Loss: {total_loss / len(dataset):.4f}")

# ==== Save model ====
SAVE_PATH = "models/smollm_grpo_soft"
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")
