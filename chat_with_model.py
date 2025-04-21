import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

parser = argparse.ArgumentParser(description="Chat with the model")
parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
args = parser.parse_args()

MODEL_PATH = args.model_path
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True).to(device)
model.eval()

print("💬 Chat with the model! Type 'exit' to quit.")
while True:
    prompt = input("You: ")
    if prompt.lower() in ["exit", "quit"]:
        break

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Bot:", response[len(prompt):].strip())
