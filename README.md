# Learn Reinforcement Learning

This repository is a personal log of learning LLM fine-tuning using RL methods.

## Installation

1. Clone the repository:

   ```bash
   git clone git@github.com:fathiyulatpopulix/learn-rl.git
   cd learn-rl
   ```

2. Create a virtual environment and install required dependencies in it using uv (recommended):
   ```bash
   uv sync
   ```

## Running

To run a training script, use the following command:

```bash
uv run <dpo/ppo/grpo_scratch>_script.py
```

Your new model should be saved in the `models/` directory. You can load it later for inference by:

```bash
uv run chat_with_model.py --model_path models/<model_name>
```
