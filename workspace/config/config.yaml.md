# config/config.yaml
#
# Main configuration file for the AGI Modular Mind Loop.

# -----------------------------------------------------------------------------
# LLM Manager Configuration
# Defines the language model engines the AGI can use.
# 'default_model' is used if no specific model is requested.
# 'engines' lists available models with their specific settings.
# -----------------------------------------------------------------------------
llm:
  default_model: "llama3-8b-instruct"
  engines:
    # A general-purpose, neutral engine for standard tasks.
    neutral_a:
      model_name: "llama3-8b-instruct"
      # Additional parameters for llama.cpp can go here (e.g., context_size)

    # A second engine that can have its mood/persona altered for debates.
    mooded_b:
      model_name: "llama3-8b-instruct"
      # We can add specific parameters here if needed, e.g., different GPU layer settings.

    # A smaller, faster model for simple, routine tasks.
    fast_utility:
      model_name: "phi3-mini-instruct"

# -----------------------------------------------------------------------------
# Runtime Configuration
# Controls the execution of the main MindLoop.
# -----------------------------------------------------------------------------
runtime:
  # max_cycles: -1 means the loop will run indefinitely until interrupted.
  # Set to a positive integer (e.g., 5) to run for a fixed number of cycles.
  max_cycles: -1

# -----------------------------------------------------------------------------
# Memory System Configuration
# Paths for the memory database and vector index.
# -----------------------------------------------------------------------------
memory:
  db_path: "data/memory.db"
  faiss_index_path: "data/memory.index"

# -----------------------------------------------------------------------------
# Trainer Configuration
# Settings for the self-improvement/fine-tuning module.
# -----------------------------------------------------------------------------
trainer:
  # Path to the base GGUF model that will be fine-tuned.
  model_path: "models/llama3-8b.gguf"
  # Path to the script that executes the fine-tuning (e.g., using llama.cpp).
  training_script_path: "scripts/run_finetune.py"
  # Path to store the formatted data for a training run.
  training_data_path: "data/training_set.jsonl"
