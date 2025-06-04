# Qwen2 Colab Demo

This repository includes a minimal script to run the open source Qwen2 model on
Google Colab. The script installs dependencies automatically when executed in a
Colab notebook.

## Usage

1. Upload `colqwen2_google_colab.py` to a Google Colab notebook or open a
   terminal in Colab.
2. Run the script with your prompt:

```bash
python colqwen2_google_colab.py "こんにちは、世界!"
```

When running in Colab, required Python packages (`transformers` and
`accelerate`) will be installed automatically. If you run locally, make sure the
packages are available.

The default model used is `Qwen/Qwen2-0.5B`. You can change `model_name` inside
the script to use a different checkpoint.
