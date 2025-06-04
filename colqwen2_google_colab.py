"""ColQwen2 demonstration script for Google Colab.

This script shows how to load the Qwen2 model using Hugging Face's
``transformers`` library. It installs required dependencies when running in
Google Colab and generates a short sample response from the model.

Usage::

    python colqwen2_google_colab.py "<your prompt here>"

When executed inside Google Colab, dependencies are installed automatically.
The script falls back to the local environment otherwise.
"""

import subprocess
import sys

try:
    import google.colab  # type: ignore
    IN_COLAB = True
except ImportError:  # pragma: no cover - running locally
    IN_COLAB = False

if IN_COLAB:
    # Install required packages silently when running in Colab
    subprocess.run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "transformers",
        "accelerate",
        ], check=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def generate(prompt: str) -> str:
    """Generate a reply from the ColQwen2 model for ``prompt``."""
    model_name = "Qwen/Qwen2-0.5B"  # adjust to a larger variant if needed
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main(argv: list[str]) -> None:
    prompt = " ".join(argv) if argv else "こんにちは、Qwen2!"
    response = generate(prompt)
    print(response)


if __name__ == "__main__":
    main(sys.argv[1:])
