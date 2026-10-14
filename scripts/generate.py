from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_basics.model import Transformer_lm
from cs336_basics.tokenizer import Tokenizer


EOS = "<|endoftext|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer LM.")
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature == 0:
        return logits.argmax().item()

    probabilities = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1:
        sorted_probabilities, sorted_ids = probabilities.sort(descending=True)
        cumulative = sorted_probabilities.cumsum(dim=-1)
        keep = cumulative - sorted_probabilities < top_p
        sorted_probabilities = sorted_probabilities * keep
        sorted_probabilities /= sorted_probabilities.sum()
        sampled_index = torch.multinomial(sorted_probabilities, 1)
        return sorted_ids[sampled_index].item()

    return torch.multinomial(probabilities, 1).item()


@torch.inference_mode()
def generate(
    model: Transformer_lm,
    tokenizer: Tokenizer,
    prompt: str,
    context_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[int]:
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError("The prompt must encode to at least one token.")

    eos_id = tokenizer.reversed_vocab[EOS.encode("utf-8")]
    for _ in range(max_new_tokens):
        inputs = torch.tensor(token_ids[-context_length:], dtype=torch.long, device=device).unsqueeze(0)
        logits = model(inputs)[0, -1]
        next_id = sample_next_token(logits, temperature, top_p)
        token_ids.append(next_id)
        if next_id == eos_id:
            break
    return token_ids


def main() -> None:
    args = parse_args()
    if args.context_length <= 0:
        raise ValueError("--context-length must be positive.")
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens cannot be negative.")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative.")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must lie in (0, 1].")

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    tokenizer = Tokenizer.from_files(
        args.tokenizer_dir / "vocab.pkl",
        args.tokenizer_dir / "merges.pkl",
        special_tokens=[EOS],
    )
    model = Transformer_lm(
        vocab_size=len(tokenizer.vocab),
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    token_ids = generate(
        model,
        tokenizer,
        args.prompt,
        args.context_length,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        device,
    )
    eos_id = tokenizer.reversed_vocab[EOS.encode("utf-8")]
    if args.max_new_tokens > 0 and token_ids[-1] == eos_id:
        token_ids = token_ids[:-1]
    print(tokenizer.decode(token_ids))


if __name__ == "__main__":
    main()
