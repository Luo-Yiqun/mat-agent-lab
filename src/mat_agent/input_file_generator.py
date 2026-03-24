from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .ai import load_gateway_config

try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover
    openai = None


DEFAULT_MODEL = "gpt-5"
SOFTWARE_ALIASES = {
    "fhi-aims": "FHI-aims",
    "fhi aims": "FHI-aims",
    "aims": "FHI-aims",
    "qe": "Quantum ESPRESSO",
    "quantum espresso": "Quantum ESPRESSO",
    "espresso": "Quantum ESPRESSO",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_software(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in SOFTWARE_ALIASES:
        options = ", ".join(sorted(set(SOFTWARE_ALIASES)))
        raise ValueError(f"Unsupported software '{value}'. Choose one of: {options}")
    return SOFTWARE_ALIASES[normalized]


def _examples_dir(software: str) -> Path:
    root = _repo_root()
    if software == "FHI-aims":
        return root / "data" / "FHI-aims"
    return root / "data" / "QE"


def _default_output_name(software: str) -> str:
    if software == "FHI-aims":
        return "control.generated.in"
    return "qe.generated.in"


def _read_examples(directory: Path) -> list[dict[str, str]]:
    if not directory.exists():
        raise FileNotFoundError(f"Example directory does not exist: {directory}")
    files = sorted([path for path in directory.iterdir() if path.is_file()])
    if not files:
        raise FileNotFoundError(f"No example files found in {directory}")

    examples: list[dict[str, str]] = []
    for path in files:
        content = path.read_text(encoding="utf-8", errors="replace").strip()
        if content:
            examples.append({"filename": path.name, "content": content})
    if not examples:
        raise ValueError(f"All example files are empty in {directory}")
    return examples


def _build_prompt(software: str, requirement: str, examples: list[dict[str, str]]) -> list[dict[str, str]]:
    blocks: list[str] = []
    for index, sample in enumerate(examples, start=1):
        blocks.append(
            "\n".join(
                [
                    f"Example {index}: {sample['filename']}",
                    "```",
                    sample["content"],
                    "```",
                ]
            )
        )

    system = (
        "You are an expert computational materials scientist. "
        f"Generate a valid {software} input file from user requirements using the few-shot examples. "
        "Match the style and conventions seen in examples when appropriate. "
        "Return only the final input file text, with no markdown and no extra explanation."
    )
    user = "\n\n".join(
        [
            f"Target software: {software}",
            f"User requirements:\n{requirement.strip()}",
            "Few-shot examples:",
            "\n\n".join(blocks),
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _extract_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)

    output = getattr(response, "output", None) or []
    fragments: list[str] = []
    for item in output:
        for content in getattr(item, "content", []) or []:
            text_value = getattr(content, "text", None)
            if text_value:
                fragments.append(str(text_value))
    return "\n".join(fragments)


def _strip_fences(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```") and raw.endswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return raw


def generate_input_file(
    requirement: str,
    software: str,
    output_path: Path,
    config_path: Path,
    model: str = DEFAULT_MODEL,
) -> Path:
    config = load_gateway_config(config_path)
    if config is None:
        raise RuntimeError(
            f"Cannot load gateway config from {config_path}. Ensure api_key and base_url are set."
        )
    if openai is None:
        raise RuntimeError("openai package is not installed in this environment.")

    examples = _read_examples(_examples_dir(software))
    messages = _build_prompt(software, requirement, examples)

    client = openai.OpenAI(api_key=config.api_key, base_url=config.base_url)
    try:
        response = client.responses.create(model=model, messages=messages)
    except TypeError:
        response = client.responses.create(model=model, input=messages)

    content = _strip_fences(_extract_text(response))
    if not content:
        raise RuntimeError("LLM returned an empty response; no input file was generated.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content + "\n", encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mat-agent-inputgen",
        description="Few-shot LLM generator for FHI-aims or Quantum ESPRESSO input files.",
    )
    parser.add_argument(
        "--software",
        required=True,
        help="Target software: fhi-aims | qe | quantum espresso",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="User requirement prompt that describes the desired calculation setup.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_repo_root() / "test_generated_files"),
        help="Directory where generated files will be written.",
    )
    parser.add_argument(
        "--output-name",
        default="",
        help="Optional output filename; defaults by software.",
    )
    parser.add_argument(
        "--config",
        default=str(_repo_root() / "config.json"),
        help="Path to gateway config JSON containing api_key and base_url.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name for generation (default: {DEFAULT_MODEL}).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    software = _normalize_software(args.software)
    output_name = args.output_name or _default_output_name(software)
    output_path = Path(args.output_dir) / output_name

    path = generate_input_file(
        requirement=args.prompt,
        software=software,
        output_path=output_path,
        config_path=Path(args.config),
        model=args.model,
    )
    print(f"Generated {software} input file: {path.resolve()}")


if __name__ == "__main__":
    main()
