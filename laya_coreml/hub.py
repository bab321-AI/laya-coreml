"""Resolve a local bundle or a Hugging Face snapshot before inference starts."""

from pathlib import Path

DEFAULT_MODEL = "aac6fef/laya-multilingual-coreml"


def resolve_checkpoint(model, *, revision=None, local_files_only=False):
    path = Path(model).expanduser()
    if path.is_dir():
        return path
    if isinstance(model, Path) or path.is_absolute() or str(model).startswith((".", "~")):
        raise FileNotFoundError(f"Local model directory does not exist: {model}")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            str(model),
            revision=revision,
            local_files_only=local_files_only,
            allow_patterns=[
                "coreml_config.json",
                "rl_agent_config.json",
                "encoder/config.json",
                "tokenizer/*",
                "model.mlpackage/**",
                "host_weights.safetensors",
            ],
        )
    )
