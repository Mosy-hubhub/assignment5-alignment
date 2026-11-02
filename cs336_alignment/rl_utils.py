"""Shared checkpoint and configuration helpers for RL training pipelines."""

import json
from pathlib import Path


def save_checkpoint(model, tokenizer, output_dir, step, metrics):
    """Save reloadable weights/tokenizer and selection metrics, not optimizer state."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_cache = model.config.use_cache
    try:
        # 保存后的模型可直接用于生成；保存结束后继续使用训练时的配置。
        model.config.use_cache = True
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    finally:
        model.config.use_cache = use_cache
    (output_dir / "checkpoint.json").write_text(
        json.dumps({"step": step, **metrics}, indent=2), encoding="utf-8"
    )


def save_train_config(args, **metadata):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(metadata)
    (output_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config
