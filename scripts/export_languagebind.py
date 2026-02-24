#!/usr/bin/env python3
"""Export LanguageBind video, audio, and text encoders to ONNX format.

One-time developer script (not shipped with ClipShow). Requires PyTorch +
LanguageBind installed in a separate dev environment.

Requirements:
    pip install torch torchvision torchaudio
    pip install onnx onnxruntime
    # Clone LanguageBind repo and install:
    git clone https://github.com/PKU-YuanGroup/LanguageBind
    cd LanguageBind && pip install -r requirements.txt

Usage:
    python scripts/export_languagebind.py --output-dir ./exported_models/
    python scripts/export_languagebind.py --output-dir ./exported_models/ --validate-only
    python scripts/export_languagebind.py --output-dir ./exported_models/ --encoder video
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _check_dependencies():
    """Verify all required packages are available."""
    missing = []
    for pkg in ("torch", "onnx", "onnxruntime", "languagebind"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(
            f"Missing required packages: {', '.join(missing)}\n"
            "Install with:\n"
            "  pip install torch torchvision torchaudio onnx onnxruntime\n"
            "  # Plus LanguageBind from https://github.com/PKU-YuanGroup/LanguageBind",
            file=sys.stderr,
        )
        sys.exit(1)


def load_languagebind(cache_dir: str = "./cache_dir"):
    """Load LanguageBind with video and audio FT encoders.

    Returns (model, tokenizer) with model in eval mode on CPU.
    """
    import torch
    from languagebind import LanguageBind, LanguageBindImageTokenizer

    clip_type = {
        "video": "LanguageBind_Video_FT",
        "audio": "LanguageBind_Audio_FT",
    }

    logger.info("Loading LanguageBind models (this downloads ~4GB on first run)...")
    model = LanguageBind(clip_type=clip_type, cache_dir=cache_dir)
    model = model.to(torch.device("cpu"))
    model.eval()

    tokenizer = LanguageBindImageTokenizer.from_pretrained(
        "LanguageBind/LanguageBind_Image",
        cache_dir=f"{cache_dir}/tokenizer_cache_dir",
    )

    return model, tokenizer


# ---------------------------------------------------------------------------
# Encoder wrappers — bundle encoder + projection + L2 norm for clean export
# ---------------------------------------------------------------------------


class _VideoEncoderWrapper:
    """Wraps video encoder + projection + L2 norm for ONNX export."""

    def __init__(self, model):
        import torch.nn as nn

        class _Module(nn.Module):
            def __init__(self, encoder, projection):
                super().__init__()
                self.encoder = encoder
                self.projection = projection

            def forward(self, pixel_values):
                # pixel_values: (B, 3, 8, 224, 224)
                outputs = self.encoder(pixel_values=pixel_values)
                pooled = outputs[1]  # CLS token pooled output
                projected = self.projection(pooled)
                return projected / projected.norm(dim=-1, keepdim=True)

        self.module = _Module(
            model.modality_encoder["video"],
            model.modality_proj["video"],
        )
        self.module.eval()

    @property
    def dummy_input(self):
        import torch

        return torch.randn(1, 3, 8, 224, 224)

    @property
    def input_names(self):
        return ["pixel_values"]

    @property
    def dynamic_axes(self):
        return {"pixel_values": {0: "batch_size"}, "embeddings": {0: "batch_size"}}


class _AudioEncoderWrapper:
    """Wraps audio encoder + projection + L2 norm for ONNX export."""

    def __init__(self, model):
        import torch.nn as nn

        class _Module(nn.Module):
            def __init__(self, encoder, projection):
                super().__init__()
                self.encoder = encoder
                self.projection = projection

            def forward(self, pixel_values):
                # pixel_values: (B, 3, 112, 1036) — mel spectrogram as "image"
                outputs = self.encoder(pixel_values=pixel_values)
                pooled = outputs[1]
                projected = self.projection(pooled)
                return projected / projected.norm(dim=-1, keepdim=True)

        self.module = _Module(
            model.modality_encoder["audio"],
            model.modality_proj["audio"],
        )
        self.module.eval()

    @property
    def dummy_input(self):
        import torch

        return torch.randn(1, 3, 112, 1036)

    @property
    def input_names(self):
        return ["pixel_values"]

    @property
    def dynamic_axes(self):
        return {"pixel_values": {0: "batch_size"}, "embeddings": {0: "batch_size"}}


class _TextEncoderWrapper:
    """Wraps text encoder + projection + L2 norm for ONNX export."""

    def __init__(self, model):
        import torch.nn as nn

        class _Module(nn.Module):
            def __init__(self, encoder, projection):
                super().__init__()
                self.encoder = encoder
                self.projection = projection

            def forward(self, input_ids, attention_mask):
                # input_ids: (B, 77), attention_mask: (B, 77)
                outputs = self.encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                pooled = outputs[1]
                projected = self.projection(pooled)
                return projected / projected.norm(dim=-1, keepdim=True)

        self.module = _Module(
            model.modality_encoder["language"],
            model.modality_proj["language"],
        )
        self.module.eval()

    @property
    def dummy_input(self):
        import torch

        return (
            torch.randint(0, 49408, (1, 77)),
            torch.ones(1, 77, dtype=torch.long),
        )

    @property
    def input_names(self):
        return ["input_ids", "attention_mask"]

    @property
    def dynamic_axes(self):
        return {
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "embeddings": {0: "batch_size"},
        }


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

ENCODER_MAP = {
    "video": (
        "languagebind-video-encoder.onnx",
        _VideoEncoderWrapper,
    ),
    "audio": (
        "languagebind-audio-encoder.onnx",
        _AudioEncoderWrapper,
    ),
    "text": (
        "languagebind-text-encoder.onnx",
        _TextEncoderWrapper,
    ),
}


def export_encoder(
    model,
    encoder_name: str,
    output_dir: Path,
    opset_version: int = 17,
) -> Path:
    """Export a single encoder to ONNX.

    Args:
        model: The loaded LanguageBind model.
        encoder_name: One of "video", "audio", "text".
        output_dir: Directory to write the .onnx file.
        opset_version: ONNX opset version (default 17).

    Returns:
        Path to the exported .onnx file.
    """
    import torch

    filename, wrapper_cls = ENCODER_MAP[encoder_name]
    wrapper = wrapper_cls(model)
    output_path = output_dir / filename

    dummy = wrapper.dummy_input
    if not isinstance(dummy, tuple):
        dummy = (dummy,)

    logger.info("Exporting %s encoder to %s ...", encoder_name, output_path)
    torch.onnx.export(
        wrapper.module,
        dummy,
        str(output_path),
        input_names=wrapper.input_names,
        output_names=["embeddings"],
        dynamic_axes=wrapper.dynamic_axes,
        opset_version=opset_version,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("  -> %s (%.1f MB)", output_path.name, size_mb)
    return output_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_encoder(
    model,
    encoder_name: str,
    output_dir: Path,
    atol: float = 1e-4,
) -> float:
    """Validate ONNX export matches PyTorch output.

    Returns the maximum absolute difference.
    """
    import onnxruntime as ort
    import torch

    filename, wrapper_cls = ENCODER_MAP[encoder_name]
    wrapper = wrapper_cls(model)
    onnx_path = output_dir / filename

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    # PyTorch reference output
    dummy = wrapper.dummy_input
    if not isinstance(dummy, tuple):
        dummy = (dummy,)

    with torch.no_grad():
        pt_output = wrapper.module(*dummy).numpy()

    # ONNX Runtime output
    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    ort_inputs = {
        name: tensor.numpy() for name, tensor in zip(wrapper.input_names, dummy)
    }
    ort_output = sess.run(None, ort_inputs)[0]

    max_diff = float(np.abs(pt_output - ort_output).max())
    mean_diff = float(np.abs(pt_output - ort_output).mean())

    logger.info(
        "  %s encoder: max_diff=%.6f, mean_diff=%.6f (atol=%.1e)",
        encoder_name,
        max_diff,
        mean_diff,
        atol,
    )

    if max_diff > atol:
        raise ValueError(
            f"{encoder_name} encoder validation FAILED: "
            f"max_diff={max_diff:.6f} > atol={atol}"
        )

    return max_diff


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Export LanguageBind encoders to ONNX format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --output-dir ./exported_models/\n"
            "  %(prog)s --output-dir ./exported_models/ --encoder video\n"
            "  %(prog)s --output-dir ./exported_models/ --validate-only\n"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./exported_models",
        help="Directory for exported ONNX files (default: ./exported_models)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["video", "audio", "text"],
        help="Export a single encoder (default: all three)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip export, only validate existing ONNX files",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for validation (default: 1e-4)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache_dir",
        help="HuggingFace cache directory for model downloads",
    )
    parser.add_argument(
        "--save-tokenizer",
        action="store_true",
        default=True,
        help="Save the tokenizer alongside ONNX files (default: True)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    _check_dependencies()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoders = [args.encoder] if args.encoder else ["video", "audio", "text"]

    logger.info("Loading LanguageBind models...")
    model, tokenizer = load_languagebind(cache_dir=args.cache_dir)

    if not args.validate_only:
        for name in encoders:
            export_encoder(
                model, name, output_dir, opset_version=args.opset_version
            )

    logger.info("Validating exports...")
    for name in encoders:
        validate_encoder(model, name, output_dir, atol=args.atol)

    if args.save_tokenizer and not args.validate_only:
        tokenizer_dir = output_dir / "tokenizer"
        tokenizer.save_pretrained(str(tokenizer_dir))
        logger.info("Saved tokenizer to %s", tokenizer_dir)

    logger.info("Done! Exported files:")
    for f in sorted(output_dir.glob("*.onnx")):
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info("  %s (%.1f MB)", f.name, size_mb)


if __name__ == "__main__":
    main()
