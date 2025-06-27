"""
Setup script for downloading and converting the sentence-transformers model to ONNX format.

This script downloads the all-MiniLM-L6-v2 model and exports it to ONNX format
for efficient CPU inference in the embedding pipeline.

Usage:
    python scripts/setup_model.py [--model-name MODEL] [--output-dir DIR]
"""

import argparse
import logging
from pathlib import Path
import sys
import torch
import numpy as np
from typing import Dict, Any

# Try to import required libraries
try:
    from sentence_transformers import SentenceTransformer
    import onnx
    import onnxruntime as ort
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install: pip install sentence-transformers onnx onnxruntime")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_and_convert_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: str = "models/embeddings",
) -> None:
    """
    Download a sentence-transformers model and convert to ONNX format.

    Args:
        model_name: Name of the model to download
        output_dir: Directory to save the ONNX model
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading model: {model_name}")

        # Load the model
        model = SentenceTransformer(model_name)

        # Get model dimensions
        test_embedding = model.encode("test", convert_to_numpy=True)
        embedding_dim = test_embedding.shape[0]
        logger.info(f"Model embedding dimension: {embedding_dim}")

        # Save model config
        config = {
            "model_name": model_name,
            "embedding_dimension": int(embedding_dim),
            "max_seq_length": model.max_seq_length,
        }

        # Save config as JSON
        import json

        config_path = output_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved model config to: {config_path}")

        # Export to ONNX
        logger.info("Converting to ONNX format...")

        # Get the underlying transformer model
        auto_model = model[0].auto_model

        # Create dummy input
        dummy_input_ids = torch.ones(1, 128, dtype=torch.long)
        dummy_attention_mask = torch.ones(1, 128, dtype=torch.long)
        dummy_token_type_ids = torch.zeros(1, 128, dtype=torch.long)

        # Export the model
        onnx_path = output_path / "model.onnx"

        # Dynamic axes for variable sequence length
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "token_type_ids": {0: "batch_size", 1: "sequence"},
            "output": {0: "batch_size"},
        }

        torch.onnx.export(
            auto_model,
            (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

        logger.info(f"Exported ONNX model to: {onnx_path}")

        # Verify the ONNX model
        logger.info("Verifying ONNX model...")
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("✓ ONNX model verification passed")

        # Test inference
        logger.info("Testing ONNX inference...")
        ort_session = ort.InferenceSession(str(onnx_path))

        # Prepare test input
        test_input = {
            "input_ids": dummy_input_ids.numpy(),
            "attention_mask": dummy_attention_mask.numpy(),
            "token_type_ids": dummy_token_type_ids.numpy(),
        }

        # Run inference
        outputs = ort_session.run(None, test_input)
        output_shape = outputs[0].shape
        logger.info(f"✓ ONNX inference successful. Output shape: {output_shape}")

        # Save tokenizer
        logger.info("Saving tokenizer...")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer_path = output_path / "tokenizer"
        tokenizer.save_pretrained(str(tokenizer_path))
        logger.info(f"Saved tokenizer to: {tokenizer_path}")

        # Create a summary file
        summary_path = output_path / "model_info.txt"
        with open(summary_path, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Embedding Dimension: {embedding_dim}\n")
            f.write(f"Max Sequence Length: {model.max_seq_length}\n")
            f.write(f"ONNX Opset Version: 14\n")
            f.write(f"ONNX File: model.onnx\n")
            f.write(f"Tokenizer Directory: tokenizer/\n")

        logger.info("✅ Model setup completed successfully!")

    except Exception as e:
        logger.error(f"Failed to setup model: {e}")
        raise


def verify_model_files(output_dir: str = "models/embeddings") -> bool:
    """
    Verify that all required model files exist.

    Args:
        output_dir: Directory containing model files

    Returns:
        True if all files exist, False otherwise
    """
    output_path = Path(output_dir)

    required_files = [
        "model.onnx",
        "config.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.txt",
        "model_info.txt",
    ]

    all_exist = True
    for file_path in required_files:
        full_path = output_path / file_path
        if full_path.exists():
            logger.info(f"✓ Found: {file_path}")
        else:
            logger.error(f"✗ Missing: {file_path}")
            all_exist = False

    return all_exist


def benchmark_model(output_dir: str = "models/embeddings") -> None:
    """
    Benchmark the ONNX model performance.

    Args:
        output_dir: Directory containing model files
    """
    import time

    output_path = Path(output_dir)
    onnx_path = output_path / "model.onnx"

    if not onnx_path.exists():
        logger.error(f"ONNX model not found at: {onnx_path}")
        return

    logger.info("Benchmarking ONNX model performance...")

    # Load model
    ort_session = ort.InferenceSession(str(onnx_path))

    # Test sentences of various lengths
    test_sentences = [
        "Short sentence.",
        "This is a medium length sentence with more words to process.",
        "This is a much longer sentence that contains significantly more words and should take a bit more time to process through the embedding model, but still within acceptable performance bounds for our use case.",
    ]

    # Warm up
    dummy_input = {
        "input_ids": np.ones((1, 128), dtype=np.int64),
        "attention_mask": np.ones((1, 128), dtype=np.int64),
        "token_type_ids": np.zeros((1, 128), dtype=np.int64),
    }

    for _ in range(10):
        ort_session.run(None, dummy_input)

    # Benchmark
    for sentence in test_sentences:
        # Simple tokenization for benchmark (in practice, use proper tokenizer)
        seq_len = min(len(sentence.split()) * 2, 128)  # Rough estimate

        test_input = {
            "input_ids": np.ones((1, seq_len), dtype=np.int64),
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
            "token_type_ids": np.zeros((1, seq_len), dtype=np.int64),
        }

        # Time multiple runs
        num_runs = 100
        start_time = time.time()

        for _ in range(num_runs):
            outputs = ort_session.run(None, test_input)

        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs * 1000  # Convert to ms

        logger.info(f"Sentence length ~{len(sentence)} chars: {avg_time:.2f}ms per inference")

    logger.info("✓ Benchmark completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup ONNX embedding model for Cognitive Meeting Intelligence"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Name of the sentence-transformers model to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/embeddings",
        help="Directory to save the ONNX model",
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify existing model files"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmark after setup"
    )

    args = parser.parse_args()

    if args.verify_only:
        if verify_model_files(args.output_dir):
            logger.info("✅ All model files verified successfully!")
        else:
            logger.error("❌ Model verification failed!")
            sys.exit(1)
    else:
        download_and_convert_model(args.model_name, args.output_dir)

        if args.benchmark:
            benchmark_model(args.output_dir)


if __name__ == "__main__":
    main()
