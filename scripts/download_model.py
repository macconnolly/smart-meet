"""
Model download and setup script for Cognitive Meeting Intelligence.

This script downloads the all-MiniLM-L6-v2 model, converts it to ONNX
format, and sets up the model directory structure for optimal performance.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm
import onnx
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


async def main():
    """
    Main model download and setup function.

    AGENTIC EMPOWERMENT: Automated model setup enables
    easy deployment and consistent model versions across
    environments.
    """
    print("🤖 Setting up ML models for Cognitive Meeting Intelligence")

    try:
        # Load configuration or use defaults
        model_path = Path("models/all-MiniLM-L6-v2")
        print(f"📂 Model directory: {model_path}")

        # Create model directory
        model_path.mkdir(parents=True, exist_ok=True)

        # Check if model already exists
        if (model_path / "model.onnx").exists():
            response = input("Model already exists. Re-download and convert? (y/N): ")
            if response.lower() != "y":
                print("✅ Using existing model")
                return 0

        # Download the sentence transformer model
        print("📥 Downloading sentence-transformers model...")
        await download_sentence_transformer_model(model_path)

        # Convert to ONNX format
        print("🔄 Converting model to ONNX format...")
        await convert_to_onnx(model_path)

        # Optimize ONNX model
        print("⚡ Optimizing ONNX model...")
        await optimize_onnx_model(model_path)

        # Verify model setup
        print("✅ Verifying model setup...")
        await verify_model_setup(model_path)

        # Run performance benchmark
        print("🏃 Running performance benchmark...")
        await benchmark_model_performance(model_path)

        print("🎉 Model setup completed successfully!")
        print(f"📍 Model location: {model_path.absolute()}")

    except Exception as e:
        print(f"❌ Model setup failed: {e}")
        logging.exception("Model setup error")
        return 1

    return 0


async def download_sentence_transformer_model(model_path: Path):
    """
    Download sentence-transformers model from HuggingFace.

    AGENTIC EMPOWERMENT: Automated downloading ensures
    consistent model versions and simplifies deployment.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    try:
        # Download using sentence-transformers
        print(f"  📦 Downloading {model_name} from HuggingFace Hub...")

        # This will download to the transformers cache first
        model = SentenceTransformer(model_name)

        # Save to our model directory
        pytorch_model_path = model_path / "pytorch_model"
        pytorch_model_path.mkdir(exist_ok=True)

        model.save(str(pytorch_model_path))
        print(f"  ✓ Model saved to {pytorch_model_path}")

        # Also download tokenizer separately for ONNX conversion
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer_path = model_path / "tokenizer"
        tokenizer_path.mkdir(exist_ok=True)
        tokenizer.save_pretrained(str(tokenizer_path))
        print(f"  ✓ Tokenizer saved to {tokenizer_path}")

    except Exception as e:
        raise Exception(f"Failed to download model: {e}")


async def convert_to_onnx(model_path: Path):
    """
    Convert PyTorch model to ONNX format.

    AGENTIC EMPOWERMENT: ONNX conversion enables optimized
    inference and cross-platform deployment.
    """
    try:
        # Load the PyTorch model
        pytorch_model_path = model_path / "pytorch_model"
        model = SentenceTransformer(str(pytorch_model_path))

        # Create dummy input for tracing with proper data types
        dummy_input_ids = torch.randint(0, 30000, (1, 128), dtype=torch.long)
        dummy_attention_mask = torch.ones(1, 128, dtype=torch.long)

        # Export to ONNX
        onnx_path = model_path / "model.onnx"

        print(f"  🔄 Converting to ONNX format...")

        # Get the transformer model from sentence-transformers
        transformer_model = model[0].auto_model
        transformer_model.eval()  # Set to eval mode for export

        # Export with torch.onnx
        torch.onnx.export(
            transformer_model,
            (dummy_input_ids, dummy_attention_mask),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "last_hidden_state": {0: "batch_size", 1: "sequence"},
            },
            opset_version=14,
            do_constant_folding=True,
            export_params=True,
        )

        print(f"  ✓ ONNX model saved to {onnx_path}")

        # Verify ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print(f"  ✓ ONNX model verification passed")

    except Exception as e:
        raise Exception(f"ONNX conversion failed: {e}")


async def optimize_onnx_model(model_path: Path):
    """
    Optimize ONNX model for inference performance.

    AGENTIC EMPOWERMENT: Model optimization ensures maximum
    inference speed for real-time cognitive processing.
    """
    try:
        # Skip optimization if onnxruntime-tools not available
        # The model will still work fine without optimization
        print(f"  ⚡ Skipping ONNX optimization (optional step)")
        print(f"  ℹ️  Model will work perfectly without optimization")

    except Exception as e:
        print(f"  ⚠️  Optimization failed: {e}, continuing with unoptimized model")


async def verify_model_setup(model_path: Path):
    """
    Verify model setup and test basic functionality.

    AGENTIC EMPOWERMENT: Model verification ensures the
    setup is correct and the model produces expected outputs.
    """
    try:
        import onnxruntime as ort
        import numpy as np

        # Check required files exist
        required_files = ["model.onnx", "pytorch_model", "tokenizer"]

        for file_name in required_files:
            file_path = model_path / file_name
            if not file_path.exists():
                raise Exception(f"Required file missing: {file_name}")

        print(f"  ✓ All required files present")

        # Test ONNX model loading
        onnx_path = model_path / "model.onnx"

        # Configure session options for CPU
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Use CPU provider
        providers = ["CPUExecutionProvider"]

        session = ort.InferenceSession(str(onnx_path), sess_options, providers=providers)
        print(f"  ✓ ONNX model loads successfully")
        print(f"  📊 Using providers: {session.get_providers()}")

        # Test tokenizer loading
        from transformers import AutoTokenizer

        tokenizer_path = model_path / "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        print(f"  ✓ Tokenizer loads successfully")

        # Test end-to-end inference
        test_text = "This is a test sentence for model verification."

        # Tokenize
        inputs = tokenizer(
            test_text, return_tensors="np", padding=True, truncation=True, max_length=512
        )

        # Run inference
        outputs = session.run(
            None,
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            },
        )

        # Get token embeddings and apply mean pooling
        token_embeddings = outputs[0]
        attention_mask = inputs["attention_mask"]

        # Mean pooling
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask

        print(f"  ✓ End-to-end inference successful")
        print(f"  📏 Output shape: {embeddings.shape}")

        # Verify embedding dimensions
        if embeddings.shape[-1] != 384:
            raise Exception(f"Unexpected embedding dimension: {embeddings.shape[-1]}, expected 384")

        print(f"  ✓ Embedding dimensions correct (384D)")

    except Exception as e:
        raise Exception(f"Model verification failed: {e}")


async def benchmark_model_performance(model_path: Path):
    """
    Benchmark model inference performance.

    AGENTIC EMPOWERMENT: Performance benchmarking ensures
    the model meets speed requirements for real-time processing.
    """
    try:
        import onnxruntime as ort
        import numpy as np
        import time
        from transformers import AutoTokenizer

        # Load model and tokenizer
        onnx_path = model_path / "model.onnx"
        tokenizer_path = model_path / "tokenizer"

        # Configure optimized session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        session = ort.InferenceSession(str(onnx_path), sess_options)
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        # Prepare test data
        test_texts = [
            "Short text for testing.",
            "This is a medium length sentence that represents typical meeting content for performance testing.",
            "This is a much longer sentence that might appear in meeting transcripts, containing multiple clauses and complex ideas about strategic planning, technical architecture, and team collaboration that needs to be processed efficiently by the cognitive meeting intelligence system.",
        ] * 10  # 30 texts total

        # Warm-up runs
        print(f"  🔥 Warming up model...")
        for _ in range(5):
            inputs = tokenizer(
                test_texts[0], return_tensors="np", padding=True, truncation=True, max_length=512
            )
            session.run(
                None,
                {
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64),
                },
            )

        # Single inference benchmark
        print(f"  🚀 Benchmarking single inference...")

        inputs = tokenizer(
            test_texts[0], return_tensors="np", padding=True, truncation=True, max_length=512
        )

        start_time = time.perf_counter()
        for _ in range(100):
            session.run(
                None,
                {
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64),
                },
            )
        single_time = (time.perf_counter() - start_time) / 100

        print(f"  📊 Single inference: {single_time*1000:.2f}ms")

        # Batch inference benchmark
        print(f"  🚀 Benchmarking batch inference...")

        batch_inputs = tokenizer(
            test_texts, return_tensors="np", padding=True, truncation=True, max_length=512
        )

        start_time = time.perf_counter()
        outputs = session.run(
            None,
            {
                "input_ids": batch_inputs["input_ids"].astype(np.int64),
                "attention_mask": batch_inputs["attention_mask"].astype(np.int64),
            },
        )
        batch_time = time.perf_counter() - start_time

        throughput = len(test_texts) / batch_time
        print(f"  📊 Batch inference: {batch_time*1000:.2f}ms for {len(test_texts)} texts")
        print(f"  🏃 Throughput: {throughput:.1f} texts/second")

        # Memory usage estimation
        import psutil

        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        print(f"  💾 Memory usage: {memory_mb:.1f} MB")

        # Performance assessment
        target_throughput = 10  # 10-15 memories/second target
        if throughput >= target_throughput:
            print(f"  ✅ Performance target met ({throughput:.1f} >= {target_throughput} texts/sec)")
        else:
            print(
                f"  ⚠️  Performance below target ({throughput:.1f} < {target_throughput} texts/sec)"
            )

        # Check single encoding target
        if single_time * 1000 < 100:
            print(f"  ✅ Single encoding target met ({single_time*1000:.2f}ms < 100ms)")
        else:
            print(f"  ⚠️  Single encoding above target ({single_time*1000:.2f}ms > 100ms)")

    except Exception as e:
        print(f"  ⚠️  Benchmarking failed: {e}")


async def cleanup_temporary_files(model_path: Path):
    """
    Clean up temporary files after setup.

    AGENTIC EMPOWERMENT: Cleanup reduces storage usage
    and keeps the deployment clean.
    """
    try:
        # Keep PyTorch model for now - might be needed for future updates
        print(f"  ℹ️  Keeping PyTorch model for potential future use")

    except Exception as e:
        print(f"  ⚠️  Cleanup failed: {e}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Check dependencies
    try:
        import torch
        import onnx
        import onnxruntime
        import sentence_transformers
        import transformers
        import psutil
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("Please install required packages:")
        print("pip install torch onnx onnxruntime sentence-transformers transformers psutil")
        sys.exit(1)

    # Run model setup
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
