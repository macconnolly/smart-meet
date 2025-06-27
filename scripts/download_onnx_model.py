"""
Download and setup ONNX model for embeddings.

This script downloads the all-MiniLM-L6-v2 model converted to ONNX format
and sets up the necessary directory structure.
"""

import json
import logging
from pathlib import Path
import urllib.request
from urllib.error import URLError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_INFO = {
    "name": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dimension": 384,
    "max_seq_length": 256,
    "pooling": "mean",
    "normalize": True,
}

# URLs for downloading model components
MODEL_URLS = {
    "onnx_model": (
        "https://huggingface.co/sentence-transformers/"
        "all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
    ),
    "tokenizer_config": (
        "https://huggingface.co/sentence-transformers/"
        "all-MiniLM-L6-v2/resolve/main/tokenizer_config.json"
    ),
    "vocab": (
        "https://huggingface.co/sentence-transformers/"
        "all-MiniLM-L6-v2/resolve/main/vocab.txt"
    ),
    "special_tokens_map": (
        "https://huggingface.co/sentence-transformers/"
        "all-MiniLM-L6-v2/resolve/main/special_tokens_map.json"
    ),
    "config": (
        "https://huggingface.co/sentence-transformers/"
        "all-MiniLM-L6-v2/resolve/main/config.json"
    ),
}


def download_file(url: str, destination: Path, description: str) -> bool:
    """
    Download a file from URL to destination.

    Args:
        url: URL to download from
        destination: Path to save the file
        description: Description for logging

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading {description} from {url}")

        # Create parent directory if it doesn't exist
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress reporting
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100) if total_size > 0 else 0
            print(f"\rProgress: {percent:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(url, destination, download_progress)
        print()  # New line after progress

        logger.info(f"Successfully downloaded {description}")
        return True

    except URLError as e:
        logger.error(f"Failed to download {description}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading {description}: {e}")
        return False


def setup_model_directory(base_path: Path = Path("models/embeddings")) -> Path:
    """
    Set up the model directory structure.

    Args:
        base_path: Base path for models

    Returns:
        Path to the model directory
    """
    model_dir = base_path
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    tokenizer_dir = model_dir / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)

    logger.info(f"Model directory set up at: {model_dir}")
    return model_dir


def download_onnx_model(model_dir: Path) -> bool:
    """
    Download the ONNX model file.

    Args:
        model_dir: Directory to save the model

    Returns:
        True if successful, False otherwise
    """
    model_path = model_dir / "model.onnx"

    # Skip if already exists
    if model_path.exists():
        logger.info(f"ONNX model already exists at {model_path}")
        return True

    return download_file(MODEL_URLS["onnx_model"], model_path, "ONNX model")


def download_tokenizer_files(model_dir: Path) -> bool:
    """
    Download tokenizer files.

    Args:
        model_dir: Directory to save the tokenizer

    Returns:
        True if all files downloaded successfully
    """
    tokenizer_dir = model_dir / "tokenizer"

    files_to_download = [
        ("tokenizer_config", "tokenizer_config.json"),
        ("vocab", "vocab.txt"),
        ("special_tokens_map", "special_tokens_map.json"),
        ("config", "config.json"),
    ]

    success = True
    for url_key, filename in files_to_download:
        file_path = tokenizer_dir / filename

        # Skip if already exists
        if file_path.exists():
            logger.info(f"{filename} already exists")
            continue

        if not download_file(MODEL_URLS[url_key], file_path, filename):
            success = False

    # Create tokenizer.json for compatibility
    if success:
        create_tokenizer_json(tokenizer_dir)

    return success


def create_tokenizer_json(tokenizer_dir: Path) -> None:
    """
    Create a minimal tokenizer.json for compatibility.

    Args:
        tokenizer_dir: Directory containing tokenizer files
    """
    tokenizer_json = {
        "version": "1.0",
        "truncation": {"max_length": 256, "strategy": "LongestFirst"},
        "padding": {"strategy": "BatchLongest", "pad_token": "[PAD]"},
    }

    tokenizer_json_path = tokenizer_dir / "tokenizer.json"
    with open(tokenizer_json_path, "w") as f:
        json.dump(tokenizer_json, f, indent=2)

    logger.info("Created tokenizer.json")


def create_model_config(model_dir: Path) -> None:
    """
    Create model configuration file.

    Args:
        model_dir: Directory to save the config
    """
    config_path = model_dir / "config.json"

    with open(config_path, "w") as f:
        json.dump(MODEL_INFO, f, indent=2)

    logger.info(f"Created model config at {config_path}")


def verify_installation(model_dir: Path) -> bool:
    """
    Verify that all required files are present.

    Args:
        model_dir: Model directory to verify

    Returns:
        True if all files present, False otherwise
    """
    required_files = [
        model_dir / "model.onnx",
        model_dir / "config.json",
        model_dir / "tokenizer" / "tokenizer_config.json",
        model_dir / "tokenizer" / "vocab.txt",
        model_dir / "tokenizer" / "special_tokens_map.json",
        model_dir / "tokenizer" / "config.json",
    ]

    missing_files = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(str(file_path))

    if missing_files:
        logger.error(f"Missing files: {', '.join(missing_files)}")
        return False

    logger.info("All required files are present")
    return True


def main():
    """Main function to download and setup the ONNX model."""
    logger.info("Starting ONNX model setup")

    # Setup directory structure
    model_dir = setup_model_directory()

    # Download ONNX model
    if not download_onnx_model(model_dir):
        logger.error("Failed to download ONNX model")
        return False

    # Download tokenizer files
    if not download_tokenizer_files(model_dir):
        logger.error("Failed to download tokenizer files")
        return False

    # Create model config
    create_model_config(model_dir)

    # Verify installation
    if verify_installation(model_dir):
        logger.info("ONNX model setup completed successfully!")
        logger.info(f"Model location: {model_dir.absolute()}")

        # Print model info
        print("\nModel Information:")
        print(f"  Name: {MODEL_INFO['name']}")
        print(f"  Embedding dimension: {MODEL_INFO['embedding_dimension']}")
        print(f"  Max sequence length: {MODEL_INFO['max_seq_length']}")
        print(f"  Pooling: {MODEL_INFO['pooling']}")
        print(f"  Normalize: {MODEL_INFO['normalize']}")

        return True
    else:
        logger.error("Model setup verification failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
