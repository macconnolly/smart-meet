"""
Windows Setup Helper Script for Cognitive Meeting Intelligence
This script provides multiple options for setting up your development environment
"""

import os
import sys
import subprocess
import platform

def check_command_exists(command):
    """Check if a command exists in the system"""
    try:
        subprocess.run([command, "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def setup_docker():
    """Setup and run with Docker"""
    print("\n🐳 Setting up with Docker...")
    if not check_command_exists("docker"):
        print("❌ Docker not installed. Please install Docker Desktop for Windows.")
        print("   Download from: https://www.docker.com/products/docker-desktop/")
        return False
    
    print("✅ Docker found. Starting containers...")
    subprocess.run(["docker-compose", "up", "-d"], check=True)
    print("\n✅ Docker setup complete!")
    print("\nTo run tests:")
    print("  docker-compose exec api pytest")
    print("\nTo view logs:")
    print("  docker-compose logs -f")
    return True

def setup_poetry():
    """Setup with Poetry"""
    print("\n📜 Setting up with Poetry...")
    if not check_command_exists("poetry"):
        print("❌ Poetry not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "poetry"], check=True)
    
    print("✅ Poetry found. Installing dependencies...")
    subprocess.run(["poetry", "install"], check=True)
    print("\n✅ Poetry setup complete!")
    print("\nTo run tests:")
    print("  poetry run pytest")
    print("\nTo activate shell:")
    print("  poetry shell")
    return True

def setup_conda():
    """Setup with Conda"""
    print("\n🐍 Setting up with Conda...")
    if not check_command_exists("conda"):
        print("❌ Conda not installed. Please install Miniconda or Anaconda.")
        print("   Miniconda (recommended): https://docs.conda.io/en/latest/miniconda.html")
        return False
    
    env_name = "cognitive-meet"
    print(f"Creating conda environment '{env_name}'...")
    
    # Create environment file
    conda_env = """name: cognitive-meet
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
    - -r requirements.txt
"""
    with open("environment.yml", "w") as f:
        f.write(conda_env)
    
    subprocess.run(["conda", "env", "create", "-f", "environment.yml"], check=True)
    print(f"\n✅ Conda setup complete!")
    print(f"\nTo activate environment:")
    print(f"  conda activate {env_name}")
    return True

def setup_venv_improved():
    """Improved venv setup with better error handling"""
    print("\n🔧 Setting up improved virtual environment...")
    
    venv_path = "venv"
    if os.path.exists(venv_path):
        print("Removing existing venv...")
        import shutil
        shutil.rmtree(venv_path)
    
    print("Creating fresh virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
    
    # Determine pip path based on OS
    if platform.system() == "Windows":
        pip_path = os.path.join(venv_path, "Scripts", "pip.exe")
        activate_cmd = os.path.join(venv_path, "Scripts", "activate.bat")
    else:
        pip_path = os.path.join(venv_path, "bin", "pip")
        activate_cmd = f"source {os.path.join(venv_path, 'bin', 'activate')}"
    
    # Upgrade pip first
    print("Upgrading pip...")
    subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
    
    # Install wheel and setuptools
    print("Installing build tools...")
    subprocess.run([pip_path, "install", "--upgrade", "wheel", "setuptools"], check=True)
    
    # Install requirements
    print("Installing requirements...")
    subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
    
    # Download NLTK data
    print("Downloading NLTK data...")
    python_path = os.path.join(venv_path, "Scripts", "python.exe") if platform.system() == "Windows" else os.path.join(venv_path, "bin", "python")
    subprocess.run([python_path, "-c", "import nltk; nltk.download('vader_lexicon')"], check=True)
    
    print(f"\n✅ Virtual environment setup complete!")
    print(f"\nTo activate:")
    print(f"  {activate_cmd}")
    return True

def run_quick_test():
    """Run a quick test to verify setup"""
    print("\n🧪 Running quick verification test...")
    test_script = """
import sys
print(f"Python: {sys.version}")

try:
    import fastapi
    print("✅ FastAPI installed")
except ImportError:
    print("❌ FastAPI not found")

try:
    import qdrant_client
    print("✅ Qdrant client installed")
except ImportError:
    print("❌ Qdrant client not found")

try:
    import onnxruntime
    print("✅ ONNX Runtime installed")
except ImportError:
    print("❌ ONNX Runtime not found")

try:
    import sentence_transformers
    print("✅ Sentence Transformers installed")
except ImportError:
    print("❌ Sentence Transformers not found")
"""
    
    with open("test_setup.py", "w") as f:
        f.write(test_script)
    
    subprocess.run([sys.executable, "test_setup.py"])
    os.remove("test_setup.py")

def main():
    print("🚀 Cognitive Meeting Intelligence - Setup Helper")
    print("=" * 50)
    print("\nChoose your setup method:")
    print("1. Docker (Recommended - Isolated, no dependency issues)")
    print("2. Poetry (Modern Python dependency management)")
    print("3. Conda/Miniconda (Good for ML/Data Science)")
    print("4. Improved venv (Traditional with better setup)")
    print("5. Run verification test only")
    print("0. Exit")
    
    while True:
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            if setup_docker():
                break
        elif choice == "2":
            if setup_poetry():
                run_quick_test()
                break
        elif choice == "3":
            if setup_conda():
                break
        elif choice == "4":
            if setup_venv_improved():
                run_quick_test()
                break
        elif choice == "5":
            run_quick_test()
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
