# WSL Development Guide for Cognitive Meeting Intelligence

## Quick Start for WSL

```bash
# Make the setup script executable
chmod +x setup_wsl.sh

# Run the setup
./setup_wsl.sh
```

## WSL-Specific Best Practices

### 1. **File System Performance**

**IMPORTANT**: Store your project in the WSL file system, NOT in `/mnt/c/`

```bash
# Good - Native WSL filesystem (fast)
cd ~
mkdir -p dev
cd dev
git clone <your-repo> meet

# Bad - Windows filesystem (slow)
cd /mnt/c/Users/username/dev/meet  # Don't do this!
```

### 2. **Python Setup in WSL**

WSL Ubuntu typically comes with Python, but you might need 3.11+:

```bash
# Check Python version
python3 --version

# If you need Python 3.11
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Set Python 3.11 as default (optional)
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
```

### 3. **Virtual Environment (WSL Style)**

```bash
# Create venv with specific Python version
python3.11 -m venv venv

# Activate (no .exe or .bat files in WSL!)
source venv/bin/activate

# Your prompt should change to show (venv)

# Install requirements
pip install -r requirements.txt
```

### 4. **Docker in WSL2**

You have two options:

**Option A: Docker Desktop Integration (Recommended)**
- Install Docker Desktop on Windows
- Enable WSL2 integration in Docker Desktop settings
- Docker commands work seamlessly in WSL

**Option B: Native Docker in WSL**
```bash
# Install Docker in WSL
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# Test
docker run hello-world
```

### 5. **Running the Project**

```bash
# Start all services with Docker
docker-compose up -d

# Or run natively
source venv/bin/activate
uvicorn src.api.simple_api:app --reload

# Run tests
pytest -v

# Or with Docker
docker-compose exec api pytest
```

### 6. **Common WSL Issues & Solutions**

#### Permission Issues
```bash
# If you get permission denied
sudo chown -R $USER:$USER .

# For Docker socket
sudo chmod 666 /var/run/docker.sock
```

#### Line Ending Issues
```bash
# Configure Git for WSL
git config --global core.autocrlf input

# Fix line endings if needed
dos2unix setup_wsl.sh
```

#### Port Access from Windows
- Services running in WSL2 are accessible from Windows at `localhost:port`
- If not working, check Windows Firewall

#### Memory Issues
Create `.wslconfig` in Windows user home:
```ini
[wsl2]
memory=8GB
processors=4
swap=2GB
```

### 7. **VS Code Integration**

```bash
# Install VS Code WSL extension, then:
code .

# This opens VS Code connected to WSL
# All terminal commands run in WSL
# Extensions run in WSL
```

### 8. **Quick Commands Cheatsheet**

```bash
# Activate environment
source venv/bin/activate

# Run API
uvicorn src.api.simple_api:app --reload

# Run tests
pytest
pytest tests/unit -v
pytest --cov=src

# Docker commands
docker-compose up -d
docker-compose logs -f
docker-compose exec api bash
docker-compose down

# Database
sqlite3 data/memories.db ".tables"
```

### 9. **Performance Tips**

1. **Use WSL2** (not WSL1):
   ```bash
   wsl --set-version Ubuntu 2
   ```

2. **Store files in WSL**:
   - `~/dev/meet` ✅ Fast
   - `/mnt/c/dev/meet` ❌ Slow

3. **Exclude from Windows Defender**:
   Add WSL directories to Windows Defender exclusions

4. **Use Docker BuildKit**:
   ```bash
   export DOCKER_BUILDKIT=1
   ```

### 10. **Testing Workflow**

```bash
# 1. Quick test single file
pytest tests/test_memory_repo.py -v

# 2. Test with coverage
pytest --cov=src --cov-report=html
# Open coverage/html/index.html in browser

# 3. Run specific test
pytest -k "test_memory_creation"

# 4. Debug mode
pytest -vv -s  # verbose + print statements

# 5. Parallel testing
pip install pytest-xdist
pytest -n auto
```

## Quick Diagnostic Script

Save as `wsl_check.py`:

```python
#!/usr/bin/env python3
import os
import sys
import subprocess
import platform

print("WSL Environment Check")
print("=" * 40)

# Check if running in WSL
if "microsoft" in platform.uname().release.lower():
    print("✅ Running in WSL")
else:
    print("❌ Not running in WSL")

# Check Python
print(f"\nPython: {sys.version}")
print(f"Executable: {sys.executable}")

# Check file location
cwd = os.getcwd()
if cwd.startswith("/mnt/"):
    print(f"\n⚠️  WARNING: Project in Windows filesystem: {cwd}")
    print("   Consider moving to WSL filesystem for better performance")
else:
    print(f"\n✅ Project in WSL filesystem: {cwd}")

# Check virtual environment
if os.environ.get("VIRTUAL_ENV"):
    print(f"\n✅ Virtual environment active: {os.environ['VIRTUAL_ENV']}")
else:
    print("\n❌ No virtual environment active")

# Check Docker
try:
    subprocess.run(["docker", "--version"], capture_output=True, check=True)
    print("\n✅ Docker is available")
except:
    print("\n❌ Docker not found")

# Check key services
import socket

def check_port(port, service):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    if result == 0:
        print(f"✅ {service} is running on port {port}")
    else:
        print(f"❌ {service} is not running on port {port}")

print("\nService Status:")
check_port(8000, "API")
check_port(6333, "Qdrant")
```

## Need Help?

1. Run the diagnostic: `python3 wsl_check.py`
2. Use the setup script: `./setup_wsl.sh`
3. Check service logs: `docker-compose logs -f`
4. Verify environment: `pip list | grep -E "fastapi|qdrant|pytest"`
