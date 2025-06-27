# 🐍 Virtual Environment Quick Reference

## 🚀 Initial Setup (One Time)
```bash
# Run this once to create venvs for all worktrees
./setup_worktree_venvs.sh
```

## 📂 Venv Locations
Each worktree has its own isolated Python environment:
- **Main**: `/mnt/c/Users/EL436GA/dev/meet/venv`
- **Tests**: `worktree-tests/venv`
- **Day 1**: `worktree-day1/venv`
- **Day 2**: `worktree-day2/venv`
- **Day 3**: `worktree-day3/venv`

## 🔧 Activation Commands

### Quick Activation (from anywhere)
```bash
# For main
source /mnt/c/Users/EL436GA/dev/meet/venv/bin/activate

# For test worktree
source /mnt/c/Users/EL436GA/dev/meet/worktree-tests/venv/bin/activate

# For day 1
source /mnt/c/Users/EL436GA/dev/meet/worktree-day1/venv/bin/activate

# For day 2
source /mnt/c/Users/EL436GA/dev/meet/worktree-day2/venv/bin/activate

# For day 3
source /mnt/c/Users/EL436GA/dev/meet/worktree-day3/venv/bin/activate
```

### Standard Activation (when in worktree)
```bash
cd worktree-tests
source venv/bin/activate
# or
./activate_venv.sh
```

## ⚡ Quick Commands

### Check active environment
```bash
which python
# Should show: /path/to/worktree/venv/bin/python
```

### Deactivate current venv
```bash
deactivate
```

### Install new package
```bash
# Make sure venv is activated first!
pip install package-name
```

### Update requirements
```bash
# After installing new packages
pip freeze > requirements.txt
```

## 🎯 Best Practices

1. **ALWAYS** activate the venv before starting work in a worktree
2. **NEVER** install packages globally - use the worktree's venv
3. **CHECK** you're in the right venv with `which python`
4. **COMMIT** requirements.txt changes when adding packages

## 🔍 Troubleshooting

### "Command not found" errors
```bash
# You forgot to activate the venv!
source venv/bin/activate
```

### Wrong Python version
```bash
# Check which Python is active
which python
python --version
```

### Package conflicts
```bash
# Recreate the venv
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 💡 Pro Tips

1. Add to your shell profile for quick activation:
```bash
alias venv='source venv/bin/activate'
alias vtest='source /mnt/c/Users/EL436GA/dev/meet/worktree-tests/venv/bin/activate'
alias vday1='source /mnt/c/Users/EL436GA/dev/meet/worktree-day1/venv/bin/activate'
```

2. VS Code / Claude Code will auto-activate if you open from the worktree:
```bash
cd worktree-tests
claude code .  # Will detect and use the local venv
```

3. Each venv is independent - packages installed in one won't affect others!