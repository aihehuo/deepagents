# Installing DeepAgents CLI in Editable Mode

## ✅ Successfully Installed!

Your local `deepagents-cli` is now installed in **editable mode**. This means:

- ✅ Changes to code in `libs/deepagents-cli/` are **immediately available**
- ✅ No need to reinstall after making changes
- ✅ The `deepagents` command uses your local code

## Verification

You can verify it's using local code:

```bash
python3 -c "import deepagents_cli; import os; print(os.path.dirname(deepagents_cli.__file__))"
# Should show: /Users/yc/workspace/deepagents/libs/deepagents-cli/deepagents_cli
```

## How to Make Changes

1. **Edit code** in `libs/deepagents-cli/deepagents_cli/`
2. **Run `deepagents`** - changes are immediately available!
3. **No reinstall needed** - that's the beauty of editable mode

## Testing Your Changes

### Example: Add a Print Statement

1. Edit `libs/deepagents-cli/deepagents_cli/main.py`:
   ```python
   def cli_main() -> None:
       print("🔧 Using LOCAL version!")  # Add this line
       # ... rest of the function
   ```

2. Run the CLI:
   ```bash
   deepagents help
   # You'll see your print statement immediately!
   ```

### Example: Modify Help Text

1. Edit the help text in `libs/deepagents-cli/deepagents_cli/main.py`
2. Run `deepagents help` - see changes immediately

## Reinstalling (If Needed)

If you ever need to reinstall:

```bash
# Uninstall first
python3 -m pip uninstall deepagents-cli -y

# Install local version in editable mode
cd libs/deepagents-cli
python3 -m pip install -e .
```

## How Editable Install Works

When you run `pip install -e .`:

1. **Creates a link** in `site-packages` pointing to your local directory
2. **Python imports from your local code** instead of a copy
3. **Changes are immediately visible** because Python reads directly from source

The `.pth` file or egg-link in `site-packages` tells Python:
```
When someone imports deepagents_cli, look in:
/Users/yc/workspace/deepagents/libs/deepagents-cli/
```

## Troubleshooting

### Changes Not Appearing?

1. **Check you're using the right Python**:
   ```bash
   which python3
   which deepagents
   # Should be in the same environment
   ```

2. **Verify editable install**:
   ```bash
   python3 -m pip show -f deepagents-cli | grep Location
   # Should show your local directory
   ```

3. **Reinstall if needed**:
   ```bash
   cd libs/deepagents-cli
   python3 -m pip install -e . --force-reinstall
   ```

### Command Not Found After Reinstall?

Make sure your Python's `bin/` directory is in PATH:
```bash
echo $PATH | grep -o '[^:]*bin' | head -5
# Should include your Python's bin directory
```

## Summary

- ✅ **Uninstalled**: Remote version removed
- ✅ **Installed**: Local version in editable mode
- ✅ **Ready**: Changes to code are immediately available
- 🚀 **Usage**: Just run `deepagents` and see your changes!

