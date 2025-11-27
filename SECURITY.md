# Security Guidelines

## API Key Management

**IMPORTANT: Never commit API keys or secrets to git!**

### ✅ Safe Practices

1. **Use `.env` file** (already in `.gitignore`):
   ```bash
   cp .env.example .env
   # Edit .env with your actual keys
   ```

2. **Use environment variables**:
   ```bash
   export AZURE_OPENAI_API_KEY="your-key-here"
   ```

3. **Use `getpass` in notebooks** for interactive prompts:
   ```python
   from getpass import getpass
   os.environ["API_KEY"] = getpass("Enter API key: ")
   ```

### ❌ Never Do This

- ❌ Hardcode API keys in code
- ❌ Commit `.env` files
- ❌ Share API keys in screenshots or documentation
- ❌ Store keys in version control

### What Was Fixed

The notebook previously contained hardcoded API keys. These have been removed and replaced with:
- Environment variable loading from `.env` file
- Secure prompts using `getpass`
- Fallback to system environment variables

### If Keys Are Exposed

If you accidentally commit API keys:

1. **Immediately rotate/revoke the exposed keys**
2. **Remove from git history**:
   ```bash
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch notebooks/deepRAG.ipynb" \
     --prune-empty --tag-name-filter cat -- --all
   ```
3. **Force push** (coordinate with team):
   ```bash
   git push origin --force --all
   ```

### Current Status

✅ All hardcoded API keys have been removed from the notebook
✅ `.env` file is in `.gitignore`
✅ `.env.example` provides a template without real keys

