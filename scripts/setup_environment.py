"""Script to set up the environment and verify configuration."""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config, ensure_directories


def check_environment():
    """Check if environment is properly configured."""
    print("=" * 60)
    print("ENVIRONMENT SETUP CHECK")
    print("=" * 60)
    
    # Check .env file
    env_file = project_root / ".env"
    if env_file.exists():
        print("✓ .env file exists")
    else:
        print("⚠ .env file not found")
        print("  Copy .env.example to .env and fill in your API keys")
    
    # Check required environment variables
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME"
    ]
    
    print("\nChecking required environment variables:")
    all_present = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✓ {var}: {'*' * min(len(value), 20)}")
        else:
            print(f"  ✗ {var}: NOT SET")
            all_present = False
    
    # Get and validate config
    try:
        config = get_config()
        ensure_directories(config)
        print("\n✓ Configuration loaded successfully")
        print(f"  Data directory: {config['data_dir']}")
        print(f"  Vector store directory: {config['vector_store_dir']}")
        print(f"  LLM Provider: {config['llm_provider']}")
        print(f"  Embedding Model: {config['embedding_model']}")
    except Exception as e:
        print(f"\n✗ Error loading configuration: {e}")
        all_present = False
    
    # Check data directory
    data_dir = project_root / "data"
    if data_dir.exists():
        pdf_files = list(data_dir.glob("*.pdf"))
        txt_files = list(data_dir.glob("*.txt"))
        print(f"\n✓ Data directory exists")
        print(f"  PDF files: {len(pdf_files)}")
        print(f"  Text files: {len(txt_files)}")
    else:
        print(f"\n⚠ Data directory does not exist")
        data_dir.mkdir(exist_ok=True)
        print(f"  Created data directory")
    
    print("\n" + "=" * 60)
    if all_present:
        print("✓ Environment setup complete!")
    else:
        print("⚠ Please fix the issues above before proceeding")
    print("=" * 60)


if __name__ == "__main__":
    check_environment()

