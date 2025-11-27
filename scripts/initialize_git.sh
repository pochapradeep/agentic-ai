#!/bin/bash
# Script to initialize git repository

echo "Initializing git repository..."

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    git init
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already exists"
fi

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Agentic AI Deep RAG project structure"

echo ""
echo "✓ Git repository initialized and initial commit created"
echo ""
echo "Next steps:"
echo "1. Add your remote repository:"
echo "   git remote add origin <your-repo-url>"
echo ""
echo "2. Push to remote:"
echo "   git branch -M main"
echo "   git push -u origin main"

