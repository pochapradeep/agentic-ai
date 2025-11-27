#!/bin/bash
# Example test commands for the Deep RAG API

echo "Deep RAG API Test Examples"
echo "=========================="
echo ""

# Make sure the script is executable
chmod +x scripts/test_api.py

echo "1. Test all endpoints with default question:"
echo "   python scripts/test_api.py"
echo ""

echo "2. Test with custom question:"
echo "   python scripts/test_api.py --question 'What are green hydrogen cost benchmarks?'"
echo ""

echo "3. Test only health endpoint:"
echo "   python scripts/test_api.py --health-only"
echo ""

echo "4. Test only synchronous query:"
echo "   python scripts/test_api.py --sync-only"
echo ""

echo "5. Test only streaming query:"
echo "   python scripts/test_api.py --stream-only"
echo ""

echo "6. Test with custom base URL (Azure deployment):"
echo "   python scripts/test_api.py --base-url https://your-app.azurecontainerapps.io"
echo ""

echo "7. Test with max steps limit:"
echo "   python scripts/test_api.py --max-steps 3"
echo ""

echo "8. Test with custom timeout:"
echo "   python scripts/test_api.py --timeout 600"
echo ""

echo "9. Test without colored output:"
echo "   python scripts/test_api.py --no-color"
echo ""

echo "Running basic health check test..."
python scripts/test_api.py --health-only

