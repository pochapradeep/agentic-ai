#!/usr/bin/env python3
"""Test script for Deep RAG API endpoints."""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import requests
except ImportError:
    print("Error: requests library not found. Install it with: pip install requests")
    sys.exit(1)


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓{Colors.RESET} {message}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.RED}✗{Colors.RESET} {message}")


def print_info(message: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {message}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {message}")


def print_header(message: str):
    """Print header message."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{message}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")


def test_health_endpoint(base_url: str, timeout: int = 10) -> bool:
    """Test the health check endpoint."""
    print_header("Testing Health Endpoint")
    
    url = f"{base_url}/health"
    print_info(f"GET {url}")
    
    try:
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        elapsed = time.time() - start_time
        
        print_info(f"Response time: {elapsed:.2f}s")
        print_info(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print_success("Health check passed")
            print(f"  Status: {data.get('status')}")
            print(f"  Version: {data.get('version')}")
            
            system_info = data.get('system_info', {})
            if system_info:
                print("  System Info:")
                for key, value in system_info.items():
                    print(f"    {key}: {value}")
            
            return True
        else:
            print_error(f"Health check failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print_error(f"Health check timed out after {timeout}s")
        return False
    except requests.exceptions.ConnectionError:
        print_error("Could not connect to the API. Is the server running?")
        return False
    except Exception as e:
        print_error(f"Error testing health endpoint: {e}")
        return False


def test_sync_query(base_url: str, question: str, max_steps: Optional[int] = None, timeout: int = 300) -> bool:
    """Test the synchronous query endpoint."""
    print_header("Testing Synchronous Query Endpoint")
    
    url = f"{base_url}/api/v1/query"
    print_info(f"POST {url}")
    
    payload = {"question": question}
    if max_steps:
        payload["max_steps"] = max_steps
    
    print_info(f"Question: {question}")
    if max_steps:
        print_info(f"Max steps: {max_steps}")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=timeout)
        elapsed = time.time() - start_time
        
        print_info(f"Response time: {elapsed:.2f}s")
        print_info(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print_success("Query completed successfully")
            
            print(f"\n{Colors.BOLD}Answer:{Colors.RESET}")
            print(f"{data.get('answer', '')[:500]}...")
            
            print(f"\n{Colors.BOLD}Metadata:{Colors.RESET}")
            print(f"  Question: {data.get('question')}")
            print(f"  Steps taken: {data.get('steps_taken')}")
            print(f"  Processing time: {data.get('processing_time', 0):.2f}s")
            
            sources = data.get('sources')
            if sources:
                print(f"  Sources: {len(sources)}")
            else:
                print("  Sources: None")
            
            return True
        else:
            print_error(f"Query failed with status {response.status_code}")
            try:
                error_data = response.json()
                print(f"  Error: {error_data.get('error', 'Unknown error')}")
                if error_data.get('details'):
                    print(f"  Details: {error_data.get('details')}")
            except:
                print(f"  Response: {response.text[:500]}")
            return False
            
    except requests.exceptions.Timeout:
        print_error(f"Query timed out after {timeout}s")
        return False
    except requests.exceptions.ConnectionError:
        print_error("Could not connect to the API. Is the server running?")
        return False
    except Exception as e:
        print_error(f"Error testing query endpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_streaming_query(base_url: str, question: str, max_steps: Optional[int] = None, timeout: int = 300) -> bool:
    """Test the streaming query endpoint."""
    print_header("Testing Streaming Query Endpoint")
    
    url = f"{base_url}/api/v1/query/stream"
    print_info(f"POST {url}")
    
    payload = {"question": question}
    if max_steps:
        payload["max_steps"] = max_steps
    
    print_info(f"Question: {question}")
    if max_steps:
        print_info(f"Max steps: {max_steps}")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, stream=True, timeout=timeout)
        
        print_info(f"Status code: {response.status_code}")
        
        if response.status_code != 200:
            print_error(f"Streaming query failed with status {response.status_code}")
            try:
                error_data = response.json()
                print(f"  Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"  Response: {response.text[:500]}")
            return False
        
        print_success("Streaming started")
        print(f"\n{Colors.BOLD}Streaming Events:{Colors.RESET}\n")
        
        chunk_count = 0
        answer_received = False
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # Remove 'data: ' prefix
                    
                    if data_str == '[DONE]':
                        print(f"\n{Colors.GREEN}✓ Stream completed{Colors.RESET}")
                        break
                    
                    try:
                        chunk = json.loads(data_str)
                        chunk_count += 1
                        chunk_type = chunk.get('type', 'unknown')
                        content = chunk.get('content', '')
                        step = chunk.get('step')
                        
                        # Color code by type
                        if chunk_type == 'plan':
                            color = Colors.CYAN
                            icon = '📋'
                        elif chunk_type == 'retrieval':
                            color = Colors.BLUE
                            icon = '🔍'
                        elif chunk_type == 'reflection':
                            color = Colors.YELLOW
                            icon = '💭'
                        elif chunk_type == 'answer':
                            color = Colors.GREEN
                            icon = '✅'
                            answer_received = True
                        elif chunk_type == 'complete':
                            color = Colors.GREEN
                            icon = '✓'
                        elif chunk_type == 'error':
                            color = Colors.RED
                            icon = '✗'
                        else:
                            color = Colors.RESET
                            icon = '•'
                        
                        step_str = f" [Step {step}]" if step else ""
                        print(f"{color}{icon} [{chunk_type.upper()}]{step_str}{Colors.RESET}")
                        
                        # Print content preview
                        if content:
                            preview = content[:200] + "..." if len(content) > 200 else content
                            print(f"   {preview}")
                        
                        # Print metadata if available
                        metadata = chunk.get('metadata')
                        if metadata:
                            meta_str = ", ".join([f"{k}: {v}" for k, v in metadata.items() if v])
                            if meta_str:
                                print(f"   {Colors.YELLOW}Metadata: {meta_str}{Colors.RESET}")
                        
                        print()
                        
                    except json.JSONDecodeError:
                        print_warning(f"Could not parse chunk: {data_str[:100]}")
        
        elapsed = time.time() - start_time
        print_info(f"Total chunks received: {chunk_count}")
        print_info(f"Total time: {elapsed:.2f}s")
        
        if answer_received:
            print_success("Answer received in stream")
            return True
        else:
            print_warning("No answer received in stream")
            return False
            
    except requests.exceptions.Timeout:
        print_error(f"Streaming query timed out after {timeout}s")
        return False
    except requests.exceptions.ConnectionError:
        print_error("Could not connect to the API. Is the server running?")
        return False
    except KeyboardInterrupt:
        print_warning("\nStreaming interrupted by user")
        return False
    except Exception as e:
        print_error(f"Error testing streaming endpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_root_endpoint(base_url: str) -> bool:
    """Test the root endpoint."""
    print_header("Testing Root Endpoint")
    
    url = base_url.rstrip('/')
    print_info(f"GET {url}")
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print_success("Root endpoint accessible")
            print(f"  Name: {data.get('name')}")
            print(f"  Version: {data.get('version')}")
            print(f"  API URL: {data.get('api_url')}")
            return True
        else:
            print_error(f"Root endpoint failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error testing root endpoint: {e}")
        return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="Test Deep RAG API endpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all endpoints with default question
  python scripts/test_api.py

  # Test with custom question
  python scripts/test_api.py --question "What are green hydrogen cost benchmarks?"

  # Test only health endpoint
  python scripts/test_api.py --health-only

  # Test only streaming
  python scripts/test_api.py --stream-only

  # Test with custom base URL
  python scripts/test_api.py --base-url http://localhost:8000
        """
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    
    parser.add_argument(
        "--question",
        type=str,
        default="What are the key cost benchmarks for green hydrogen production in India?",
        help="Question to test with"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum reasoning steps"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Test only the health endpoint"
    )
    
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Test only the synchronous query endpoint"
    )
    
    parser.add_argument(
        "--stream-only",
        action="store_true",
        help="Test only the streaming query endpoint"
    )
    
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    
    args = parser.parse_args()
    
    # Disable colors if requested
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith('_'):
                setattr(Colors, attr, '')
    
    base_url = args.base_url.rstrip('/')
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}Deep RAG API Test Suite{Colors.RESET}")
    print(f"{Colors.CYAN}Base URL: {base_url}{Colors.RESET}\n")
    
    results = {}
    
    # Test root endpoint (always test)
    if not (args.health_only or args.sync_only or args.stream_only):
        results['root'] = test_root_endpoint(base_url)
    
    # Test health endpoint
    if args.health_only or not (args.sync_only or args.stream_only):
        results['health'] = test_health_endpoint(base_url, timeout=args.timeout)
    
    # Test synchronous query
    if args.sync_only or not (args.health_only or args.stream_only):
        results['sync'] = test_sync_query(
            base_url, 
            args.question, 
            max_steps=args.max_steps,
            timeout=args.timeout
        )
    
    # Test streaming query
    if args.stream_only or not (args.health_only or args.sync_only):
        results['stream'] = test_streaming_query(
            base_url, 
            args.question, 
            max_steps=args.max_steps,
            timeout=args.timeout
        )
    
    # Print summary
    print_header("Test Summary")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test_name, result in results.items():
        status = f"{Colors.GREEN}PASSED{Colors.RESET}" if result else f"{Colors.RED}FAILED{Colors.RESET}"
        print(f"  {test_name.upper()}: {status}")
    
    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.RESET}\n")
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()

