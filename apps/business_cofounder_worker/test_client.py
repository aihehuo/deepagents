"""Simple test client to submit tasks to the Celery worker.

Minimal implementation to test worker threading.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Ensure we use the same broker URL as the worker
# This should match what the worker is using
if "CELERY_BROKER_URL" not in os.environ:
    os.environ["CELERY_BROKER_URL"] = "redis://localhost:6379/0"

from apps.business_cofounder_worker.tasks import process_chat


def test_worker():
    """Submit a test task to the worker and wait for result."""
    print("=" * 80)
    print("Testing Business Co-Founder Celery Worker")
    print("=" * 80)
    
    # Submit task
    user_id = "test_user"
    conversation_id = "test_conv"
    message = "Hello, can you help me with a business idea?"
    
    # Show broker URL being used
    broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    print(f"\nUsing broker: {broker_url}")
    print(f"\nSubmitting task:")
    print(f"  User ID: {user_id}")
    print(f"  Conversation ID: {conversation_id}")
    print(f"  Message: {message}")
    print()
    print("NOTE: Make sure the Celery worker is running!")
    print("  Run: ./apps/business_cofounder_worker/start_worker.sh")
    print()
    
    # Submit task asynchronously
    result = process_chat.delay(user_id, conversation_id, message)
    
    print(f"Task submitted: {result.id}")
    print("Waiting for result...")
    print()
    
    # Wait for result (with timeout)
    try:
        task_result = result.get(timeout=300)  # 5 minute timeout
        print("=" * 80)
        print("Task Result:")
        print("=" * 80)
        import json
        print(json.dumps(task_result, indent=2, ensure_ascii=False))
        print()
        
        if task_result.get("success"):
            print("✓ Task completed successfully!")
            print(f"  Reply: {task_result.get('reply', '')[:200]}...")
        else:
            print("✗ Task failed!")
            error = task_result.get("error", {})
            print(f"  Error: {error.get('type')}: {error.get('message')}")
            
    except Exception as e:
        print(f"✗ Error waiting for result: {e}")
        import traceback
        traceback.print_exc()


def test_threadpool():
    """Test ThreadPoolExecutor to verify if thread creation works."""
    print("=" * 80)
    print("Testing ThreadPoolExecutor")
    print("=" * 80)
    
    # Show broker URL being used
    broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    print(f"\nUsing broker: {broker_url}")
    print()
    print("NOTE: Make sure the Celery worker is running!")
    print("  Run: ./apps/business_cofounder_worker/start_worker.sh")
    print()
    
    from apps.business_cofounder_worker.tasks import test_threadpool
    
    # Submit test task
    result = test_threadpool.delay(num_threads=3, num_tasks=5)
    
    print(f"Task submitted: {result.id}")
    print("Waiting for result...")
    print()
    
    try:
        task_result = result.get(timeout=60)  # 1 minute timeout
        print("=" * 80)
        print("ThreadPoolExecutor Test Result:")
        print("=" * 80)
        import json
        print(json.dumps(task_result, indent=2, ensure_ascii=False))
        print()
        
        if task_result.get("success"):
            print("✓ ThreadPoolExecutor test passed!")
            summary = task_result.get("summary", {})
            print(f"  Successful tasks: {summary.get('successful', 0)}/{summary.get('total_tasks', 0)}")
        else:
            print("✗ ThreadPoolExecutor test failed!")
            error = task_result.get("error", {})
            print(f"  Error: {error.get('type')}: {error.get('message')}")
            if error.get("is_thread_error"):
                print("\n  ⚠️  This confirms that ThreadPoolExecutor cannot create new threads!")
                print("     This is likely the root cause of the agent execution failures.")
            
    except Exception as e:
        print(f"✗ Error waiting for result: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Celery worker")
    parser.add_argument(
        "--threadpool",
        action="store_true",
        help="Test ThreadPoolExecutor (to verify thread creation works)"
    )
    args = parser.parse_args()
    
    if args.threadpool:
        test_threadpool()
    else:
        test_worker()

