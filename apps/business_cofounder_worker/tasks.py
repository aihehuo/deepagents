"""Celery tasks for Business Co-Founder Worker.

Minimal task implementation to test threading in production.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from apps.business_cofounder_worker.celery_app import app

_logger = logging.getLogger(__name__)


@app.task(bind=True, name="business_cofounder.process_chat")
def process_chat(self, user_id: str, conversation_id: str, message: str) -> dict[str, Any]:
    """Process a chat message using the Business Co-Founder agent.
    
    This is a minimal implementation to test that:
    1. Celery worker can run in production
    2. Threading works in the worker process
    3. Agent execution works without thread creation errors
    
    Args:
        user_id: User identifier
        conversation_id: Conversation identifier
        message: User message
        
    Returns:
        Dictionary with reply and status
    """
    thread_id = f"bc::{user_id}::{conversation_id}"
    _logger.info(
        "[process_chat] Starting task user_id=%s conversation_id=%s thread_id=%s message_len=%s",
        user_id,
        conversation_id,
        thread_id,
        len(message),
    )
    
    # Log current thread info
    current_thread = threading.current_thread()
    _logger.info(
        "[process_chat] Running in thread: name=%s ident=%s daemon=%s",
        current_thread.name,
        current_thread.ident,
        current_thread.daemon,
    )
    
    try:
        # Import agent factory (lazy import to avoid issues at module level)
        from apps.business_cofounder_api.agent_factory import create_business_cofounder_agent
        from langchain_core.messages import HumanMessage
        
        # Get or create agent (in production, this would be a singleton)
        # For minimal test, we'll create it each time (not optimal, but simple)
        _logger.info("[process_chat] Creating agent for thread_id=%s", thread_id)
        agent, checkpoints_path = create_business_cofounder_agent(agent_id="business_cofounder_agent")
        
        # Prepare inputs
        inputs = {"messages": [HumanMessage(content=message)]}
        config = {
            "configurable": {"thread_id": thread_id},
            "metadata": {"user_id": user_id, "conversation_id": conversation_id},
        }
        
        # Invoke agent (sync - this is where threading issues would occur)
        _logger.info("[process_chat] Calling agent.invoke for thread_id=%s", thread_id)
        result = agent.invoke(inputs, config)
        _logger.info(
            "[process_chat] agent.invoke completed for thread_id=%s, result_keys=%s",
            thread_id,
            list(result.keys()) if isinstance(result, dict) else type(result).__name__,
        )
        
        # Extract reply
        messages = result.get("messages", [])
        ai_messages = [m for m in messages if getattr(m, "type", None) == "ai"]
        reply = str(ai_messages[-1].content) if ai_messages else ""
        
        _logger.info(
            "[process_chat] Task completed successfully for thread_id=%s reply_len=%s",
            thread_id,
            len(reply),
        )
        
        return {
            "success": True,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "thread_id": thread_id,
            "reply": reply,
        }
        
    except Exception as e:  # noqa: BLE001
        import traceback
        
        _logger.exception(
            "[process_chat] Task failed for thread_id=%s: %s: %s",
            thread_id,
            type(e).__name__,
            str(e),
        )
        
        return {
            "success": False,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "thread_id": thread_id,
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
        }


@app.task(bind=True, name="business_cofounder.test_threadpool")
def test_threadpool(self, num_threads: int = 3, num_tasks: int = 5) -> dict[str, Any]:
    """Test ThreadPoolExecutor to verify if thread creation is allowed.
    
    This is a minimal test to isolate whether ThreadPoolExecutor is the cause
    of "can't start new thread" errors in production.
    
    Args:
        num_threads: Number of threads in the pool
        num_tasks: Number of tasks to submit to the pool
        
    Returns:
        Dictionary with test results
    """
    _logger.info(
        "[test_threadpool] Starting test num_threads=%s num_tasks=%s",
        num_threads,
        num_tasks,
    )
    
    # Log current thread info
    current_thread = threading.current_thread()
    _logger.info(
        "[test_threadpool] Running in thread: name=%s ident=%s daemon=%s",
        current_thread.name,
        current_thread.ident,
        current_thread.daemon,
    )
    
    try:
        # Simple function to execute in thread pool
        def dummy_task(task_id: int) -> dict[str, Any]:
            thread_name = threading.current_thread().name
            thread_id = threading.current_thread().ident
            _logger.info(
                "[test_threadpool] Task %s running in thread: name=%s ident=%s",
                task_id,
                thread_name,
                thread_id,
            )
            # Simulate some work
            import time
            time.sleep(0.1)
            return {
                "task_id": task_id,
                "thread_name": thread_name,
                "thread_id": thread_id,
                "status": "completed",
            }
        
        _logger.info("[test_threadpool] Creating ThreadPoolExecutor with %s threads", num_threads)
        
        # Try to create ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            _logger.info("[test_threadpool] ThreadPoolExecutor created successfully")
            
            # Submit tasks
            _logger.info("[test_threadpool] Submitting %s tasks", num_tasks)
            futures = [executor.submit(dummy_task, i) for i in range(num_tasks)]
            
            # Collect results
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    _logger.info(
                        "[test_threadpool] Task %s completed: %s",
                        result["task_id"],
                        result["thread_name"],
                    )
                except Exception as e:
                    _logger.exception("[test_threadpool] Task failed: %s", e)
                    results.append({"error": str(e)})
            
            _logger.info(
                "[test_threadpool] All tasks completed. Results: %s",
                len(results),
            )
            
            return {
                "success": True,
                "num_threads": num_threads,
                "num_tasks": num_tasks,
                "results": results,
                "summary": {
                    "total_tasks": len(results),
                    "successful": len([r for r in results if "error" not in r]),
                    "failed": len([r for r in results if "error" in r]),
                },
            }
            
    except RuntimeError as e:
        if "can't start new thread" in str(e).lower():
            _logger.exception(
                "[test_threadpool] ThreadPoolExecutor failed with 'can't start new thread' error"
            )
            return {
                "success": False,
                "error": {
                    "type": "RuntimeError",
                    "message": str(e),
                    "is_thread_error": True,
                },
            }
        else:
            raise
    except Exception as e:  # noqa: BLE001
        import traceback
        
        _logger.exception(
            "[test_threadpool] Test failed: %s: %s",
            type(e).__name__,
            str(e),
        )
        
        return {
            "success": False,
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
        }

