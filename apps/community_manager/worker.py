"""Natural manager execution and backend-event scheduling on the dedicated queue."""
import logging
import traceback
from apps.community_manager.celery_app import app
from apps.community_manager.protocol import rpc
from apps.community_manager.agent import execute
from apps.community_manager.model import PROMPT_VERSION


LOGGER = logging.getLogger(__name__)


def log_failure(run_id: str, error: Exception) -> None:
    """Log exception type and source location without messages, inputs or credentials.

    Args:
        run_id: Opaque backend-issued run identifier.
        error: Exception whose text and provider payload must remain private.
    """
    frames = traceback.extract_tb(error.__traceback__)
    location = next((f for f in reversed(frames) if '/apps/community_manager/' in f.filename), None)
    status = getattr(error, 'status_code', getattr(error, 'code', None))
    status = status if isinstance(status, int) and 100 <= status <= 599 else None
    LOGGER.error('manager_execution_failed run_id=%s class=%s source=%s line=%s http_status=%s',
                 run_id, type(error).__name__, location.name if location else 'unknown',
                 location.lineno if location else 0, status)


@app.task(name="community_manager.run", soft_time_limit=500, time_limit=540)
def run(run_id: str) -> None:
    """Claim once and execute without creating new identities after uncertain writes.

    Args:
        run_id: Opaque Micro run identifier.
    """
    started = rpc('begin', run_id=run_id)
    token = started.get('token')
    if not token:
        return
    def call(action: str, **values: object) -> dict:
        """Invoke a run-bound operation.

        Args:
            action: Registered operation.
            values: Structured parameters validated by Micro.
        """
        return rpc(action, run_id=run_id, token=token, **values)
    try:
        if started.get('trigger', {}).get('kind') == 'customer_assessment':
            raise ValueError('wrong_manager_queue')
        text, model = execute(started, call)
        output = {'text': text.encode('utf-8')[:2000].decode('utf-8', errors='ignore'),
                  'model_id': model, 'prompt_version': PROMPT_VERSION}
    except Exception as error:
        log_failure(run_id, error)
        output = {'text': '本次经理分析未完成，已完成的操作请以记录为准；现有客服仍可继续工作，没有向客户发送消息。',
                  'model_id': 'unknown', 'prompt_version': PROMPT_VERSION, 'error_code': 'manager_execution_failed'}
    rpc('result', run_id=run_id, token=token, output=output)


@app.task(name="community_manager.tick")
def tick() -> None:
    """Publish platform events and recover Micro events/replies independently."""
    try:
        rpc('platform_drain')
    except Exception:
        # One source failure must not block the other source or recovery.
        pass
    rpc('tick')


@app.task(name="community_customer.run")
def customer_run(run_id: str) -> None:
    """Claim one background assessment, never invoke manager tools or customer sends.

    Args:
        run_id: Opaque Micro assessment run identifier.
    """
    from apps.community_manager.customer_model import assess, VERSION
    started = rpc('begin', run_id=run_id)
    token = started.get('token')
    if not token:
        return
    if started.get('trigger', {}).get('kind') != 'customer_assessment':
        # Mark wrong routing failed without running either model.
        rpc('result', run_id=run_id, token=token, output={'text': '本次运行路由不匹配。', 'model_id': 'unknown', 'prompt_version': VERSION, 'error_code': 'wrong_customer_queue'})
        return
    try:
        output, model = assess(started['customer_assessment'])
        output.update(model_id=model, prompt_version=VERSION)
    except Exception:
        output = {'error_code': 'customer_assessment_failed'}
    rpc('result', run_id=run_id, token=token, output=output)
