"""Load the shared pure-data journey contract without importing the customer app."""
import importlib.util
from pathlib import Path

# Fixed repository/image path; never a model-provided module, URL or executable.
path = Path(__file__).resolve().parents[1] / "group_agent_api" / "app" / "wecom_journey.py"
spec = importlib.util.spec_from_file_location("_community_customer_journey_contract", path)
if spec is None or spec.loader is None:
    raise ImportError("customer_journey_contract_unavailable")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)
validate_context = contract.validate_context
validate_journey = contract.validate_journey
CONVERGENCE_PROMPT = contract.CONVERGENCE_PROMPT
