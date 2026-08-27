"""Brain-as-SUT harness: production brain + contract-shaped fakes for ear/hand/mouth.

See ``harness.py`` for install points and ``test_group_agent_brain_sut_poc.py`` for the PoC.
"""

from tests.support.brain_sut.fakes import FakeEar, FakeHand, FakeMouth
from tests.support.brain_sut.harness import (
    BrainSutHarness,
    install_brain_sut,
    install_instrumented_real_model,
)

__all__ = [
    "BrainSutHarness",
    "FakeEar",
    "FakeHand",
    "FakeMouth",
    "install_brain_sut",
    "install_instrumented_real_model",
]
