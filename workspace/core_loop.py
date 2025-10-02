# core_loop.py
"""
Central orchestrator of the AGI Mind Loop.
Drives the perceive-think-act cycle and integrates cognitive modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from cognition.planner import Planner, Plan  # noqa: F401
    from cognition.goal_manager import GoalManager, Goal  # noqa: F401
    from io.interface import Interface  # noqa: F401

logger = logging.getLogger(__name__)


class Phase(Enum):
    PERCEIVE = "PERCEIVE"
    RECALL = "RECALL"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    EVALUATE = "EVALUATE"
    STORE = "STORE"
    TRAIN = "TRAIN"


@dataclass
class LoopConfig:
    """Runtime configuration for the mind loop."""
    max_cycles: int = -1  # -1 for infinite loop


class MindLoop:
    """
    Orchestrator that wires together and executes the cognitive cycle.
    Dependencies are injected via the constructor.
    """

    def __init__(self, config: LoopConfig, interface: "Interface", **kwargs: Any) -> None:
        self.cfg: LoopConfig = config
        self.running: bool = False

        # Core injected dependencies
        self.interface: "Interface" = interface
        self.memory: Optional[Any] = kwargs.get("memory")
        self.goals: Optional[Any] = kwargs.get("goals")
        self.planner: Optional[Any] = kwargs.get("planner")
        self.debate: Optional[Any] = kwargs.get("debate")
        self.critic: Optional[Any] = kwargs.get("critic")
        self.trainer: Optional[Any] = kwargs.get("trainer")
        self.experimenter: Optional[Any] = kwargs.get("experimenter")
        self.decider: Optional[Any] = kwargs.get("decider")

        logger.info("MindLoop initialized with injected dependencies.")

    def start(self) -> None:
        """Start the main, potentially infinite, mind loop."""
        self.running = True
        cycle_count = 0
        logger.info(
            "MindLoop starting. Max cycles: %s",
            "infinite" if self.cfg.max_cycles == -1 else self.cfg.max_cycles,
        )

        while self.running and (self.cfg.max_cycles == -1 or cycle_count < self.cfg.max_cycles):
            cycle_count += 1
            logger.info("--- Starting Mind Loop Cycle %d ---", cycle_count)

            try:
                # 1) PERCEIVE
                phase = Phase.PERCEIVE
                user_event = self.interface.fetch_event()
                etype = getattr(user_event, "type", None)
                payload = getattr(user_event, "payload", {}) or {}

                if isinstance(etype, str) and etype.upper() in {"ABORT", "STOP", "EXIT"}:
                    self.stop()
                    continue

                # 2) RECALL
                # phase = Phase.RECALL
                # relevant_memories = self.memory.recall(payload.get("text", ""))

                # 3) PLAN
                # phase = Phase.PLAN
                # active_goal = self.goals.select_goal(user_event)
                # current_plan = self.planner.generate_plan(active_goal, relevant_memories)

                # 4) CRITIQUE & DECIDE
                # phase = Phase.EVALUATE
                # critique = self.critic.critique_plan(current_plan)
                # next_action = self.decider.decide_action(current_plan, critique)

                # 5) EXECUTE
                # phase = Phase.EXECUTE
                # result = self.experimenter.execute(next_action)

                # 6) EVALUATE & STORE
                # phase = Phase.EVALUATE
                # reflection = self.critic.critique_result(result)
                # phase = Phase.STORE
                # self.memory.store(reflection)

                # Placeholder output until the above are implemented
                ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
                received_text = payload.get("text") if isinstance(payload, dict) else None
                self.interface.send_output(
                    f"[{ts}] (Cycle {cycle_count}) Phase={phase.value} | Received: {received_text!r} | Logic not yet implemented."
                )

            except NotImplementedError as e:
                logger.error("Halting loop. Not yet implemented: %s", e)
                self.interface.send_output(f"[ERROR] Halting loop. Functionality not yet implemented: {e}")
                self.stop()
            except KeyboardInterrupt:
                logger.warning("Keyboard interrupt detected. Shutting down gracefully.")
                self.stop()
            except Exception as e:  # noqa: BLE001
                logger.critical("Unexpected error in the mind loop: %s", e, exc_info=True)
                self.interface.send_output(f"[FATAL] Unexpected error occurred: {e}")
                self.stop()

        logger.info("MindLoop has stopped.")

    def stop(self) -> None:
        """Stop the mind loop gracefully."""
        if self.running:
            logger.info("MindLoop stopping...")
            self.running = False

