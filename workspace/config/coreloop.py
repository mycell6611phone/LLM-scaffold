'''
core_loop.py

This is the central orchestrator of the AGI Mind Loop. It drives the agent's
perceive-think-act cycle, integrating all other cognitive modules.
'''

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime

# Optional static type imports only
if TYPE_CHECKING:
    from cognition.planner import Planner, Plan
    from cognition.goal_manager import GoalManager, Goal
    from io.interface import Interface


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
    # This dataclass can be expanded with more runtime options
    max_cycles: int = -1  # -1 for infinite loop


class MindLoop:
    '''
    Orchestrator that wires together and executes the cognitive cycle.
    '''

    def __init__(self, config: LoopConfig, interface: Interface, **kwargs):
        '''
        Initializes the MindLoop with all necessary modules (dependency injection).
        '''
        self.cfg = config
        self.running = False
        
        # Core injected dependencies
        self.interface = interface
        self.memory = kwargs.get('memory')
        self.goals = kwargs.get('goals')
        self.planner = kwargs.get('planner')
        self.debate = kwargs.get('debate')
        self.critic = kwargs.get('critic')
        self.trainer = kwargs.get('trainer')
        self.experimenter = kwargs.get('experimenter')
        self.decider = kwargs.get('decider')

        logging.info("MindLoop initialized with all dependencies.")

    def start(self):
        '''
        Starts the main, potentially infinite, mind loop.
        '''
        self.running = True
        cycle_count = 0
        logging.info(f"MindLoop starting. Max cycles: {'infinite' if self.cfg.max_cycles == -1 else self.cfg.max_cycles}")

        while self.running and (self.cfg.max_cycles == -1 or cycle_count < self.cfg.max_cycles):
            cycle_count += 1
            logging.info(f"--- Starting Mind Loop Cycle {cycle_count} ---")
            
            try:
                # This is the full cognitive cycle execution
                # Each of these methods is a scaffold and will raise NotImplementedError
                # until we implement them one by one.

                # 1. PERCEIVE: Get input from the user/environment.
                user_event = self.interface.fetch_event()
                if user_event.type == 'ABORT':
                    self.stop()
                    continue

                # 2. RECALL: Retrieve relevant memories based on the input.
                # relevant_memories = self.memory.recall(user_event.payload['text'])

                # 3. PLAN: Select a goal and generate a plan.
                # active_goal = self.goals.select_goal(user_event)
                # current_plan = self.planner.generate_plan(active_goal, relevant_memories)

                # 4. CRITIQUE & DECIDE: Analyze the plan and choose an action.
                # critique = self.critic.critique_plan(current_plan)
                # next_action = self.decider.decide_action(current_plan, critique)

                # 5. EXECUTE: Perform the action.
                # result = self.experimenter.execute(next_action)

                # 6. EVALUATE & STORE: Reflect on the result and store learnings.
                # reflection = self.critic.critique_result(result)
                # self.memory.store(reflection)

                # Placeholder until the above are implemented
                self.interface.send_output(f"(Cycle {cycle_count}) Received: {user_event.payload.get('text')}. Logic not yet implemented.")

            except NotImplementedError as e:
                logging.error(f"Halting loop. Not yet implemented: {e}")
                self.interface.send_output(f"[ERROR] Halting loop. Functionality not yet implemented: {e}")
                self.stop()
            except KeyboardInterrupt:
                logging.warning("Keyboard interrupt detected. Shutting down gracefully.")
                self.stop()
            except Exception as e:
                logging.critical(f"An unexpected error occurred in the mind loop: {e}", exc_info=True)
                self.interface.send_output(f"[FATAL] An unexpected error occurred: {e}")
                self.stop()

        logging.info("MindLoop has stopped.")

    def stop(self):
        '''Stops the mind loop gracefully.'''
        if self.running:
            logging.info("MindLoop stopping...")
            self.running = False

