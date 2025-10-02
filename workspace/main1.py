'''
main.py

The main entry point for the AGI Mind Loop application.

This script is responsible for:
- Loading the application configuration.
- Initializing all core modules (LLMs, memory, cognitive functions).
- Instantiating the main MindLoop.
- Injecting the initialized modules into the loop (Dependency Injection).
- Starting and managing the lifecycle of the agent's mind loop.
'''

import yaml
import logging
from pathlib import Path

from core_loop import MindLoop, LoopConfig
from llm.manager import LLMManager
from cognition.goal_manager import GoalManager
from cognition.planner import Planner
from cognition.self_critic import SelfCritic
from action.decider import Decider
from action.experimenter import Experimenter
# Other modules like Interface, Memory, etc., will be imported here

# --- Configuration --- #
CONFIG_PATH = Path('config/config.yaml')

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_config(path: Path) -> dict:
    '''Loads the main YAML configuration file.'''
    logging.info(f"Loading configuration from {path}...")
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    logging.info("Configuration loaded successfully.")
    return config

def main():
    '''
    The main function to bootstrap and run the agent.
    '''
    setup_logging()
    config = load_config(CONFIG_PATH)

    try:
        # 1. Initialize Core Services
        logging.info("Initializing core services...")
        llm_manager = LLMManager(config.get('llm', {}))

        # Initialize other modules, passing LLM engines as needed
        goal_manager = GoalManager()
        planner = Planner()
        self_critic = SelfCritic(llm_client=llm_manager.get_engine('neutral_a'))
        decider = Decider(
            llm_client_a=llm_manager.get_engine('neutral_a'),
            llm_client_b=llm_manager.get_engine('mooded_b')
        )
        experimenter = Experimenter()
        # ... initialize memory, interface, etc. here

        # 2. Configure and Instantiate the MindLoop
        logging.info("Configuring and instantiating the MindLoop...")
        loop_config = LoopConfig(**config.get('runtime', {}))
        mind_loop = MindLoop()

        # 3. Inject Dependencies into the MindLoop
        mind_loop.init(
            config=loop_config,
            goals=goal_manager,
            planner=planner,
            critic=self_critic,
            # ... inject other dependencies here
        )

        # 4. Start the Loop
        logging.info("Starting the agent's MindLoop.")
        # This will eventually be a long-running process, e.g., mind_loop.start()
        # For now, we just run the scaffold's demo method.
        mind_loop.run_once()

    except (FileNotFoundError, KeyError, RuntimeError) as e:
        logging.critical(f"An error occurred during startup: {e}", exc_info=True)
        print("\nFATAL ERROR: Could not start the agent. See logs for details.")
        return

if __name__ == '__main__':
    main()

