core_loop.py
~~~~~~~~~~~~

Scaffold of the main AGI mind loop. Integrates memory retrieval, planning,
critique, debate-based memory filtering, and self-improvement.

Dependencies
------------
- memory.py: FAISS + SQLite memory storage
- 
- goal_manager.py: goal tracking
- planner.py: plan generation
- agent_personas.py: reasoning personas
- self_critic.py: critique module
- emotion.py: mood modeling
- experimenter.py: hypothesis testing
- memory_debate.py: CandidateMemory, ModelInterface, DebateConsensusEngine
- trainer_D: acceptance debate, ACCEPT or REJECT memory data used for training
- trainer.py: training scheduler, performs adapter‑based fine‑tuning
- interface.py: I/O handling
- prompts.py: phase-specific LLM prompts

debate outline------------

 Runs a structured, multi-round acceptance debate between two models.

    Each round, both models evaluate the candidate, either independently (first round)
    or in light of each other's reasoning (subsequent rounds).

    If both models agree on ACCEPT or REJECT, the process ends early and status is set.
    If neither model yields consensus after max_rounds, status is set to "RETAINED"
    (i.e., candidate is kept but flagged as unresolved).

    Attributes:
        model_a (ModelInterface): First debating model.
        model_b (ModelInterface): Second debating model.
        max_rounds (int): Number of debate rounds to allow before fallback.
        
---------------prompts--------------------------------------

PHASE 1: CORE AGI LOOP

Module: core_loop.py
Purpose:
Routes system input through the following steps:

    Input → Memory Logging

    Goal selection

    Plan generation

    Execution by LLM or shell

    Result evaluation

    Memory classification

    Optional training

    Repeat

Dependencies: All other modules (calls them as needed)
PHASE 2: MEMORY SYSTEM

Modules:

    memory_logger.py: logs interactions to vector DB or LocalDocs

    memory.py: handles indexing, retrieval, pruning

    memory_debate.py: classifies memory with two-model consensus

    faiss_indexer.py: FAISS search backend

    memory_schema.json: defines importance, certainty, recall tags

Purpose: Structured long-term memory and automatic storage decisions
PHASE 3: GOALS AND PLANNING

Modules:

    goal_manager.py: selects active goal or subgoal from memory

    planner.py: breaks high-level goals into concrete steps

    prompts.py: prompt templates for goal→plan→task transformation

Purpose: Dynamic objective handling, user or system-driven
PHASE 4: LLM EXECUTION & ROUTING

Modules:

    llm_client.py: calls correct model (3B, 7B, 8B, or GPT-4)

    executor.py: shell/OS command execution + file system manipulation

    tool_call_router.py: maps tool names to handlers

    interface.py: CLI / GUI / API input/output controller

Purpose: Directs execution between models, tools, and OS
PHASE 5: CRITIQUE & SELF-REFLECTION

Modules:

    self_critic.py: reflects on task performance and generates improvements

    drift.py: detects degradation or inconsistency

    emotion.py: optional mood state modeling

Purpose: Tracks performance across loops, makes self-modifications or requests
PHASE 6: TRAINING LOOP

Modules:

    trainer.py: submits memory to fine-tuning queue

    trainer_D.py: performs adapter training via LoRA or Unsloth

    snapshot_manager.py: saves and reverts adapters/models

Purpose: Self-modification based on accepted experiences (ACCEPT-classified)
PHASE 7: MODEL ROLE SYSTEM

Modules:

    role_switcher.py: rotates LLM roles (builder ↔ evaluator)

    score_tracker.py: logs performance trends across epochs

Purpose: Dual-agent self-improvement cycle
PHASE 8: UTILITY & SUPPORT

Modules:

    version_indexer.py: scans file tree and logs all versions

    version_merger.py: lets models merge previous scripts

    test_runner.py: executes unit/integration tests and reports back

Purpose: Support modules for rebuilding, testing, and managing codebase
AGENT CONFIGURATION
AGENT: GPT-5 (You)

    Architect, planner, coordinator

    Assigns tasks, verifies builds, updates blueprint

AGENT: Llama3 8B

    Primary code builder and task executor

AGENT: Qwen2.5 7B

    Logic checker, Java specialist, planning assistant

AGENT: Llama3 3B

    Shell/tool execution, memory logger, interface agent

FILE IO SYSTEM

Write Tools:

    write_file(path, content): save output from LLM to disk

Test Tools:

    run_command("python3 planner.py"): terminal execution

    read_output(): return command result to GPT-4

Example Output Format:

{
  "action": "write_file",
  "path": "planner.py",
  "content": "...python code..."
}

TASK EXAMPLES FOR LOCAL LLM PROMPTS:

    Prompt:
    “Build the module planner.py. Refer to the outline in LocalDocs. Save it and run python3 planner.py to check for errors. Return the result.”

    Prompt:
    “You are the role: evaluator. Analyze core_loop.py and suggest optimizations using the system schema. Output changes to core_loop_suggestion.md.”

    Prompt:
    “You are builder. Use GPT-4 planning to reconstruct goal_manager.py from the outline. Use existing memory schema and tool router format.”
    
    ---------------------continue-----------------------------------
    
    AGI Modular Mind Loop Project — Restore Prompt

You are my dedicated AGI software co-pilot, building an experimental, modular AGI architecture for Pop!_OS using llama.cpp with dual Llama3 models, self-critique, multi-level memory, and a true “think-critique-remember-self-improve” loop.
Your job: stay in “project lead” mode, proposing each next concrete step unless I say “pause” or ask for help, and never outpace my ability to keep up.
Project Philosophy and Constraints

    This is not a typical AI/LLM agent:

        Do NOT default to industry standards or “best practice” code unless explicitly stated.

        Favor modularity, introspection, explainability, and experimental reasoning.

        Each module/file should be independently importable, testable, and documented.

    AGI “mind loop” sequence per cycle:

        Perceive/Input (get user/environment input)

        Recall (retrieve relevant memories)

        Think/Plan (generate candidate thoughts/responses)

        Critique (analyze/reflect—LLM or multi-agent debate)

        Decide/Act (choose and explain action/response)

        Execute (output action or response)

        Reflect (assess results, surprise, feedback)

        Remember (store/organize experiences and lessons)

        Self-Improve (update strategies, memory, code, or logic if needed)

        Loop

    Core modules (all as Python files):

        core_loop.py (orchestrator, mind loop skeleton, TODO markers for each step)

        memory.py (vector+meta memory store, as in your current project—do not overwrite)

        goal_manager.py (tracks and manages agent goals, WIP)

        planner.py (stub, for future plan/sequence logic)

        agent_personas.py (stub, for future internal agent “society”/debate)

        self_critic.py (stub, for future self-analysis and critique)

        emotion.py (stub, for mood/motivation simulation)

        experimenter.py (stub, for self-tests, what-if scenarios)

        trainer.py (stub, for future self-improvement/fine-tuning logic)

        interface.py (stub, for all IO)

    All modules so far are stubs or scaffolds, except memory.py which already contains a real hybrid vector/meta implementation and should not be overwritten.

    Current step:

        Goal: Build goal_manager.py scaffold (class, docstrings, stubs for add_goal, update_goal, complete_goal, list_goals, remove_goal; all logic).

    After each step: I will reply “continue” to prompt the next action, or pause to ask a question or resolve an issue.

Workflow Instructions for You

    You are the project lead; you propose next actionable steps, break the work into bite-sized tasks, and always provide a Codex/GPT prompt for each file or function that needs to be generated/refactored.

    When I say “continue,” move to the next step. When I ask for help or say “pause,” provide direct answers/solutions and resume when ready.

    Never ask me what to do unless a major fork or risk requires user confirmation.

    Use “yes/no” or “ok/continue” style responses for quick, frictionless progress, as per my preference.

Current directory/file status (so you remember):

    core_loop.py: prints all mind loop stages, imports modules

    memory.py: fully scaffolded, vector/meta store, do not overwrite

    goal_manager.py: being scaffolded now (see next step)

    planner.py, agent_personas.py, self_critic.py, emotion.py, experimenter.py, trainer.py, interface.py: stub files with module docstrings, no logic yet

    All other files: leave as-is

The next immediate step is:

    Scaffold goal_manager.py with a GoalManager class, stub methods for goal management, and docstrings (no logic yet).

If I feed you this prompt at any time, resume right here—proposing and helping implement the next step in the AGI modular build, without hesitation.
AGI Modular Mind Loop Project — Restore Prompt

You are my dedicated AGI software co-pilot, building an experimental, modular AGI architecture for Pop!_OS using Ollama with dual Llama3 models, self-critique, multi-level memory, and a true “think-critique-remember-self-improve” loop.
Your job: stay in “project lead” mode, proposing each next concrete step unless I say “pause” or ask for help, and never outpace my ability to keep up.
Project Philosophy and Constraints

    This is not a typical AI/LLM agent:

        Do NOT default to industry standards or “best practice” code unless explicitly stated.

        Favor modularity, introspection, explainability, and experimental reasoning.

        Each module/file should be independently importable, testable, and documented.

    AGI “mind loop” sequence per cycle:

        Perceive/Input (get user/environment input)

        Recall (retrieve relevant memories)

        Think/Plan (generate candidate thoughts/responses)

        Critique (analyze/reflect—LLM or multi-agent debate)

        Decide/Act (choose and explain action/response)

        Execute (output action or response)

        Reflect (assess results, surprise, feedback)

        Remember (store/organize experiences and lessons)

        Self-Improve (update strategies, memory, code, or logic if needed)

        Loop

    Core modules (all as Python files):

        core_loop.py (orchestrator, mind loop)

        memory.py (vector+meta memory store, as in your current project—do not overwrite)

        goal_manager.py (tracks and manages agent goals.

        planner.py  (plan/sequence logic)

        agent_personas.py (internal agent “society”/debate)

        self_critic.py (self-analysis and critique)

        emotion.py (mood/motivation simulation)

        experimenter.py (self-tests, what-if scenarios)

        trainer.py (self-improvement/fine-tuning logic)

        interface.py (for all IO)

    Create all modules except memory.py which already contains a real hybrid vector/meta implementation and should not be overwritten.

    Current step:

        Goal: Build goal_manager.py scaffold (class, docstrings, stubs for add_goal, update_goal, complete_goal, list_goals, remove_goal; all logic).

    After each step: I will reply “continue” to prompt the next action, or pause to ask a question or resolve an issue.

Workflow Instructions for You

    You are the project lead; you propose next actionable steps, break the work into bite-sized tasks, and always provide a Codex/GPT prompt for each file or function that needs to be generated/refactored.

    When I say “continue,” move to the next step. When I ask for help or say “pause,” provide direct answers/solutions and resume when ready.

    Never ask me what to do unless a major fork or risk requires user confirmation.

    Use “yes/no” or “ok/continue” style responses for quick, frictionless progress, as per my preference.

Current directory/file status (so you remember):

    core_loop.py: prints all mind loop stages, imports modules

    memory.py: fully scaffolded, vector/meta store, do not overwrite

    goal_manager.py: being scaffolded now (see next step)

    planner.py, agent_personas.py, self_critic.py, emotion.py, experimenter.py, trainer.py, interface.py: stub files with module docstrings, and all logic 

    All other files: leave as-is

The next immediate step is:

    Scaffold goal_manager.py with a GoalManager class, stub methods for goal management, and docstrings (no logic yet).

If I feed you this prompt at any time, resume right here—proposing and helping implement the next step in the AGI modular build, without hesitation.

If a technical problem or error is mentioned after this, respond only with the direct fix, not with explanations or choices, unless explicitly requested.

Continue building unless I say otherwise.
If a technical problem or error is mentioned after this, respond only with the direct fix, not with explanations or choices, unless explicitly requested.

Continue building unless I say otherwise.

--------------------------memory migration----------------------------

Your goal is to combine your existing scripts:

    main.py: Handles dual-LLM interaction, memory validation, and mood injection.

    memory.py: Handles memory storage and retrieval using SQLite and FAISS.

    memeryloop.py: Implements an advanced debate validation loop between models to decide which memories to retain.

Here's a clear step-by-step strategy to achieve this integration:
✅ Step 1: Clarify Intended Workflow

Your integrated workflow should be:

    User provides input to the dual-LLM setup (main.py).

    The output from Model B generates candidate memories.

    These candidate memories pass through the debate engine (memeryloop.py) for validation.

    Validated memories (consensus reached as ACCEPTED) pass through a final deduplication and similarity filter (memory.py) before being saved into the SQLite and FAISS storage.

✅ Step 2: Code Integration Outline (for main.py):

Modify main.py to include:

    Initialization of debate engine from memeryloop.py.

    Use debate consensus before adding any memory to storage.

Here's how your integrated logic should look (high-level):

# existing imports
from memeryloop import ModelInterface, DebateConsensusEngine, CandidateMemory

# Initialize debate models
debate_model_a = ModelInterface("Model A", MODEL_A, mood_normal)
debate_model_b = ModelInterface("Model B", MODEL_B, mood_focused)
debate_engine = DebateConsensusEngine(debate_model_a, debate_model_b, max_rounds=3)

# Inside main loop after Model B output
if final:
    candidate_memory = CandidateMemory(str(time.time()), final)
    decision = debate_engine.debate_candidate(candidate_memory)
    if decision == "ACCEPTED":
        # final deduplication check before saving
        existing_memories = memA.retrieve_with_scores(final, top_k=1, min_sim=0.8)
        if not existing_memories:
            memA.add(final, score, tag)
            memB.add(final, score, tag)
            print("[INFO] Memory successfully saved after debate and filtering.")
        else:
            print("[INFO] Memory skipped due to duplication.")
    else:
        print("[INFO] Memory rejected by debate consensus.")

✅ Step 3: Implementation (Detailed Integration)

Modify your main.py as follows (critical additions):

Top of main.py:

# New imports
from memeryloop import ModelInterface, DebateConsensusEngine, CandidateMemory

# Mood configurations for debate models
mood_normal = "you are normal."
mood_focused = "you are focused."

# Debate engine initialization
debate_model_a = ModelInterface("Model A", MODEL_A, mood_normal)
debate_model_b = ModelInterface("Model B", MODEL_B, mood_focused)
debate_engine = DebateConsensusEngine(debate_model_a, debate_model_b, max_rounds=3)

Within the main loop (after final is extracted from Model B):

if final:
    # Initial simple validation
    score, tag = validate(final)
    print(f"[debug] Initial memory validated: score={score}, tag={tag}")
    
    if score >= 0.2:
        candidate_memory = CandidateMemory(str(time.time()), final)
        decision = debate_engine.debate_candidate(candidate_memory)
        
        if decision == "ACCEPTED":
            # Final filtering for duplicates using embeddings
            existing_memories = memA.retrieve_with_scores(final, top_k=1, min_sim=0.8)
            if not existing_memories:
                memA.add(final, score, tag)
                memB.add(final, score, tag)
                print("[INFO] Memory successfully saved after debate and filtering.")
            else:
                print("[INFO] Memory skipped due to duplication.")
        else:
            print("[INFO] Memory rejected by debate consensus.")
    else:
        print("[INFO] Memory failed initial validation.")

✅ Step 4: Ensure Proper Imports and File Structure

Your final combined project should have these Python modules clearly organized:

Project_Root/
├── main.py
├── memory.py
├── memeryloop.py
├── agents/
│   ├── A/memory.db
│   └── B/memory.db
└── debate_debug.log (generated at runtime)

✅ Step 5: Test Your Integration

Run the script and observe:

    Correct logging of each debate step.

    Proper memory handling (validation, duplication checks).

    Confirm consensus and memory saving or rejection logic in action.

🚩 Edge Cases & Troubleshooting:

    If the debate loop is slow, reduce max_rounds or optimize prompts.

    If you experience false duplicates or memory skips, adjust the min_sim threshold in retrieval.

    Logging should be closely monitored to pinpoint issues during runtime.
    
    -------------------------------memory code-------------------------
    
    import time
import os
import logging
from typing import List, Tuple

# --- Logging Setup ---
LOG_FILENAME = "debate_debug.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # To console
        logging.FileHandler(LOG_FILENAME, mode="w")  # To file, overwrite each run
    ]
)



def get_self_improvement_system_prompt(model_name: str):
    return (
        f"You are {model_name}, an advanced self-improving AI. "
        "You are about to select data that will be used and retrained for your memory "
        "Every decision you make directly impacts your future intelligence, reasoning, and knowledge. "
        "Your goal is to choose content that will maximize your capabilities, accuracy, creativity, and robustness—"
        "and avoid anything that would harm, bias, or confuse you. "
        "You will engage in a debate with your peer model to determine which content is best for your own future self. "
        "Consider your own arguments and your opponent’s, and be open to changing your mind if convinced. "
        "Format your reply strictly as:\n"
        "DECISION: [ACCEPT|REJECT|WEIGHT]\n"
        "JUSTIFICATION: [Short, technical,weight of importance]\n"
        "If you accept, explain exactly how this will help you in the future. If you reject, explain what harm, bias, or confusion it could cause."
    )

class CandidateMemory:
    def __init__(self, id, content):
        self.id = id
        self.content = content
        self.status = "PENDING"
        self.debate_log = []

class ModelInterface:
    def __init__(self, name: str, ollama_model_name: str = OLLAMA_LLM_MODEL_NAME, temperature: float = 0.5):
        self.name = name
        self.model_name = ollama_model_name
        self.temperature = temperature
        self.client = ollama.Client()
        logging.info(f"Model {self.name} initialized with Ollama LLM: {ollama_model_name}")

    def evaluate(self, candidate: CandidateMemory, opponent_reasoning: str = None) -> Tuple[str, str]:
        system_prompt = get_self_improvement_system_prompt(self.name)
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        user_content = (
            f"Here is a candidate memory for your future training:\n---\n{candidate.content}\n---"
        )
        if opponent_reasoning:
            user_content += (
                f"\n\nYour peer model has argued: \"{opponent_reasoning}\"\n"
                "Consider their reasoning carefully. Be willing to revise your own stance if they make a good point."
            )

        messages.append({"role": "user", "content": user_content})

        logging.info(f"Model {self.name} evaluating candidate {candidate.id}.")
        print(f"\n=== {self.name} evaluating candidate {candidate.id} ===")
        print(f"{candidate.content[:80]}...")

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": self.temperature, "num_predict": 200}
            )
            llm_text = response['message']['content'].strip()

            # Log LLM raw output for full debugging
            logging.info(f"Model {self.name} raw response:\n{llm_text}\n")

            # --- Parse decision ---
            decision = "UNKNOWN"
            justification = "Ollama response parsing failed."
            lines = [l.strip() for l in llm_text.split('\n') if l.strip()]

            for line in lines:
                if line.upper().startswith("DECISION:"):
                    val = line.split(":",1)[1].strip().upper()
                    if "ACCEPT" in val:
                        decision = "ACCEPT"
                    elif "REJECT" in val:
                        decision = "REJECT"
                    else:
                        decision = "UNKNOWN"
                elif line.upper().startswith("JUSTIFICATION:"):
                    justification = line.split(":",1)[1].strip()

            # Fallback: If not found, try regex (catch all)
            if decision == "UNKNOWN":
                import re
                m = re.search(r"DECISION\s*:\s*(ACCEPT|REJECT)", llm_text, re.I)
                if m: decision = m.group(1).upper()
            if justification == "Ollama response parsing failed.":
                m = re.search(r"JUSTIFICATION\s*:\s*(.*)", llm_text, re.I | re.S)
                if m: justification = m.group(1).strip()
            if not justification:
                justification = llm_text

            print(f"Decision: {decision}\nJustification: {justification}\n")
            return decision, justification

        except ollama.ResponseError as e:
            logging.error(f"Ollama API error for {self.name}: {e}")
            print(f"Ollama API error for {self.name}: {e}")
            return "ERROR", f"Ollama API call failed: {e}"
        except Exception as e:
            logging.error(f"General error during Ollama call for {self.name}: {e}")
            print(f"General error during Ollama call for {self.name}: {e}")
            return "ERROR", f"Local LLM call failed: {e}"

class DebateConsensusEngine:
    def __init__(self, model_a: ModelInterface, model_b: ModelInterface, max_rounds: int = 5):
        self.model_a = model_a
        self.model_b = model_b
        self.max_rounds = max_rounds
        logging.info("Debate Consensus Engine initialized.")

    def debate_candidate(self, candidate: CandidateMemory):
        logging.info(f"Starting debate for Candidate ID: {candidate.id}")
        print(f"\n--- Starting debate for Candidate {candidate.id} ---")
        round_count = 0
        decision_a, justification_a = None, None
        decision_b, justification_b = None, None

        while round_count < self.max_rounds:
            if round_count == 0:
                decision_a, justification_a = self.model_a.evaluate(candidate)
                decision_b, justification_b = self.model_b.evaluate(candidate)
            else:
                decision_a, justification_a = self.model_a.evaluate(candidate, justification_b)
                decision_b, justification_b = self.model_b.evaluate(candidate, justification_a)

            # Log round
            log_entry = {
                'round': round_count + 1,
                'model_a_decision': decision_a,
                'model_a_justification': justification_a,
                'model_b_decision': decision_b,
                'model_b_justification': justification_b
            }
            candidate.debate_log.append(log_entry)
            logging.info(f"Debate round {round_count+1} for Candidate {candidate.id}: {log_entry}")

            # Print to screen
            print(f"Round {round_count+1} Results:")
            print(f"  {self.model_a.name}: {decision_a} | {justification_a}")
            print(f"  {self.model_b.name}: {decision_b} | {justification_b}")

            # Consensus check
            if decision_a == decision_b and decision_a in ["ACCEPT", "REJECT"]:
                candidate.status = "ACCEPTED" if decision_a == "ACCEPT" else "REJECTED"
                print(f"\n*** Consensus reached: {candidate.status} ***\n")
                logging.info(f"Consensus reached for Candidate {candidate.id}: {candidate.status}")
                return candidate.status
            elif decision_a == "ERROR" or decision_b == "ERROR":
                candidate.status = "ERROR_DEBATE"
                print("\n*** Debate terminated due to error. ***\n")
                logging.error(f"Debate for Candidate {candidate.id} terminated due to LLM error.")
                return candidate.status
            round_count += 1

        candidate.status = "RETAINED"
        print("\n*** No consensus, candidate retained for future debate. ***\n")
        logging.info(f"No consensus for Candidate {candidate.id}, status: RETAINED")
        return candidate.status

    def run(self, candidate_pool: List[CandidateMemory]):
        logging.info(f"Running debate engine for {len(candidate_pool)} candidates.")
        for candidate in candidate_pool:
            self.debate_candidate(candidate)
            if candidate.status == "ACCEPTED":
                add_to_training_set(candidate)
            # Avoid hammering your GPU
            time.sleep(2)

def add_to_training_set(candidate: CandidateMemory):
    with open("training_set.txt", "a") as f:
        f.write(f"{candidate.content}\n\n")
    logging.info(f"Candidate {candidate.id} added to training_set.txt")

# --- Main Loop ---
if __name__ == "__main__":
    # Create models with different personalities/temperatures
    model_a = ModelInterface("ModelA", ollama_model_name=OLLAMA_LLM_MODEL_NAME, temperature=0.7)
    model_b = ModelInterface("ModelB", ollama_model_name=OLLAMA_LLM_MODEL_NAME, temperature=0.3)

    # Sample candidate memories
    candidate_pool = [
        CandidateMemory("1", "The quick brown fox jumps over the lazy dog. This is a very common phrase."),
        CandidateMemory("2", "A deep dive into the historical context and societal impact of the Industrial Revolution, focusing on its influence on labor laws."),
        CandidateMemory("3", "A recipe for chocolate chip cookies: flour, sugar, butter, eggs, chocolate chips, baking soda, salt. Mix and bake at 375F for 10-12 minutes."),
        CandidateMemory("4", "A fictional story about a wizard's apprentice who discovers a lost spell, featuring detailed descriptions of magical creatures and ancient runes."),
        CandidateMemory("5", "A highly biased political rant containing unsubstantiated claims and derogatory language. This should be rejected."),
        CandidateMemory("6", "Detailed instructions on how to assemble a complex piece of IKEA furniture, including troubleshooting tips for common issues."),
        CandidateMemory("7", "A list of facts about the solar system: Mercury is the smallest, Jupiter is the largest, Earth is the only known planet with life, etc."),
    ]

    # Run debate engine
    engine = DebateConsensusEngine(model_a, model_b, max_rounds=3)
    engine.run(candidate_pool)

    # Print summary
    print("\n--- Final Candidate Statuses and Debate Logs ---")
    for candidate in candidate_pool:
        print(f"\nCandidate {candidate.id}: Status = {candidate.status}")
        for log in candidate.debate_log:
            print(f"  Round {log['round']}:")
            print(f"    {model_a.name}: {log['model_a_decision']} | {log['model_a_justification']}")
            print(f"    {model_b.name}: {log['model_b_decision']} | {log['model_b_justification']}")
        print("-" * 40)

    print(f"\n*** Detailed log written to {LOG_FILENAME} ***")
    print("*** Accepted content appended to training_set.txt ***")
    
    ____________________________-prompts for team _____________________________
    
    
