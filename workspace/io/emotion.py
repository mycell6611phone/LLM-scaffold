"""
emotion.py

Manages the operational "mood" of the AGI. These moods are not simulations
of human emotion but are functional states that modify the system prompts
of the LLMs to alter their behavior during tasks like planning, debate,
or self-critique. This provides a mechanism for behavioral control without
relying on model parameters like temperature.
"""
from typing import Dict, Optional, List

# A dictionary mapping mood names to descriptive prompts that will be
# prepended to an LLM's main system prompt.
MOOD_PROMPTS: Dict[str, str] = {
    "NEUTRAL": "You are operating in a standard, neutral state.",
    "FOCUSED": "You are in a focused state. Prioritize precision, logic, and accuracy above all else. Avoid speculation.",
    "CREATIVE": "You are in a creative state. Brainstorm expansive ideas and explore novel solutions. Do not limit yourself to obvious answers.",
    "CAUTIOUS": "You are in a cautious state. Be skeptical and risk-averse. Double-check all facts and assumptions before proceeding. Prioritize safety and stability.",
    "EFFICIENT": "You are in an efficient state. Provide the most direct and concise answers possible to complete the task quickly."
}

class EmotionManager:
    """
    Manages the AGI's current operational mood.
    """

    def __init__(self, initial_mood: str = "NEUTRAL"):
        """
        Initializes the EmotionManager with a starting mood.

        Args:
            initial_mood (str): The name of the initial mood. Must be a key
                                in the MOOD_PROMPTS dictionary.
        """
        if initial_mood not in MOOD_PROMPTS:
            raise ValueError(f"Invalid initial mood '{initial_mood}'. Please use one of {list(MOOD_PROMPTS.keys())}")
        self._current_mood: str = initial_mood
        print(f"EmotionManager: Initialized with mood '{self._current_mood}'.")

    def get_current_mood_prompt(self) -> str:
        """
        Returns the descriptive prompt for the current mood.

        Returns:
            str: The system prompt text associated with the current mood.
        """
        return MOOD_PROMPTS[self._current_mood]

    def set_mood(self, new_mood: str) -> bool:
        """
        Sets a new operational mood.

        Args:
            new_mood (str): The name of the new mood to set.

        Returns:
            bool: True if the mood was successfully changed, False otherwise.
        """
        if new_mood in MOOD_PROMPTS:
            self._current_mood = new_mood
            print(f"EmotionManager: Mood changed to '{new_mood}'.")
            return True
        else:
            print(f"EmotionManager: Error - Mood '{new_mood}' is not a valid mood.")
            return False

    def list_available_moods(self) -> List[str]:
        """
        Returns a list of all available mood names.

        Returns:
            List[str]: A list of keys from the MOOD_PROMPTS dictionary.
        """
        return list(MOOD_PROMPTS.keys())

# --- Example Usage ---
if __name__ == "__main__":
    print("--- Initializing Emotion Manager ---")
    emotion_manager = EmotionManager()

    print("\n--- Current Mood Prompt ---")
    print(emotion_manager.get_current_mood_prompt())

    print("\n--- Setting a New Mood ---")
    emotion_manager.set_mood("CAUTIOUS")
    print(f"New mood is: {emotion_manager._current_mood}")
    print(emotion_manager.get_current_mood_prompt())
    
    print("\n--- Attempting to Set an Invalid Mood ---")
    emotion_manager.set_mood("ANGRY")

    print("\n--- Listing All Available Moods ---")
    available_moods = emotion_manager.list_available_moods()
    print(available_moods)
