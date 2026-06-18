"""
Base class for all Smart Lamp feature modules.
Every feature (gesture, focus, pomodoro, environment, OCR) implements this interface
so the SystemManager can treat them uniformly.
"""

from abc import ABC, abstractmethod


class FeatureModule(ABC):
    """Standard interface for all smart lamp feature modules."""

    @abstractmethod
    def get_state(self) -> dict:
        """Return the current state of this module as a dict for the web dashboard."""
        pass

    @abstractmethod
    def handle_voice_command(self, text: str) -> bool:
        """
        Handle a voice command string.
        Return True if this module consumed the command, False otherwise.
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Release any hardware or software resources held by this module."""
        pass
