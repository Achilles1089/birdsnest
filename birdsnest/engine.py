########################################################################################################
# Bird's Nest — Engine Base Class
# Abstract interface for all non-transformer inference engines
########################################################################################################

from abc import ABC, abstractmethod
from typing import Generator, Optional, Dict, Any
import torch


class InferenceEngine(ABC):
    """Base class for all Bird's Nest inference engines."""

    name: str = "base"
    architecture: str = "unknown"

    def __init__(self):
        self.model = None
        self.model_path: Optional[str] = None
        self.model_name: Optional[str] = None
        self.is_loaded: bool = False
        self.device: str = "cpu"
        self.model_info: Dict[str, Any] = {}

    @abstractmethod
    def load(self, model_path: str) -> Dict[str, Any]:
        """Load a model from disk. Returns model metadata."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free GPU memory."""
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        temperature: float = 1.0,
        top_p: float = 0.7,
        max_tokens: int = 500,
        system_prefix: str = "",
    ) -> Generator[str, None, None]:
        """Generate text token-by-token, yielding each decoded piece."""
        pass

    @abstractmethod
    def encode(self, text: str) -> list:
        """Encode text to token IDs."""
        pass

    @abstractmethod
    def decode(self, tokens: list) -> str:
        """Decode token IDs to text."""
        pass

    def get_status(self) -> Dict[str, Any]:
        """Return engine status for API."""
        return {
            "engine": self.name,
            "architecture": self.architecture,
            "loaded": self.is_loaded,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "info": self.model_info,
        }

    @staticmethod
    def detect_device() -> str:
        """Pick the best available device for inference."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"
