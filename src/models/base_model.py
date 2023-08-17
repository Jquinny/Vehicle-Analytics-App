from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import norfair
import numpy as np


class BaseModel(ABC):
    """abstract base class for defining model objects for detection and classification"""

    @abstractmethod
    def setup(self, model_path: str, params: dict):
        pass

    @abstractmethod
    def inference(
        self, img: np.ndarray, **runtime_args
    ) -> Tuple[int, float] | List[norfair.tracker.Detection]:
        pass

    @abstractmethod
    def get_classes(self) -> Dict[int, str | None]:
        pass
