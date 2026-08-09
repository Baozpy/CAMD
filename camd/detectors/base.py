from abc import ABC, abstractmethod

from camd.context.models import DefectPrediction


class BaseDetector(ABC):

    @abstractmethod
    def detect(self, code: str) -> DefectPrediction:
        raise NotImplementedError