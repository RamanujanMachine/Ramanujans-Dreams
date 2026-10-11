from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class JSONable(ABC):
    """Mixin contract for objects that can be exported through JSON storage paths."""

    @abstractmethod
    def to_json(self) -> Dict[str, Any]:
        """
        Convert the object into a JSON-serializable dictionary.
        :return: Serializable dictionary payload.
        """
        raise NotImplementedError()

