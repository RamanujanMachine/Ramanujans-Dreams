"""
Global config file for system flow regarding databases
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from .configurable import Configurable


class DBUsages(Enum):
    RETRIEVE_DATA = auto()
    STORE_DATA = auto()
    STORE_THEN_RETRIEVE = auto()


@dataclass
class DBConfig(Configurable):
    USAGE: DBUsages = field(
        default=DBUsages.STORE_THEN_RETRIEVE,
        metadata={"description": "Database access mode controlling retrieve/store behavior across pipeline runs."},
    )


db_config: DBConfig = DBConfig()
