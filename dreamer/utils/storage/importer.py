import json
import os
import pickle as pkl
from .formats import Formats
from pathlib import Path


class Importer:
    """
    A utility class for importing data from pickle or JSON files.
    """

    @classmethod
    def imprt(cls, path: str):
        """
        Imports data from the provided path.
        :param path: Path to the file where the data is stored.
        """
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")

        if os.path.isdir(path):
            data = dict()
            for f in os.listdir(path):
                data[f] = cls.imprt(os.path.join(path, f))
            return data

        match path.split('.')[-1]:
            case Formats.JSON.value:
                with open(path, 'r') as f:
                    data = json.load(f)
                    return cls._json_restore(data)
            case Formats.PICKLE.value:
                with open(path, 'rb') as f:
                    return pkl.load(f)
            case Formats.JSONL.value:
                return cls._read_jsonl(path, merge=False)
            case _:
                raise ValueError(f"File {path} has unsupported format")

    @classmethod
    def _read_jsonl(cls, path: str, merge: bool = False) -> list:
        """Read a JSON-Lines file into a list of dicts, skipping blank/malformed lines.

        When *merge* is ``True``, records are merged by the composite
        ``(trajectory_id, constant)`` key (later lines win for conflicting
        columns).  Records are **flat**, so this is a plain last-write-wins
        ``dict.update`` — no special-cased nested merge.  Records without a
        ``trajectory_id`` key are appended unchanged after the merged rows.

        DTO reconstruction is left to the caller (e.g. via ``TrajectoryDTO.from_dict``)
        because a JSONL file may mix records from different DTO classes.
        """
        records = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not merge:
            return records

        merged: dict = {}
        unkeyed: list = []
        for r in records:
            tid = r.get("trajectory_id")
            if tid is None:
                unkeyed.append(r)
                continue
            key = (tid, r.get("constant"))
            if key not in merged:
                merged[key] = r
            else:
                merged[key].update(r)   # flat merge, later wins
        return list(merged.values()) + unkeyed

    @classmethod
    def _json_restore(cls, data):
        """Recursively rebuild supported objects from JSON payloads."""
        if isinstance(data, list):
            return [cls._json_restore(v) for v in data]
        if isinstance(data, dict):
            class_name = data.get("__class__")
            if class_name == "DataManager":
                from dreamer.utils.storage.storage_objects import DataManager
                return DataManager.from_json_obj(data)
            if class_name == "Shard":
                from dreamer.extraction.shard import Shard
                return Shard.from_json_obj(data)
            return {k: cls._json_restore(v) for k, v in data.items()}
        return data

    @classmethod
    def import_stream(cls, path):
        """
        A generator for data (imports data from directory in chunks)
        :param path: Path of directory to import from as stream
        """
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")

        if not os.path.isdir(path):
            raise NotADirectoryError(f"{path} is not a directory")

        path = Path(path)
        for file in path.rglob('*'):
            if file.is_file():
                yield cls.imprt(str(file))
