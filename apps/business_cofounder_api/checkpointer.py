from __future__ import annotations

import os
import pickle
import threading
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver


class DiskBackedInMemorySaver(InMemorySaver):
    """A disk-persisted variant of InMemorySaver (single-file, atomic writes).

    This persists LangGraph checkpoints/writes/blobs so API calls can resume the same
    conversation (same thread_id) across process restarts.
    """

    def __init__(self, *, file_path: str | Path) -> None:
        super().__init__()
        self._file_path = Path(file_path)
        self._lock = threading.RLock()
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        with self._lock:
            if not self._file_path.exists():
                return
            try:
                payload = pickle.loads(self._file_path.read_bytes())
                if not isinstance(payload, dict):
                    return

                storage = payload.get("storage", {})
                writes = payload.get("writes", {})
                blobs = payload.get("blobs", {})
                if (
                    not isinstance(storage, dict)
                    or not isinstance(writes, dict)
                    or not isinstance(blobs, dict)
                ):
                    return

                # Rehydrate defaultdicts without pickling lambdas.
                self.storage = defaultdict(lambda: defaultdict(dict))
                for thread_id, ns_map in storage.items():
                    if not isinstance(ns_map, dict):
                        continue
                    self.storage[thread_id] = defaultdict(dict)
                    for checkpoint_ns, ckpt_map in ns_map.items():
                        if isinstance(ckpt_map, dict):
                            self.storage[thread_id][checkpoint_ns] = dict(ckpt_map)

                self.writes = defaultdict(dict)
                for outer_key, inner_map in writes.items():
                    if isinstance(inner_map, dict):
                        self.writes[outer_key] = dict(inner_map)

                self.blobs = dict(blobs)
            except Exception as e:  # noqa: BLE001
                warnings.warn(
                    f"Failed to load checkpoints from {self._file_path}: {e!s}. "
                    "Starting with a fresh in-memory state.",
                    stacklevel=2,
                )

    def _dump_to_disk(self) -> None:
        with self._lock:
            try:
                self._file_path.parent.mkdir(parents=True, exist_ok=True)
                payload: dict[str, Any] = {
                    "storage": {
                        thread_id: {ns: dict(ckpts) for ns, ckpts in ns_map.items()}
                        for thread_id, ns_map in self.storage.items()
                    },
                    "writes": {k: dict(v) for k, v in self.writes.items()},
                    "blobs": dict(self.blobs),
                }
                tmp = self._file_path.with_suffix(self._file_path.suffix + ".tmp")
                tmp.write_bytes(pickle.dumps(payload))
                os.replace(tmp, self._file_path)
            except Exception as e:  # noqa: BLE001
                warnings.warn(
                    f"Failed to persist checkpoints to {self._file_path}: {e!s}.",
                    stacklevel=2,
                )

    def put(self, config, checkpoint, metadata, new_versions):  # type: ignore[override]
        out = super().put(config, checkpoint, metadata, new_versions)
        self._dump_to_disk()
        return out

    def put_writes(self, config, writes, task_id, task_path: str = ""):  # type: ignore[override]
        super().put_writes(config, writes, task_id, task_path)
        self._dump_to_disk()

    def delete_thread(self, thread_id: str) -> None:  # type: ignore[override]
        super().delete_thread(thread_id)
        self._dump_to_disk()


