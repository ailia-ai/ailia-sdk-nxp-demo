"""Append-only measurements; never clear another backend when a run starts."""

import json
from collections import deque
from datetime import datetime

from settings import BACKENDS


def load_history(path, limit=200):
    records = deque(maxlen=limit)
    latest = {}
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
                if (
                    record.get("backend") in BACKENDS
                    and len(record.get("values", [])) == 5
                    and all(isinstance(v, str) for v in record["values"])
                    and isinstance(record.get("time"), str)
                ):
                    records.append(record)
                    latest[record["backend"]] = record
            except (ValueError, AttributeError, TypeError):
                continue
    # Restore every backend's latest value even if it predates the visible
    # history window. The GUI still limits the scrolling list to 200 rows.
    visible = {id(record) for record in records}
    older = [record for record in latest.values() if id(record) not in visible]
    return sorted(older, key=lambda record: record["time"]) + list(records)


def append_history(path, backend, values, image):
    record = dict(
        time=datetime.now().astimezone().isoformat(timespec="seconds"),
        backend=backend,
        values=values,
        image=str(image),
        memory_mode=11,
    )
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
