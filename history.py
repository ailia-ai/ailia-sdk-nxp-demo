"""Append-only measurements; never clear another backend when a run starts."""

import json
from collections import deque
from datetime import datetime

from settings import BACKENDS, MODEL_REVISION


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
                    and record.get("model", "sam") in ("sam", "grounding", "depth")
                    and (record.get("caption") is None or isinstance(record["caption"], str))
                ):
                    records.append(record)
                    latest[(record.get("model", "sam"), record["backend"])] = record
            except (ValueError, AttributeError, TypeError):
                continue
    # Restore every backend's latest value even if it predates the visible
    # history window. The GUI still limits the scrolling list to 200 rows.
    visible = {id(record) for record in records}
    older = [record for record in latest.values() if id(record) not in visible]
    return sorted(older, key=lambda record: record["time"]) + list(records)


def append_history(path, backend, values, image, model="sam", caption=None):
    from depth_inputs import REVISION as DEPTH_REVISION
    from grounding_inputs import REVISION

    record = dict(
        time=datetime.now().astimezone().isoformat(timespec="seconds"),
        backend=backend,
        values=values,
        image=str(image),
        memory_mode=11,
        model_revision={"grounding": REVISION, "depth": DEPTH_REVISION}.get(model, MODEL_REVISION),
        model=model,
        caption=caption,
    )
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
