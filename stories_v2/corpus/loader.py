"""Load exemplars from JSON files on disk.

The corpus directory layout:

    corpus/
        exemplars/
            prose_techniques_*.json     # bundled by us
            cultivation_*.json          # user-curated public-domain
            litrpg_*.json
            isekai_*.json
            light_novel_*.json
            index.json                  # optional: catalog metadata

Each *.json file is either a single CorpusEntry dict or a list of them.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional

from pydantic import ValidationError

from .schema import CorpusEntry

logger = logging.getLogger(__name__)

EXEMPLARS_DIR = Path(__file__).parent / "exemplars"


@lru_cache(maxsize=1)
def load_index() -> List[CorpusEntry]:
    """Load all .json files under ``exemplars/`` and return validated entries.

    Cached at process start. Drop a new file in and restart to pick up.
    """
    entries: List[CorpusEntry] = []
    if not EXEMPLARS_DIR.exists():
        logger.warning("corpus: %s does not exist; returning empty index", EXEMPLARS_DIR)
        return entries

    for path in sorted(EXEMPLARS_DIR.glob("*.json")):
        if path.name == "index.json":
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("corpus: failed to load %s: %s", path, exc)
            continue

        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                entries.append(CorpusEntry.model_validate(item))
            except ValidationError as exc:
                logger.warning("corpus: invalid entry in %s: %s", path, exc)
                continue

    logger.info("corpus: loaded %d exemplars from %s", len(entries), EXEMPLARS_DIR)
    return entries


def list_exemplars(
    *,
    genres: Optional[Iterable[str]] = None,
    techniques: Optional[Iterable[str]] = None,
    pov: Optional[str] = None,
) -> List[CorpusEntry]:
    """Filter the loaded corpus by tags. Empty filters = pass-through."""
    pool = load_index()
    if not pool:
        return []

    g_set = set(g.lower() for g in (genres or []) if g)
    t_set = set(t.lower() for t in (techniques or []) if t)

    out: List[CorpusEntry] = []
    for entry in pool:
        if g_set and not (set(x.lower() for x in entry.genres) & g_set or "generic" in (x.lower() for x in entry.genres)):
            continue
        if t_set and not (set(x.lower() for x in entry.techniques) & t_set):
            continue
        if pov and entry.pov and entry.pov != pov:
            continue
        out.append(entry)
    return out
