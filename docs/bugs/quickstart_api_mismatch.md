---
title: Quickstart sample code incompatible with v0.3.0 API
labels: bug, documentation
version: 0.3.0
---

## Bug 1 — `ContextGraph.add_decision()` rejects keyword arguments

Running the README's Context & Decision Tracking sample fails immediately:

```
TypeError: ContextGraph.add_decision() got an unexpected keyword argument 'category'
```

`add_decision()` only accepted a `Decision` object, but the docs showed and described the kwargs form. Fixed by updating `add_decision()` to accept kwargs directly (delegates to `record_decision`); both call patterns now work and return the decision ID.

---

## Bug 2 — `VectorStore(backend="faiss")` silently drops all stored memories

Running any quickstart snippet with `VectorStore(backend="faiss", dimension=768)` prints:

```
Failed to store in vector store:
```

FAISS requires `pip install faiss-cpu`, which is not included in the base install. Memories fall back to an in-memory dict silently, so `find_precedents` and similarity search return empty results. Fixed by changing all quickstart snippets to `VectorStore(backend="inmemory")`.

---

Reported by: chrisguoado — tracked in KaifAhmad1/semantica#433, fixed in pr-432 follow-up.
