"""
Integration tests for the Semantica Knowledge Explorer API.

Uses FastAPI's TestClient (from starlette.testclient).
"""

import json
import pytest

from semantica.context.context_graph import ContextGraph
from semantica.explorer.app import create_app
from semantica.explorer.session import GraphSession


try:
    from starlette.testclient import TestClient
except ImportError:
    pytest.skip(
        "starlette (TestClient) is required for explorer tests. "
        "Install with: pip install semantica[explorer]",
        allow_module_level=True,
    )



def _build_sample_graph() -> ContextGraph:
    """Create a small ContextGraph with a handful of nodes and edges."""
    g = ContextGraph(advanced_analytics=False)

    g.add_node("python", node_type="language", content="Python programming language",
               popularity="high")
    g.add_node("javascript", node_type="language", content="JavaScript programming language")
    g.add_node("web_dev", node_type="concept", content="Web Development")
    g.add_node("ml", node_type="concept", content="Machine Learning")
    g.add_node("decision_1", node_type="decision", content="Approve ML framework",
               category="tech", scenario="Choosing ML framework", outcome="approved",
               confidence="0.9", reasoning="Best performance")
    g.add_node("decision_2", node_type="decision", content="Reject legacy stack",
               category="tech", scenario="Choosing ML framework alternative",
               outcome="rejected", confidence="0.4", reasoning="Outdated")
    g.add_node("temporal_node", node_type="event", content="Conference talk",
               valid_from="2025-01-01T00:00:00", valid_until="2025-12-31T23:59:59")

    g.add_edge("python", "ml", edge_type="used_in", weight=0.9)
    g.add_edge("javascript", "web_dev", edge_type="used_in", weight=0.8)
    g.add_edge("python", "web_dev", edge_type="used_in", weight=0.5)
    g.add_edge("decision_1", "ml", edge_type="about")

    return g


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient backed by a sample graph."""
    graph = _build_sample_graph()
    session = GraphSession(graph)
    app = create_app(session=session)
    with TestClient(app) as c:
        yield c



class TestHealthInfo:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_info(self, client):
        r = client.get("/api/info")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "Semantica Knowledge Explorer"
        assert "version" in body



class TestGraphNodes:
    def test_list_nodes(self, client):
        r = client.get("/api/graph/nodes")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] >= 5
        assert len(body["nodes"]) <= body["total"]

    def test_list_nodes_pagination(self, client):
        r = client.get("/api/graph/nodes?skip=0&limit=2")
        assert r.status_code == 200
        body = r.json()
        assert len(body["nodes"]) == 2
        assert body["limit"] == 2

    def test_list_nodes_filter_type(self, client):
        r = client.get("/api/graph/nodes?type=language")
        assert r.status_code == 200
        body = r.json()
        assert all(n["type"] == "language" for n in body["nodes"])

    def test_list_nodes_search(self, client):
        r = client.get("/api/graph/nodes?search=python")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] >= 1

    def test_get_node(self, client):
        r = client.get("/api/graph/node/python")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == "python"
        assert body["type"] == "language"

    def test_get_node_not_found(self, client):
        r = client.get("/api/graph/node/nonexistent_xyz")
        assert r.status_code == 404

    def test_get_neighbors(self, client):
        r = client.get("/api/graph/node/python/neighbors")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        assert len(body) >= 1
        ids = [nb["id"] for nb in body]
        assert "ml" in ids or "web_dev" in ids




class TestGraphEdges:
    def test_list_edges(self, client):
        r = client.get("/api/graph/edges")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] >= 3

    def test_list_edges_filter_type(self, client):
        r = client.get("/api/graph/edges?type=used_in")
        assert r.status_code == 200
        body = r.json()
        assert all(e["type"] == "used_in" for e in body["edges"])

    def test_list_edges_filter_source(self, client):
        r = client.get("/api/graph/edges?source=python")
        assert r.status_code == 200
        body = r.json()
        assert all(e["source"] == "python" for e in body["edges"])




class TestSearchStats:
    def test_search(self, client):
        r = client.post("/api/graph/search", json={"query": "programming", "limit": 5})
        assert r.status_code == 200
        body = r.json()
        assert body["query"] == "programming"
        assert len(body["results"]) >= 1

    def test_stats(self, client):
        r = client.get("/api/graph/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["node_count"] >= 5
        assert body["edge_count"] >= 3
        assert "density" in body


class TestDecisions:
    def test_list_decisions(self, client):
        r = client.get("/api/decisions")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        assert len(body) >= 1

    def test_list_decisions_category(self, client):
        r = client.get("/api/decisions?category=tech")
        assert r.status_code == 200
        body = r.json()
        assert all(d["category"] == "tech" for d in body)

    def test_get_decision(self, client):
        r = client.get("/api/decisions/decision_1")
        assert r.status_code == 200
        assert r.json()["decision_id"] == "decision_1"

    def test_get_decision_not_found(self, client):
        r = client.get("/api/decisions/nope")
        assert r.status_code == 404

    def test_causal_chain(self, client):
        r = client.get("/api/decisions/decision_1/chain")
        assert r.status_code == 200
        body = r.json()
        assert body["decision_id"] == "decision_1"
        assert isinstance(body["chain"], list)

    def test_precedents(self, client):
        r = client.get("/api/decisions/decision_1/precedents")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_compliance(self, client):
        r = client.get("/api/decisions/decision_1/compliance")
        assert r.status_code == 200
        body = r.json()
        assert "compliant" in body




class TestTemporal:
    def test_snapshot_now(self, client):
        r = client.get("/api/temporal/snapshot")
        assert r.status_code == 200
        body = r.json()
        assert "active_node_count" in body

    def test_snapshot_at(self, client):
        r = client.get("/api/temporal/snapshot?at=2025-06-15T00:00:00")
        assert r.status_code == 200
        body = r.json()
        ids = [n["id"] for n in body["active_nodes"]]
        assert "temporal_node" in ids

    def test_diff(self, client):
        r = client.get(
            "/api/temporal/diff"
            "?from_time=2024-01-01T00:00:00"
            "&to_time=2025-06-15T00:00:00"
        )
        assert r.status_code == 200
        body = r.json()
        assert "added_nodes" in body
        assert "removed_nodes" in body




class TestAnalytics:
    def test_analytics(self, client):
        r = client.get("/api/analytics")
        assert r.status_code == 200

    def test_validation(self, client):
        r = client.get("/api/analytics/validation")
        assert r.status_code == 200
        body = r.json()
        assert "valid" in body



class TestReasoning:
    def test_reason(self, client):
        r = client.post(
            "/api/reason",
            json={
                "facts": ["Person(Alice)", "Knows(Alice, Bob)"],
                "rules": ["IF Knows(?x, ?y) THEN Connected(?x, ?y)"],
                "mode": "forward",
            },
        )
        assert r.status_code in (200, 422)




class TestAnnotations:
    def test_create_and_list(self, client):
    
        r = client.post(
            "/api/annotations",
            json={"node_id": "python", "content": "Great language!", "tags": ["fav"]},
        )
        assert r.status_code == 201
        ann = r.json()
        assert ann["node_id"] == "python"
        ann_id = ann["annotation_id"]

        # List all
        r = client.get("/api/annotations")
        assert r.status_code == 200
        assert any(a["annotation_id"] == ann_id for a in r.json())

        # List filtered by node
        r = client.get("/api/annotations?node_id=python")
        assert r.status_code == 200
        assert all(a["node_id"] == "python" for a in r.json())

        # Delete
        r = client.delete(f"/api/annotations/{ann_id}")
        assert r.status_code == 204

    def test_create_annotation_bad_node(self, client):
        r = client.post(
            "/api/annotations",
            json={"node_id": "nonexistent", "content": "oops"},
        )
        assert r.status_code == 404

    def test_delete_annotation_not_found(self, client):
        r = client.delete("/api/annotations/no_such_id")
        assert r.status_code == 404



class TestExport:
    def test_export_json(self, client):
        r = client.post("/api/export", json={"format": "json"})
        assert r.status_code == 200
        assert "json" in r.headers.get("content-type", "").lower() or len(r.content) > 0

    def test_export_unsupported(self, client):
        r = client.post("/api/export", json={"format": "pdf"})
        assert r.status_code == 422




class TestImport:
    def test_import_json(self, client):
        payload = json.dumps({
            "nodes": [
                {"id": "imported_node", "type": "test", "properties": {"content": "hello"}}
            ],
            "edges": [],
        })
        r = client.post(
            "/api/import",
            files={"file": ("import.json", payload, "application/json")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["nodes_added"] >= 1

        r2 = client.get("/api/graph/node/imported_node")
        assert r2.status_code == 200
