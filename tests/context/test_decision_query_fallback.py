import pytest
from datetime import datetime, timedelta

from semantica.context.context_graph import ContextGraph
from semantica.context.decision_query import DecisionQuery
from semantica.context.decision_models import Decision
from semantica.context.decision_recorder import DecisionRecorder

@pytest.fixture
def memory_components():
    cg = ContextGraph()
    recorder = DecisionRecorder(graph_store=cg)
    dq = DecisionQuery(graph_store=cg)
    return cg, recorder, dq

def test_decision_query_contextgraph_fallback(memory_components):
    """Test all DecisionQuery paths with native ContextGraph fallback execution."""
    cg, recorder, dq = memory_components
    
    entity_1 = "entity_user_1"
    entity_2 = "entity_company_1"
    now = datetime.now()
    
    dec1_id = recorder.record_decision(
        Decision(
            decision_id="dec_1",
            category="loan_approval",
            scenario="User requested loan",
            reasoning="Good credit",
            outcome="approved",
            confidence=0.9,
            timestamp=now - timedelta(days=2),
            decision_maker="system",
            metadata={"amount": 5000}
        ),
        entities=[entity_1],
        source_documents=[]
    )
    
    dec2_id = recorder.record_decision(
        Decision(
            decision_id="dec_2",
            category="risk_assessment",
            scenario="Company requested credit line",
            reasoning="High debt ratio",
            outcome="rejected",
            confidence=0.95,
            timestamp=now - timedelta(days=1),
            decision_maker="analyst",
            metadata={"amount": 50000}
        ),
        entities=[entity_2],
        source_documents=[]
    )
    
    # Test find_by_category
    print("Graph Stats: ", cg.stats())
    print("All Nodes: ", cg.find_nodes())
    loans = dq.find_by_category("loan_approval")
    print("Loans Found: ", loans)
    assert len(loans) == 1
    assert loans[0].decision_id == "dec_1"
    
    #  Test find_by_entity
    user_decisions = dq.find_by_entity(entity_1)
    assert len(user_decisions) == 1
    assert user_decisions[0].decision_id == "dec_1"
    
    # Test find_by_time_range
    recent_decisions = dq.find_by_time_range(now - timedelta(days=3), now)
    assert len(recent_decisions) == 2
    
    # Add a precedent link for tracing & multihop
    recorder.link_precedents(dec1_id, [dec2_id], ["SIMILAR_SCENARIO"])
    
    # Test multi_hop_reasoning
    multi_hop = dq.multi_hop_reasoning(dec1_id, "", max_hops=2)
    assert len(multi_hop) >= 1  # Should find connected precedents
    
    # Test trace_decision_path
    paths = dq.trace_decision_path(dec1_id, ["SIMILAR_SCENARIO"])
    assert len(paths) == 1
    
    # Test find_similar_exceptions
    recorder.record_exception(
        decision_id=dec2_id,
        policy_id="pol_1",
        reason="Market downturn special condition",
        approver="manager",
        approval_method="email",
        justification="Allowed due to macro factors"
    )
    exceptions = dq.find_similar_exceptions("Market downturn", limit=5)
    assert len(exceptions) == 1
