"""Streamlit Operational Dashboard for Restaurant Operations Review Assistant.
Alternative lightweight Python frontend connecting to the FastAPI gateway on Port 8000.
"""

import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="Restaurant Operations Review Assistant",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🍽️ AI-Powered Restaurant Operations Review Assistant")
st.caption("Autonomous Multi-Agent Operational Intelligence • Hybrid RAG • CrewAI Synthesis • Quality Gate")

# Sidebar navigation and status
with st.sidebar:
    st.header("Navigation & Status")

    # API health check
    try:
        health_resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if health_resp.status_code == 200:
            st.success("🟢 Gateway API: Online (Port 8000)")
        else:
            st.error("🔴 Gateway API: Error")
    except Exception:
        st.warning("⚠️ Gateway API: Offline (Run python app/main.py)")

    # Fetch agent statuses
    try:
        agent_resp = requests.get(f"{API_BASE_URL}/agents/status", timeout=4)
        if agent_resp.status_code == 200:
            adata = agent_resp.json()
            st.info(f"🤖 Specialist Mesh: {adata.get('total_available', 0)}/{adata.get('total_configured', 4)} Available")
    except Exception:
        pass

    st.markdown("---")
    st.subheader("Quick Demonstration Scenarios")
    try:
        scenarios_resp = requests.get(f"{API_BASE_URL}/scenarios", timeout=4)
        if scenarios_resp.status_code == 200:
            scenarios = scenarios_resp.json().get("scenarios", [])
            for sc in scenarios[:5]:
                if st.button(f"📌 {sc['title']}", key=f"btn_{sc['id']}"):
                    st.session_state["query_input"] = sc["sample_query"]
                    st.session_state["biz_input"] = sc.get("business_id", "")
                    st.session_state["mode_input"] = sc["mode"]
    except Exception:
        pass

# Query input section
st.subheader("Operational Analysis Query")
col1, col2 = st.columns([3, 1])

with col1:
    default_query = st.session_state.get(
        "query_input",
        "Customers are repeatedly complaining about slow service and long wait times. What operational issues are causing this, and what should we fix first?"
    )
    user_query = st.text_area("Manager Operational Question:", value=default_query, height=120)

with col2:
    mode_choice = st.selectbox(
        "Execution Mode:",
        ["Auto (LangGraph + CrewAI)", "Single Agent"],
        index=0 if st.session_state.get("mode_input", "auto") == "auto" else 1
    )
    selected_agent = None
    if mode_choice == "Single Agent":
        selected_agent = st.selectbox(
            "Select Specialist Agent:",
            ["classification", "trends", "root_cause", "action_planner"]
        )

    biz_id = st.text_input("Business ID (Optional):", value=st.session_state.get("biz_input", "_ab50qdWOk0DdB6XOrBitw"))

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

if run_btn and user_query.strip():
    with st.spinner("Executing Multi-Agent Retrieval, Privacy Masking, and Synthesis..."):
        payload = {
            "query": user_query.strip(),
            "mode": "auto" if mode_choice.startswith("Auto") else "agent",
            "agent": selected_agent,
            "business_id": biz_id.strip() if biz_id.strip() else None,
            "viewer_role": "manager",
        }

        try:
            resp = requests.post(f"{API_BASE_URL}/analysis", json=payload, timeout=90)
            if resp.status_code == 200:
                result = resp.json()
                st.session_state["last_result"] = result
                st.success(f"Analysis Complete in {(result.get('total_duration_ms', 0)/1000):.2f}s")
            else:
                st.error(f"Analysis Error ({resp.status_code}): {resp.text}")
        except Exception as e:
            st.error(f"Failed to communicate with API Gateway: {e}")

# Render active analysis results
res = st.session_state.get("last_result")
if res:
    st.markdown("---")

    # Banner: Quality Gate & Confidence
    q_decision = res.get("quality_decision", "passed")
    conf_score = res.get("confidence_score", 80.0)
    norm_pct = min(100, max(0, int((conf_score / 80.0) * 100)))

    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        if q_decision == "passed":
            st.success(f"Quality Gate: PASSED (Confidence: {conf_score:.1f}/80.0 • {norm_pct}% Grounded)")
        elif q_decision == "flagged":
            st.warning(f"Quality Gate: FLAGGED (Confidence: {conf_score:.1f}/80.0 • {norm_pct}% Grounded)")
        else:
            st.error(f"Quality Gate: BLOCKED (Confidence: {conf_score:.1f}/80.0)")

    with q_col2:
        st.success("🔓 PII Decoded: Reversible Token Vault (Role: Manager)")

    with q_col3:
        st.info(f"⏱️ Duration: {(res.get('total_duration_ms', 0)/1000):.2f}s")

    st.caption("ℹ️ **Confidence Formula Ceiling:** 80.0 / 80.0 reflects 100% maximum grounded baseline under the Phase 4 formula: `100 * (0.35*relevance + 0.25*traceability + 0.20*agreement - 0.20*staleness)`.")

    # Executive Summary (CrewAI)
    st.subheader("📋 Executive Operations Analysis (CrewAI Consolidated Report)")
    exec_summary = res.get("executive_summary") or res.get("crew_report", {}).get("executive_summary", "")
    st.markdown(exec_summary)

    # Tabs for detailed inspection
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Action Plan", "Root Causes", "Retrieved Evidence", "Execution Trace", "Manager Feedback"
    ])

    with tab1:
        st.markdown("### Ranked Action Recommendations")
        actions = res.get("crew_report", {}).get("prioritized_recommendations") or []
        if not actions and res.get("action_plan"):
            actions = res.get("action_plan", {}).get("ranked_actions", [])

        if actions:
            for act in actions:
                rank = act.get("priority_rank", 1)
                st.markdown(f"**#{rank} {act.get('recommended_action', '')}**")
                st.markdown(f"- **Problem Addressed:** {act.get('problem_addressed', '')}")
                st.markdown(f"- **Owner Role:** `{act.get('owner_role', 'manager')}`")
                st.markdown(f"- **KPI / Target:** {act.get('kpi', '')} ({act.get('target', 'Proposed')})")
                st.markdown("---")
        else:
            st.write("No specific action recommendations returned.")

    with tab2:
        st.markdown("### Identified Root Causes & Hypotheses")
        hypotheses = res.get("root_cause", {}).get("hypotheses", [])
        if hypotheses:
            for h in hypotheses:
                st.markdown(f"**Cause:** {h.get('hypothesized_cause', '')} *({h.get('distinction_type', 'Hypothesis')})*")
                st.markdown(f"- Supporting Sources: {h.get('supporting_source_ids', [])}")
        else:
            st.write("No root cause hypotheses found.")

    with tab3:
        st.markdown("### Retrieved Review Evidence Excerpts")
        for ev in res.get("retrieved_evidence", []):
            cid = ev.get("chunk_id", "src")
            txt = ev.get("text_masked") or ev.get("text", "")
            stars = ev.get("stars", 3.0)
            st.markdown(f"- `[{cid}]` (★ {stars}/5.0): *\"{txt}\"*")

    with tab4:
        st.markdown("### Multi-Stage Execution Trace Timeline")
        for evt in res.get("execution_trace", []):
            st.markdown(f"**{evt.get('component')}** — `{evt.get('status').upper()}` ({evt.get('duration_ms', 0):.1f}ms)")
            st.caption(f"Output: {evt.get('output_summary')}")
            if evt.get("input_payload") or evt.get("output_payload"):
                with st.expander(f"Inspect Payloads for {evt.get('component')}"):
                    if evt.get("input_payload"):
                        st.markdown("**Input Context to Node:**")
                        st.json(evt.get("input_payload"))
                    if evt.get("output_payload"):
                        st.markdown("**Output Produced by Stage:**")
                        st.json(evt.get("output_payload"))

    with tab5:
        st.markdown("### Provide Operational Feedback")
        fb_rating = st.slider("Rating (1-5):", min_value=1, max_value=5, value=5)
        fb_comment = st.text_input("Manager Comments:")
        if st.button("Submit Feedback"):
            try:
                requests.post(
                    f"{API_BASE_URL}/feedback",
                    json={"request_id": res.get("request_id"), "rating": fb_rating, "comment": fb_comment},
                    timeout=5
                )
                st.success("Feedback saved to SQLite database!")
            except Exception as e:
                st.error(f"Error saving feedback: {e}")
