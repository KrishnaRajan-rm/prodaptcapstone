"""Realistic demonstration scenarios grounded in Yelp Open Dataset restaurant profiles."""

from typing import List
from app.api.schemas import ScenarioItem


DEMO_SCENARIOS: List[ScenarioItem] = [
    ScenarioItem(
        id="scenario-1",
        title="Emerging Dining Delays",
        description="Multi-agent investigation of wait time complaints, kitchen bottlenecks, and table turnover across busy dinner shifts.",
        sample_query="Customers are repeatedly complaining about slow service and long wait times. What operational issues are causing this, and what should we fix first?",
        mode="auto",
        business_id="_ab50qdWOk0DdB6XOrBitw",
        business_name="Acme Oyster House",
        city="New Orleans",
        suggested_start_date="2020-01",
        suggested_end_date="2022-01",
    ),
    ScenarioItem(
        id="scenario-2",
        title="Food Quality & Temperature Audit",
        description="Root cause and action planning for recurring food quality, lukewarm food, and order inaccuracy complaints.",
        sample_query="Why are guests reporting that food is arriving lukewarm or incorrectly prepared, and what kitchen procedures should be implemented?",
        mode="auto",
        business_id="iSRTaT9WngzB8JJ2YKJUig",
        business_name="Mother's Restaurant",
        city="New Orleans",
        suggested_start_date="2020-01",
        suggested_end_date="2022-01",
    ),
    ScenarioItem(
        id="scenario-3",
        title="Rush Hour Line & Queue Management",
        description="Examines customer friction during peak tourist and weekend rush lines, analyzing staff pacing and order flow.",
        sample_query="Analyze line pacing and queue management problems during peak weekend rushes, and suggest operational remedies.",
        mode="auto",
        business_id="PP3BBaVxZLcJU54uP_wL6Q",
        business_name="Pat's King of Steaks",
        city="Philadelphia",
        suggested_start_date="2020-01",
        suggested_end_date="2022-01",
    ),
    ScenarioItem(
        id="scenario-4",
        title="Staff Courtesy & Service Recovery",
        description="Investigates host stand interactions, staff attentiveness, and front-of-house customer service sentiment.",
        sample_query="Investigate complaints regarding front-of-house staff attentiveness and host greeting delays. Recommend training priorities.",
        mode="auto",
        business_id="ac1AeYqs8Z4_e2X5M3if2A",
        business_name="Oceana Grill",
        city="New Orleans",
        suggested_start_date="2020-01",
        suggested_end_date="2022-01",
    ),
    ScenarioItem(
        id="scenario-5",
        title="Increasing Complaint Trends",
        description="Historical trend analysis detecting worsening complaint categories over time with historical slope calculations.",
        sample_query="Which complaint categories are worsening over time, and what is the historical trajectory of wait time versus cleanliness?",
        mode="auto",
        business_id="GXFMD0Z4jEVZBCsbPf4CTQ",
        business_name="Hattie B's Hot Chicken",
        city="Nashville",
        suggested_start_date="2020-01",
        suggested_end_date="2022-01",
    ),
    ScenarioItem(
        id="scenario-6",
        title="General Cross-Location Synthesis",
        description="Broad operational synthesis analyzing industry-wide friction points without a specific restaurant restriction.",
        sample_query="Across all monitored restaurants, what are the top three operational drivers of 1-star reviews and how can managers address them?",
        mode="auto",
        business_id=None,
        business_name="All Monitored Locations",
        city=None,
        suggested_start_date="2020-01",
        suggested_end_date="2022-01",
    ),
    ScenarioItem(
        id="scenario-7",
        title="Focused Complaint Classification",
        description="Direct invocation of the Classification Specialist Agent to categorize complaint aspects, sentiment, and severity.",
        sample_query="Classify and extract key complaint aspects: 'We waited 45 minutes for our table, the waiter was dismissive, and the fish was dry.'",
        mode="agent",
        agent="classification",
        business_id=None,
        business_name=None,
    ),
    ScenarioItem(
        id="scenario-8",
        title="Focused Action Planning",
        description="Direct invocation of the Action Planner Specialist Agent to generate ranked actions with designated owner roles and KPIs.",
        sample_query="Generate a ranked operational action plan addressing kitchen bottleneck delays during dinner rush.",
        mode="agent",
        agent="action_planner",
        business_id="_ab50qdWOk0DdB6XOrBitw",
        business_name="Acme Oyster House",
    ),
]


def get_demo_scenarios() -> List[ScenarioItem]:
    return DEMO_SCENARIOS
