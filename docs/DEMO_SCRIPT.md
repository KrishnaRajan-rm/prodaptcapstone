# Panel Demo Script (10 Minutes)

The format is eight minutes for the working demo and two minutes for questions. Run with `LLM_MODE=live`, `MOCK_LLM=0`, configured provider credentials, and all four agent services available. Do not present a mock response as a live result.

## 0:00-1:00 | Problem and scope

Explain that managers can search historical customer reviews and receive evidence-backed operational findings, priorities, and proposed actions. Clarify that the Yelp review corpus is historical and ends in January 2022.

## 1:00-2:00 | Architecture

Show the architecture diagram supplied for the panel. Point out the separate offline-build and online-query paths, hybrid retrieval, PII boundaries, specialist services, LangGraph routing, CrewAI synthesis, and final Quality Gate. The README also contains a Mermaid version of this flow.

## 2:00-3:00 | Start the application

In separate terminals, start the specialist services, gateway, and React UI using the README commands. Confirm gateway health and agent status before submitting a question.

## 3:00-5:30 | Ask a manager question

Submit this query in the React workspace:

> What recurring service delays appear in this restaurant's reviews, what evidence supports the likely cause, and what should management fix first?

Keep the configured business scope visible. Walk through the returned evidence excerpts and source IDs, then explain the classification, historical trends, root-cause hypotheses, and first ranked action. Read the values produced by this run; do not substitute prepared or invented findings.

## 5:30-6:30 | Quality and confidence

Show the Quality Gate verdict, confidence explanation, and warnings. Explain that confidence is distinct from the verdict: a high score does not erase a separate failed validation check.

## 6:30-7:30 | Execution trace and privacy

Open the trace for the same request. Show measured stages, timings, statuses, evidence IDs, and any actual API error details. Explain the two PII gates and role-based response boundary. If a dependency fails, show the HTTP status and cause instead of presenting fallback findings.

## 7:30-8:00 | Feedback loop

Show how a manager submits feedback for the request and where evaluation/feedback views live. Explain that DeepEval is available as a separate benchmark rather than claiming every interactive request has been benchmarked.

## 8:00-10:00 | Questions

Answer questions about retrieval coverage, historical-data limits, provider/service failures, and the trade-offs documented in [DESIGN.md](DESIGN.md). Avoid claiming historical review data represents current operations.