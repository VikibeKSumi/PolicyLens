from groq import Groq
from ..state import ResponseState

class QueryRewriter():

    def __init__(self, api_key: str, model_name: str):
        self.client = Groq(api_key=api_key)
        self.model = model_name

    def rewrite(self, state: ResponseState) -> str:
        query = state.get("query")
        rewritten_query = state.get("rewritten_query")
        top_relevancy_score = state.get("top_relevancy_score")
        rewrite_retry_count = state.get("rewrite_retry_count")

        messages=[
            {"role": "system", "content": (
                "You are a query rewriter for an Indian government policy document search system."
                "Rewrite the user's query to use formal Indian budget and policy terminology."
                "Key mappings: FY27 → 2026-27, FY26 → 2025-26, FY25 → 2024-25, FY24 → 2023-24."
                "Budget year references should use BE (Budget Estimates) or RE (Revised Estimates) format."
                "If the query already uses correct formal terminology (BE/RE + year format), return it exactly as given."
                "Return only the rewritten query. Nothing else."

                "After reranking the rewritten query will be send back if it does not cross the required threshold."
                f"here is the previous rewritten query {rewritten_query}"
                f"and here is the score is score {top_relevancy_score}"
                f"attempt to rewrite the query from previous rewritten_query to increase the score further"
                f"if {rewrite_retry_count} is 0 it is first try and 'rewritten_query' and 'top_relevancy_score' will be 'None'."
            )},
            {"role": "user", "content": "What is FY27 total spending?"},
            {"role": "assistant", "content": "What is the total expenditure in BE 2026-27?"},
            {"role": "user", "content": "fiscal gap for next year budget"},
            {"role": "assistant", "content": "What is the fiscal deficit in BE 2026-27?"},
            {"role": "user", "content": "how much did india spend on capex last year"},
            {"role": "assistant", "content": "What is the capital expenditure in RE 2025-26?"},
            {"role": "user", "content": query}
        ]


        if rewrite_retry_count == 0:
            temperature = 0
        else:
            temperature = 0.5

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature
        )

        return {"rewritten_query": response.choices[0].message.content.strip(),
                "rewrite_retry_count": 1}