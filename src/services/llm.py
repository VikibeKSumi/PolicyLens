import time

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from groq import RateLimitError, APIConnectionError
from llama_index.llms.groq import Groq
from llama_index.core.schema import MetadataMode

from ..state import ResponseState

class LLMService:

    def __init__(self, llm_model: str, api_key: str):
        self.groq_llm = Groq(
            model=llm_model,
            api_key=api_key
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        reraise=True
    )
    def  generate_response(self, state: ResponseState):
        """Returns answer + full token metrics (final version)."""
        
        rewritten_query = state.get("rewritten_query")
        compressed_nodes = state.get("compressed_nodes")
        context_text = "\n\n".join([
            node.node.get_content(metadata_mode=MetadataMode.NONE)
            for node in compressed_nodes
        ])

        prompt = (
            f"Context Information:\n{context_text}\n\n"
            f"Query: {rewritten_query}\n\n"
            "As an Advisor, provide a concise, grounded answer based strictly on the context. "
            "If the information is not present, admit it. Match the query's language."
            "Always respond in English regardless of the language of the source documents. "

        )

        t0 = time.perf_counter()
        response = self.groq_llm.complete(prompt)
        t1 = round(time.perf_counter() - t0, 2)

        # Extract token usage from Groq (LlamaIndex 2026 format)
        usage = getattr(response.raw, 'usage', None) if hasattr(response, 'raw') else None
        input_tokens = usage.prompt_tokens if usage else len(prompt.split())
        output_tokens = usage.completion_tokens if usage else len(response.text.split())

        # Reliable TPS calculation (fixed for LlamaIndex + Groq 2026)
        tps = output_tokens / t1 if t1 > 0 else 0


        return {
            "answer": response.text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "tokens_per_second": round(tps, 2),
            "generation_time": t1
        }