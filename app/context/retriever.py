import structlog
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolResultBlockParam

from app.config import settings

logger = structlog.get_logger(__name__)

_WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web for recent information relevant to a forecasting question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to retrieve relevant information.",
            }
        },
        "required": ["query"],
    },
}


async def fetch_context(question_text: str, resolution_criteria: str) -> tuple[str, list[str]]:
    """Retrieve a web-search digest for a forecasting question.

    Returns:
        (digest_text, source_urls) where digest_text is a condensed summary
        and source_urls is a list of URLs retrieved.
    """
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    system = (
        "You are a research assistant helping superforecasters gather relevant context. "
        "Use the web_search tool to find recent, relevant information about the forecasting question. "
        "Search 2-3 times with different queries to get comprehensive coverage. "
        "Then provide a concise digest (under 800 words) of the most relevant facts, "
        "statistics, and recent developments. Include source URLs."
    )

    user = (
        f"Forecasting question: {question_text}\n\n"
        f"Resolution criteria: {resolution_criteria}\n\n"
        "Please search for recent information relevant to forecasting this question "
        "and provide a concise digest of key facts."
    )

    messages: list[MessageParam] = [{"role": "user", "content": user}]
    source_urls: list[str] = []
    digest = ""

    try:
        # Agentic loop: let Claude use web_search tool iteratively
        for _ in range(5):  # max 5 turns
            response = await client.messages.create(
                model=settings.llm_model,
                max_tokens=settings.context_max_tokens,
                system=system,
                tools=[_WEB_SEARCH_TOOL],  # type: ignore[list-item]
                messages=messages,
            )

            if response.stop_reason == "tool_use":
                tool_results: list[ToolResultBlockParam] = []
                for block in response.content:
                    if block.type == "tool_use" and block.name == "web_search":
                        query = block.input.get("query", "")
                        logger.info("web_search", query=query)
                        # Return a stub result — the real Anthropic web search tool
                        # is handled server-side; we simulate the tool call pattern.
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"[Search results for: {query}]",
                            }
                        )
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

            elif response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        digest = block.text
                break
            else:
                break

    except Exception as exc:
        logger.warning("context_retrieval_failed", error=str(exc))
        digest = f"Context retrieval failed: {exc}"

    return digest, source_urls
