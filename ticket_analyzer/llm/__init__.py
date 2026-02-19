from typing import Protocol, runtime_checkable

from ticket_analyzer.config import settings


@runtime_checkable
class LLMProvider(Protocol):
    """
    Protocol that all LLM provider implementations must satisfy.

    To add a new provider, create a class that implements `classify`
    and register it in `get_provider` below.
    """

    async def classify(self, system_prompt: str, user_text: str) -> str:
        """
        Send a prompt and ticket text to the LLM and return the raw response.

        Args:
            system_prompt: Instructions that define the classification task.
            user_text: The raw support ticket content to classify.

        Returns:
            A raw JSON string expected to conform to the TicketAnalysis schema.

        Raises:
            ValueError: If the provider returns an empty response.
        """
        ...


def get_provider() -> LLMProvider:
    """
    Instantiate and return the configured LLM provider.

    The provider is selected via the LLM_PROVIDER environment variable.
    To add a new provider, implement the LLMProvider protocol and add
    a branch here.

    Raises:
        ValueError: If LLM_PROVIDER is set to an unrecognised value.
    """
    from ticket_analyzer.llm.openai import OpenAIProvider

    if settings.llm_provider == "openai":
        return OpenAIProvider()
    raise ValueError(f"Unknown LLM provider: '{settings.llm_provider}'")
