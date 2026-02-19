from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """
    Protocol that all LLM provider implementations must satisfy.

    To add a new provider, create a class that implements `classify`
    and register it in `llm/factory.py`.
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
