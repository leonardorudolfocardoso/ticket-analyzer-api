from ticket_analyzer.config import settings
from ticket_analyzer.llm import LLMProvider
from ticket_analyzer.llm.openai import OpenAIProvider


def get_provider() -> LLMProvider:
    """
    Instantiate and return the configured LLM provider.

    The provider is selected via the LLM_PROVIDER environment variable.
    To add a new provider, implement the LLMProvider protocol and add
    a branch here.

    Raises:
        ValueError: If LLM_PROVIDER is set to an unrecognised value.
    """
    if settings.llm_provider == "openai":
        return OpenAIProvider()
    raise ValueError(f"Unknown LLM provider: '{settings.llm_provider}'")
