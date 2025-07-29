"""LLM integration and prompt engineering.

This module handles integration with large language models for mathematical
formalization, including prompt engineering and response processing.

Modules:
    base: Base LLM interface
    openai_client: OpenAI API integration
    anthropic_client: Anthropic Claude API integration
    huggingface_client: Hugging Face model integration
    prompt_templates: Mathematical formalization prompts
    response_parser: LLM response parsing and validation
"""

__all__ = [
    "BaseLLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "HuggingFaceClient",
    "PromptTemplates",
    "ResponseParser",
]