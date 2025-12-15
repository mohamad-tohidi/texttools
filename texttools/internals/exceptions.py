class TextToolsError(Exception):
    """Base exception for all TextTools errors."""

    pass


class PromptError(TextToolsError):
    """Errors related to prompt loading and formatting."""

    pass


class LLMError(TextToolsError):
    """Errors from LLM API calls."""

    pass


class ValidationError(TextToolsError):
    """Errors from output validation."""

    pass


class ConfigurationError(TextToolsError):
    """Errors from misconfiguration."""

    pass
