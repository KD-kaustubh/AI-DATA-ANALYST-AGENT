"""Errors raised while loading and validating datasets."""


class DatasetError(Exception):
    """Base class for every dataset loading or validation failure."""


class DatasetNotFoundError(DatasetError):
    """The given path does not point to an existing file."""


class UnsupportedFileTypeError(DatasetError):
    """The file extension is not one we know how to read."""


class EmptyDatasetError(DatasetError):
    """The file exists but holds no usable rows."""


class DatasetReadError(DatasetError):
    """The file could not be parsed, usually because it is corrupted."""


class AnalysisError(Exception):
    """Base class for analysis and chart failures."""


class ColumnNotFoundError(AnalysisError):
    """A requested column is not in the dataset."""


class InvalidOperationError(AnalysisError):
    """The requested operation does not apply to this data."""


class LLMError(Exception):
    """Base class for language-model failures."""


class LLMConfigurationError(LLMError):
    """The model client is missing configuration, such as an API key."""


class LLMProviderError(LLMError):
    """The provider rejected the request or could not be reached."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        # The HTTP status the provider returned, when the SDK exposes one.
        # Lets a caller (e.g. the API layer) tell a rate limit (429) apart
        # from a genuine outage without parsing the message text.
        self.status_code = status_code


class LLMResponseError(LLMError):
    """The model replied with something we could not parse."""


class ToolError(Exception):
    """Base class for tool selection and dispatch failures."""


class UnknownToolError(ToolError):
    """A tool was requested that is not in the registry."""


class InvalidToolArgumentsError(ToolError):
    """A tool was called with missing, unknown or badly typed arguments."""
