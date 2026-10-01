class DomainException(Exception):  # noqa: N818 - established public base name
    """Base exception for all domain errors."""

    def __init__(self, message: str, code: str = "domain_error"):
        super().__init__(message)
        self.message = message
        self.code = code


class ValidationError(DomainException, ValueError):
    """Raised when incoming requests or parameters fail validation.

    Also inherits ValueError so that callers treating domain validation as a
    plain ValueError (and existing tests) keep working.
    """

    def __init__(self, message: str):
        super().__init__(message, code="validation_error")


class AuthenticationError(DomainException):
    """Raised when authentication credentials are missing or rejected."""

    def __init__(self, message: str):
        super().__init__(message, code="authentication_error")


class AuthorizationError(DomainException):
    """Raised when the caller is authenticated but not permitted."""

    def __init__(self, message: str = "Invalid or missing API key"):
        super().__init__(message, code="invalid_api_key")


class TokenExpiredError(AuthenticationError):
    """Raised when tokens cannot be refreshed."""

    def __init__(self, message: str = "OAuth token has expired and refresh failed"):
        super().__init__(message)
        self.code = "token_expired"


class UpstreamServiceError(DomainException):
    """Raised when upstream API fails."""

    def __init__(
        self,
        message: str,
        status_code: int = 502,
        details: str = "",
        retryable: bool = False,
    ):
        super().__init__(message, code="upstream_service_error")
        self.status_code = status_code
        self.details = details
        self.retryable = retryable


class UpstreamTimeoutError(UpstreamServiceError):
    """Raised when the upstream does not respond within the configured budget."""

    def __init__(self, message: str, timeout: float):
        super().__init__(message, status_code=504, details="", retryable=True)
        self.timeout = timeout


class UpstreamRejectedError(UpstreamServiceError):
    """The upstream refused the request because the payload was invalid.

    A 4xx from upstream is the *caller's* fault, not ours: a malformed tool
    schema, an unknown field, an oversized request. Reporting it as 502 would
    both mislead the client and invite pointless retries, so the upstream status
    is surfaced instead.
    """

    def __init__(self, message: str, upstream_status: int, details: str = ""):
        super().__init__(message, status_code=upstream_status, details=details, retryable=False)
        self.upstream_status = upstream_status


class ModelNotFoundError(DomainException):
    """Raised when requested model does not exist."""

    def __init__(self, model_id: str):
        super().__init__(f"Model '{model_id}' not found", code="model_not_found")
        self.model_id = model_id


class ConfigurationError(DomainException):
    """Raised when required configuration is absent or invalid."""

    def __init__(self, message: str):
        super().__init__(message, code="configuration_error")
