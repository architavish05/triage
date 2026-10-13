"""Exceptions for Earth observation and geospatial data fetching."""


class TriageFetchError(Exception):
    """Base exception for all fetch-related failures."""
    pass


class CredentialsError(TriageFetchError):
    """Raised when required service credentials are missing or invalid in .env."""
    pass


class SceneNotFoundError(TriageFetchError):
    """Raised when required satellite imagery matching criteria is not found."""
    pass


class CloudCoverExceededError(SceneNotFoundError):
    """Raised when all available scenes exceed acceptable cloud thresholds."""
    pass


class OrbitMismatchError(TriageFetchError):
    """Raised when pre- and post-event SAR scenes have mismatched relative orbit tracks."""
    pass


class OhsomeAPIError(TriageFetchError):
    """Raised when the ohsome API request fails or returns an error."""
    pass


class DateConstraintError(TriageFetchError):
    """Raised when a pre-event snapshot violates hard date constraints (must be <= 2026-07-27)."""
    pass
