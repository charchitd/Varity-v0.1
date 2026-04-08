class VarityError(Exception):
    """Base exception for Varity."""
    pass

class ProviderError(VarityError):
    """Exception raised for errors in the provider."""
    pass

class DecompositionError(VarityError):
    """Exception raised for errors during claim decomposition."""
    pass

class VerificationError(VarityError):
    """Exception raised for errors during claim verification."""
    pass

class ConfigError(VarityError):
    """Exception raised for configuration errors."""
    pass
