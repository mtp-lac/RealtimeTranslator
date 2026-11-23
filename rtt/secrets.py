"""Windows Credential Manager storage for provider API keys via keyring."""

SERVICE_NAME = "RealtimeTranslator"


def get_api_key(provider: str) -> str:
    """Return the stored API key for `provider`, or empty string on error/missing."""
    if not provider:
        return ""
    try:
        import keyring

        return keyring.get_password(SERVICE_NAME, provider) or ""
    except Exception:
        return ""


def set_api_key(provider: str, key: str) -> bool:
    """Save or remove an API key in Windows Credential Manager."""
    if not provider:
        return False
    try:
        import keyring

        cleaned = (key or "").strip()
        if cleaned:
            keyring.set_password(SERVICE_NAME, provider, cleaned)
        else:
            try:
                keyring.delete_password(SERVICE_NAME, provider)
            except Exception:
                pass
        return True
    except Exception:
        return False
