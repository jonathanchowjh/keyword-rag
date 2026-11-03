"""Starting points for the connection form. Provider names live here and in wire.py."""

KINDS = ("anthropic", "openai")

# models: model id -> (label, USD per million input tokens, USD per million output tokens)
PRESETS: dict[str, dict] = {
    "anthropic": {
        "label": "Anthropic",
        "kind": "anthropic",
        "base_url": None,
        "models": {
            "claude-haiku-5-5": ("Haiku 5.5", 0.10, 0.50),
            "claude-sonnet-5-5": ("Sonnet 5.5", 2.00, 10.00),
        },
        "default_model": "claude-haiku-5-5",
    },
    "openai": {
        "label": "OpenAI",
        "kind": "openai",
        "base_url": "https://api.openai.com/v1",
        "models": {},
        "default_model": None,
    },
    "gemini": {
        "label": "Gemini",
        "kind": "openai",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": {},
        "default_model": None,
    },
    "xai": {
        "label": "xAI",
        "kind": "openai",
        "base_url": "https://api.x.ai/v1",
        "models": {},
        "default_model": None,
    },
    "openrouter": {
        "label": "OpenRouter",
        "kind": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "models": {},
        "default_model": None,
    },
    "custom": {
        "label": "Custom",
        "kind": "openai",
        "base_url": None,
        "models": {},
        "default_model": None,
    },
}
