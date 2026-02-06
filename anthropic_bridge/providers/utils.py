def map_reasoning_effort(
    budget_tokens: int | None, model_id: str | None = None
) -> str | None:
    if not budget_tokens:
        return None

    if budget_tokens >= 32000:
        if _model_supports_xhigh(model_id):
            return "xhigh"
        return "high"
    elif budget_tokens >= 15000:
        return "high"
    elif budget_tokens >= 10000:
        return "medium"
    elif budget_tokens >= 1:
        return "low"

    return None


def _model_supports_xhigh(model_id: str | None) -> bool:
    if not model_id:
        return False

    lower = model_id.lower()
    if "/" in lower:
        lower = lower.split("/", 1)[1]

    # OpenAI docs: xhigh is supported for gpt-5.1-codex-max and for models
    # after gpt-5.1-codex-max (e.g., gpt-5.2, gpt-5.3 and their codex variants).
    return lower.startswith(("gpt-5.1-codex-max", "gpt-5.2", "gpt-5.3"))
