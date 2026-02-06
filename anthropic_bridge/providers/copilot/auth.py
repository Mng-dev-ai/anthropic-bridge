import os


def get_copilot_token() -> str | None:
    return os.environ.get("GITHUB_COPILOT_TOKEN")
