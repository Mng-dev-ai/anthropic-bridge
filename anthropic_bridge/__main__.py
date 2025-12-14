import argparse
import os

import uvicorn

from .server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Anthropic Bridge Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to run on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")

    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable required")
        raise SystemExit(1)

    app = create_app(openrouter_api_key=api_key)

    print(f"Starting Anthropic Bridge on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
