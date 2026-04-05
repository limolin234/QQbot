from __future__ import annotations

import uvicorn


def main() -> None:
    """Run the local Config Studio API server."""
    uvicorn.run(
        "tools.config_studio.server.app:app", host="127.0.0.1", port=8787, reload=True
    )


if __name__ == "__main__":
    main()
