#!/usr/bin/env python3
"""
Bird's Nest Launcher — macOS .app entry point.

Starts the FastAPI server and opens a native window via pywebview.
Runs as the main process inside the .app bundle.
"""

import os
import sys
import socket
import time
import threading


def get_free_port():
    """Find an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_app_dir():
    """Get the application directory (works both bundled and dev)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))


def get_version():
    """Read version from VERSION file."""
    app_dir = get_app_dir()
    for path in [
        os.path.join(app_dir, "VERSION"),
        os.path.join(app_dir, "..", "Resources", "VERSION"),
        os.path.join(os.path.dirname(__file__), "VERSION"),
    ]:
        if os.path.exists(path):
            return open(path).read().strip()
    return "dev"


def wait_for_server(port, timeout=30):
    """Wait until the server is accepting connections."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.3)
    return False


def run_server(port):
    """Run the uvicorn server."""
    import uvicorn

    uvicorn.run(
        "birdsnest.server:app",
        host="127.0.0.1",
        port=port,
        reload=False,
        log_level="warning",
    )


def main():
    port = get_free_port()
    version = get_version()

    print(f"🪹 Bird's Nest v{version}")
    print(f"   Starting server on port {port}...")

    # Set environment so birdsnest knows it's in app mode
    os.environ["BIRDSNEST_PORT"] = str(port)
    os.environ["BIRDSNEST_VERSION"] = version
    os.environ["BIRDSNEST_APP_MODE"] = "1"

    # Ensure directories exist
    os.makedirs(os.path.expanduser("~/birdsnest_models"), exist_ok=True)
    os.makedirs(os.path.expanduser("~/birdsnest_workspace"), exist_ok=True)

    # Start the server in a background thread
    server_thread = threading.Thread(target=run_server, args=(port,), daemon=True)
    server_thread.start()

    # Wait for server to be ready
    url = f"http://localhost:{port}"
    if not wait_for_server(port):
        print("   ❌ Server failed to start within 30s")
        sys.exit(1)

    print(f"   ✅ Server ready at {url}")

    # Open native window
    import webview

    window = webview.create_window(
        "Bird's Nest",
        url,
        width=1100,
        height=750,
        min_size=(800, 500),
        text_select=True,
    )
    webview.start()  # Blocks until window is closed

    print("   Shutting down Bird's Nest...")
    sys.exit(0)


if __name__ == "__main__":
    main()
