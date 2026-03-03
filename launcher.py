#!/usr/bin/env python3
"""
Bird's Nest Launcher — macOS .app entry point.

Starts the FastAPI server on a free port and opens the browser.
Runs as the main process inside the .app bundle.
"""

import os
import sys
import socket
import signal
import subprocess
import webbrowser
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
        # Running as PyInstaller bundle
        return os.path.dirname(sys.executable)
    else:
        # Running from source
        return os.path.dirname(os.path.abspath(__file__))


def get_version():
    """Read version from VERSION file."""
    app_dir = get_app_dir()
    # Check a few locations
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


def main():
    port = get_free_port()
    version = get_version()

    print(f"🪹 Bird's Nest v{version}")
    print(f"   Starting server on port {port}...")

    # Set environment so birdsnest knows it's in app mode
    os.environ["BIRDSNEST_PORT"] = str(port)
    os.environ["BIRDSNEST_VERSION"] = version
    os.environ["BIRDSNEST_APP_MODE"] = "1"

    # Ensure models directory exists
    models_dir = os.path.expanduser("~/birdsnest_models")
    os.makedirs(models_dir, exist_ok=True)

    # Ensure workspace directory exists
    workspace_dir = os.path.expanduser("~/birdsnest_workspace")
    os.makedirs(workspace_dir, exist_ok=True)

    # Start the server in a thread
    server_thread = threading.Thread(target=run_server, args=(port,), daemon=True)
    server_thread.start()

    # Wait for server to be ready, then open browser
    if wait_for_server(port):
        url = f"http://localhost:{port}"
        print(f"   ✅ Server ready at {url}")
        webbrowser.open(url)
    else:
        print("   ❌ Server failed to start within 30s")
        sys.exit(1)

    # Keep main thread alive — handle Ctrl+C / app quit
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("\n   Shutting down Bird's Nest...")
        sys.exit(0)


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


if __name__ == "__main__":
    main()
