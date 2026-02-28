#!/usr/bin/env python3
"""
Simple HTTP server for the frontend

Usage:
    python serve.py
    python serve.py --port 3000
"""

import argparse
import http.server
import socketserver
import webbrowser
from pathlib import Path


def serve(port=3000, open_browser=True):
    """Start HTTP server and optionally open browser."""

    # Change to frontend directory
    frontend_dir = Path(__file__).parent

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(frontend_dir), **kwargs)

        def end_headers(self):
            # Add CORS headers for local development
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()

    with socketserver.TCPServer(("", port), Handler) as httpd:
        url = f"http://localhost:{port}"

        print("=" * 60)
        print("Zuora Help Agent - Frontend Server")
        print("=" * 60)
        print(f"\n✅ Server running at: {url}")
        print(f"\n📁 Serving: {frontend_dir}")
        print(f"\n🔧 Backend should be running at: http://localhost:8000")
        print(f"\nPress Ctrl+C to stop the server")
        print("=" * 60)

        if open_browser:
            print(f"\nOpening browser...")
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve Zuora Help Agent frontend")

    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Port to serve on (default: 3000)"
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )

    args = parser.parse_args()

    serve(port=args.port, open_browser=not args.no_browser)
