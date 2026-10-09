import http.server
import socketserver
import os
import sys

PORT = 8000
DIRECTORY = "/Users/dhanavasanth/Downloads/sales_analytics_project"

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def start_server():
    os.chdir(DIRECTORY)
    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        print(f"🚀 Telecom ML Web Application server running at http://localhost:{PORT}")
        sys.stdout.flush()
        httpd.serve_forever()

if __name__ == "__main__":
    start_server()
