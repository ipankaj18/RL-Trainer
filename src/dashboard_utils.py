import os
import sys
import os
import sys
import threading
import time
import subprocess
from pathlib import Path
from IPython.display import IFrame, display

def launch_dashboard(port=8000, market="NQ", height=800, base_url=None):
    """
    Launch the Cyberpunk Dashboard in a background thread and display it in Jupyter.
    
    Args:
        port: Port to run the server on.
        market: Market symbol to monitor.
        height: Height of the IFrame in pixels.
        base_url: Optional public URL (e.g., "https://my-pod-8000.runpod.net"). 
                  If None, defaults to localhost but suggests /proxy/port/.
    """
    # Paths
    # Paths
    # Use the location of this script to find the project root
    # src/dashboard_utils.py -> parent is src -> parent is project root
    base_dir = Path(__file__).parent.parent
    dashboard_dir = base_dir / "@dashboard"
    server_script = dashboard_dir / "server.py"
    
    # Check if dashboard exists
    if not dashboard_dir.exists():
        print("❌ Dashboard directory '@dashboard' not found.")
        return

    # Check if build exists
    dist_dir = dashboard_dir / "dist"
    if not dist_dir.exists():
        print("⚠️ Dashboard frontend not built. Attempting to build...")
        try:
            subprocess.run("npm install && npm run build", shell=True, cwd=dashboard_dir, check=True)
            print("✅ Build successful!")
        except Exception as e:
            print(f"❌ Build failed: {e}")
            print("Running in API-only mode (no UI, just JSON).")

    # Define server function
    def run_server():
        # Check if port is already in use (simple check)
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) == 0:
                print(f"⚠️ Port {port} is already in use. Assuming server is running.")
                return

        cmd = [sys.executable, str(server_script), "--port", str(port), "--market", market]
        subprocess.run(cmd)

    # Start server in thread
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    # Wait a bit for server to start
    time.sleep(2)
    
    # Determine URL
    if base_url:
        url = base_url
    else:
        # Try to guess or provide helpful default
        url = f"http://localhost:{port}"
        print(f"🚀 Dashboard server running locally on port {port}.")
        print(f"ℹ️  If you are on a remote server (Runpod/Colab), 'localhost' won't work.")
        print(f"    Try using the Jupyter proxy URL: launch_dashboard(..., base_url='/proxy/{port}/')")
    
    print(f"🔗 Opening Dashboard at: {url}")
    display(IFrame(src=url, width="100%", height=height))
