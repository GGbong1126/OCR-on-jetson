"""Original HTTP routes/Gunicorn, connected to externally supervised project children."""
import json
from pathlib import Path
import sys
from settings import S,SOURCE
sys.path[:0]=[str(SOURCE/'app'),str(SOURCE/'ipc')]
from serve_http import Supervisor,create_app as original_app


class Process:
    def __init__(self,pid): self.pid=pid
    def poll(self):
        try:
            state=(Path('/proc')/str(self.pid)/'stat').read_text().rsplit(')',1)[1].split()[0]
            return 1 if state=='Z' else None
        except FileNotFoundError: return 1


class External(Supervisor):
    def __init__(self):
        config=json.loads((S/'trace/server.json').read_text())
        self.root=S/'trace/runtime'
        self.backend=Process(config['backend_pid']); self.cpp=Process(config['cpp_pid'])


def create_app(): return original_app(External())
