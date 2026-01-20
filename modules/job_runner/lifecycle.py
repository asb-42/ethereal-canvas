import signal
import sys

def register_shutdown(handler):
    def _handle(sig, frame):
        handler()
        sys.exit(0)
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)