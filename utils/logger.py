import os
from datetime import datetime


class Logger:
    """Simple logger that writes to both file and stdout."""

    def __init__(self, log_dir="./logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"train_{timestamp}.log")
        self.file = open(self.log_path, "w")

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        self.file.write(line + "\n")
        self.file.flush()

    def close(self):
        self.file.close()
