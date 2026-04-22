from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


APP_URL = "http://127.0.0.1:8501"


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def streamlit_is_running(url: str = APP_URL, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= getattr(response, "status", 0) < 500
    except Exception:
        return False


def launch_streamlit(root: Path) -> None:
    pythonw = root / ".venv" / "Scripts" / "pythonw.exe"
    app_path = root / "app.py"
    if not pythonw.exists():
        raise FileNotFoundError(f"pythonw.exe tidak ditemukan: {pythonw}")
    if not app_path.exists():
        raise FileNotFoundError(f"app.py tidak ditemukan: {app_path}")

    creation_flags = 0
    if sys.platform.startswith("win"):
        creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    subprocess.Popen(
        [
            str(pythonw),
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8501",
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=creation_flags,
        close_fds=True,
    )


def wait_until_ready(url: str = APP_URL, attempts: int = 40, delay: float = 0.5) -> None:
    for _ in range(attempts):
        if streamlit_is_running(url):
            return
        time.sleep(delay)


def main() -> None:
    root = project_root()
    if not streamlit_is_running():
        launch_streamlit(root)
        wait_until_ready()
    webbrowser.open(APP_URL)


if __name__ == "__main__":
    main()
