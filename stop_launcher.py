from __future__ import annotations

import subprocess
import sys


APP_PORT = 8501


def stop_streamlit_port(port: int = APP_PORT) -> None:
    command = (
        f"$conns = Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue; "
        "if ($conns) { "
        "$pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique; "
        "foreach ($pidValue in $pids) { "
        "Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue "
        "} "
        "}"
    )

    creation_flags = 0
    if sys.platform.startswith("win"):
        creation_flags = subprocess.CREATE_NO_WINDOW

    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=creation_flags,
        check=False,
    )


def main() -> None:
    stop_streamlit_port()


if __name__ == "__main__":
    main()
