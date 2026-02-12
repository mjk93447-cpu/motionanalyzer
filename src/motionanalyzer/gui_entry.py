from __future__ import annotations

import argparse
import os
import socket
import sys
import tempfile
import textwrap
import threading
import time
import webbrowser
from pathlib import Path

from streamlit.web import cli as stcli


def _open_browser_when_ready(url: str, host: str, port: int, timeout_sec: float = 20.0) -> None:
    start = time.time()
    while time.time() - start < timeout_sec:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.4)
            if sock.connect_ex((host, port)) == 0:
                webbrowser.open_new_tab(url)
                return
        time.sleep(0.25)


def _create_temp_streamlit_app() -> Path:
    app_text = (
        textwrap.dedent(
            """
        from motionanalyzer.gui import main

        if __name__ == "__main__":
            main()
        """
        ).strip()
        + "\n"
    )
    tmp_dir = Path(tempfile.mkdtemp(prefix="motionanalyzer_gui_"))
    app_path = tmp_dir / "app.py"
    app_path.write_text(app_text, encoding="utf-8")
    return app_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch motionanalyzer GUI app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    app_script = _create_temp_streamlit_app()
    url = f"http://{args.host}:{args.port}"

    if not args.no_browser:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(url, args.host, args.port),
            daemon=True,
        ).start()

    # Disable development mode so server.port is honored (avoid RuntimeError in packaged exe)
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENTMODE"] = "false"

    sys.argv = [
        "streamlit",
        "run",
        str(app_script),
        "--global.developmentMode",
        "false",
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--browser.gatherUsageStats",
        "false",
    ]
    stcli.main()


if __name__ == "__main__":
    main()
