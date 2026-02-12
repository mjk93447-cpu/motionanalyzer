from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import tempfile
import textwrap
import threading
import time
import traceback
import webbrowser
from pathlib import Path

from streamlit.web import cli as stcli

from motionanalyzer import gui as _gui_mod  # noqa: F401

# Setup logging for packaged exe (for offline Windows exe)
_log_file = Path.home() / "motionanalyzer_gui.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger(__name__)


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
    """Create a temporary Streamlit app script that imports our GUI module."""
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
    # Use user's temp directory for better compatibility with PyInstaller
    try:
        tmp_dir = Path(tempfile.gettempdir()) / "motionanalyzer_gui"
        tmp_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(
            f"Failed to create temp dir in {tempfile.gettempdir()}, using current dir: {e}"
        )
        tmp_dir = Path(".")

    app_path = tmp_dir / "app.py"
    try:
        app_path.write_text(app_text, encoding="utf-8")
        logger.info(f"Created Streamlit app script at {app_path}")
    except Exception as e:
        logger.error(f"Failed to write app script to {app_path}: {e}")
        raise
    return app_path


def main() -> None:
    try:
        parser = argparse.ArgumentParser(description="Launch motionanalyzer GUI app.")
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8501)
        parser.add_argument("--no-browser", action="store_true")
        parser.add_argument(
            "--show-console", action="store_true", help="Show console window for debugging"
        )
        args = parser.parse_args()

        logger.info(f"Starting motionanalyzer GUI on {args.host}:{args.port}")
        logger.info(f"Log file: {_log_file}")

        # Create temp app script
        try:
            app_script = _create_temp_streamlit_app()
            logger.info(f"Created temp app script: {app_script}")
        except Exception as e:
            logger.error(f"Failed to create temp app script: {e}", exc_info=True)
            raise

        url = f"http://{args.host}:{args.port}"

        if not args.no_browser:
            logger.info("Starting browser auto-open thread")
            threading.Thread(
                target=_open_browser_when_ready,
                args=(url, args.host, args.port),
                daemon=True,
            ).start()

        # Disable development mode so server.port is honored (avoid RuntimeError in packaged exe)
        os.environ["STREAMLIT_GLOBAL_DEVELOPMENTMODE"] = "false"
        logger.info("Set STREAMLIT_GLOBAL_DEVELOPMENTMODE=false")

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

        logger.info(f"Launching Streamlit with args: {' '.join(sys.argv[1:])}")
        stcli.main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        error_msg = f"Fatal error: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        print(f"\nERROR: {error_msg}\n", file=sys.stderr)
        print(f"Check log file: {_log_file}", file=sys.stderr)
        input("Press Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
