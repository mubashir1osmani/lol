import logging
import signal
import threading

from .api import fetch_releases, post_release
from .config import get_config, load_dotenv
from .poller import poll_once
from .state import load_state, save_state


def main() -> None:
    load_dotenv()
    config = get_config()
    stop_event = threading.Event()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    def stop(_signal_number: int, _frame: object) -> None:
        logging.info("Shutting down")
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    logging.info("Watching GitHub releases for %s", config.github_repository)

    while not stop_event.is_set():
        try:
            poll_once(config, fetch_releases, post_release, load_state, save_state)
        except Exception:
            logging.exception("Release check failed")
        stop_event.wait(config.poll_interval_seconds)
