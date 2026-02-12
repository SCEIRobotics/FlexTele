from pathlib import Path

import yaml

from pico import PicoServer

if __name__ == '__main__':
    project_dir = Path(__file__).parents[1]

    config = yaml.safe_load((project_dir / "config" / "pico_server.yaml").read_text())
    pico_server= PicoServer(config)
    pico_server.start()