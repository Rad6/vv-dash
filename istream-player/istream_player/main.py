import asyncio
import json
import logging
import sys
import traceback
from typing import Dict, List

import yaml

from istream_player.config.config import PlayerConfig
from istream_player.core.module_composer import PlayerComposer
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter


def setup_logging(log_level: str):

    lexer = get_lexer_by_name("pytb" if sys.version_info.major < 3 else "py3tb")
    formatter = TerminalFormatter()

    def myexcepthook(type, value, tb):
        tbtext = ''.join(traceback.format_exception(type, value, tb))
        sys.stderr.write(highlight(tbtext, lexer, formatter))

    sys.excepthook = myexcepthook

    class CustomFormatter(logging.Formatter):
        grey = "\x1b[38;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"
        format_str = "%(asctime)s %(name)20s %(levelname)8s:\t%(message)s"

        FORMATS = {
            logging.DEBUG: grey + format_str + reset,
            logging.INFO: grey + format_str + reset,
            logging.WARNING: yellow + format_str + reset,
            logging.ERROR: red + format_str + reset,
            logging.CRITICAL: bold_red + format_str + reset
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)
        
    class InfoFilter(logging.Filter):
        def filter(self, rec):
            return rec.levelno in (logging.DEBUG, logging.INFO)

    h1 = logging.StreamHandler(sys.stdout)
    h1.setLevel(logging.DEBUG)
    h1.addFilter(InfoFilter())
    h1.setFormatter(CustomFormatter())
    h2 = logging.StreamHandler()
    h2.setLevel(logging.WARNING)
    h2.setFormatter(CustomFormatter())

    logging.basicConfig(
        level=getattr(logging, log_level), handlers=[h1, h2]
    )


def load_from_dict(d: Dict, config: PlayerConfig):
    for k, v in d.items():
        if k.startswith("_"):
            continue
        try:
            if isinstance(v, List):
                prev_list = config.__getattribute__(k)
                if prev_list is None:
                    prev_list = []
                    config.__setattr__(k, prev_list)
                prev_list.extend(v)
            elif isinstance(v, dict):
                prev_d = config.__getattribute__(k)
                if prev_d is None:
                    prev_d = {}
                    config.__setattr__(k, prev_d)
                prev_d.update(v)
            elif v is not None:
                config.__setattr__(k, v)
        except Exception:
            raise Exception(f"Failed to load config key '{k}' = {v}")
    return config


def load_from_config_file(config_path: str, config: PlayerConfig):
    extension = config_path[config_path.rindex("."):].lower()
    if extension in (".yaml", ".yml"):
        with open(config_path) as f:
            return load_from_dict(yaml.safe_load(f), config)
    elif extension == ".json":
        with open(config_path) as f:
            return load_from_dict(json.load(f), config)
    else:
        raise Exception(f"Config file format not supported. Use JSON or YAML. Used : {config_path}")


def main():
    try:
        assert sys.version_info.major >= 3 and sys.version_info.minor >= 3
    except AssertionError:
        print("Python 3.3+ is required.")
        exit(-1)

    composer = PlayerComposer()
    composer.register_core_modules()
    parser = composer.create_arg_parser()
    args = vars(parser.parse_args())

    # Load default values
    config = PlayerConfig()

    # First load from config file
    if args["config"] is not None:
        load_from_config_file(args["config"], config)
        del args["config"]

    # Then override from arguments
    load_from_dict(args, config)

    setup_logging(config.log_level.upper())

    config.validate()

    asyncio.run(composer.run(config))


if __name__ == "__main__":
    main()
