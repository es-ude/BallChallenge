"""Load the runtime configuration from a config file."""

from dataclasses import dataclass

from cerberus import Validator  # type: ignore[import-untyped]
from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader  # type: ignore


@dataclass(frozen=True)
class MQTTConfig:
    """MQTT Configuration."""

    host: str = "localhost"
    port: int = 1883
    domain: str = "eaip://ballchallenge.ies/"
    id: str = "bc-webapp"


@dataclass
class AppConfig:
    """App Configuration."""

    user_file: str


@dataclass(frozen=True)
class Configuration:
    """Configuration."""

    app: AppConfig
    mqtt: MQTTConfig


__schema = {
    "app": {
        "required": True,
        "type": "dict",
        "schema": {
            "user_file": {
                "required": True,
                "type": "string",
            }
        },
    },
    "mqtt": {
        "required": False,
        "type": "dict",
        "schema": {
            "host": {
                "required": False,
                "type": "string",
            },
            "port": {
                "required": False,
                "type": "number",
                "min": 0,
            },
            "domain": {
                "required": False,
                "type": "string",
            },
            "id": {
                "required": False,
                "type": "string",
            },
        },
    },
}

__validator = Validator(__schema)


def load_yaml(file: str) -> Configuration:
    """Load Yaml as Configuration."""
    with open(file, "rb") as config_file:
        config_raw = load(config_file, Loader=Loader)
    __validator.validate(config_raw)

    return Configuration(
        app=AppConfig(**config_raw["app"]), mqtt=MQTTConfig(**config_raw["mqtt"])
    )


if __name__ == "__main__":
    config = load_yaml("settings/config.yaml")
    print(config)
