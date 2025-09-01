# config_handler.py
import os
from pathlib import Path
import yaml
from pydantic import BaseModel, Field
from typing import Dict, Tuple, List, Optional

class MQTTConfig(BaseModel):
    BROKER: str = Field(default=None)
    PORT: int = Field(default=None)
    CLIENT_ID: str = Field(default=None)
    USERNAME: Optional[str] = Field(default=None)
    PASSWORD: Optional[str] = Field(default=None)
    TOPIC: dict = Field(default=None)

class UWBConfig(BaseModel):
    ANCHOR_SERIALS: List[str] = Field(default_factory=list)
    ANCHOR_POSITIONS: Dict[str, Tuple[float, float]] = Field(default_factory=dict)
    ANCHOR_RANGINGS: Dict[str, float] = Field(default_factory=dict)
    MODEL_PATH: str = Field(default=None)
    CALIBRATE_THRESHOLD: float = Field(default=None)

class CONFIG(BaseModel):
    MQTT: MQTTConfig
    UWB: UWBConfig

    @classmethod
    def from_yaml(cls, config_file: str = 'config.yaml'):
        with open(config_file, 'r') as file:
            config_data = yaml.safe_load(file)
        return cls(**config_data)

# Initialize the configuration
config_path =  os.path.join(str(os.getcwd()), 'config.yaml')
conf = CONFIG.from_yaml(config_path)

if __name__ == "__main__":
    conf_mqtt = conf.MQTT
    broker = conf_mqtt.BROKER
    print(f"MQTT Config: {broker}")
