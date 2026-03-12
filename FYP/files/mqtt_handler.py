"""
MQTT Handler — receives UWB distance data from the tag via MQTT broker.

Topics:
  uwb/distances  - Combined: {"A1": 3.45, "A2": 5.12, "ts": 12345}
  uwb/range      - Per-anchor: {"anchor": "A1", "distance": 3.45, "rx_power": -82.1}
"""

import json
import logging

import paho.mqtt.client as mqtt

from .config import config
from .engine import PositioningEngine

logger = logging.getLogger(__name__)


class MQTTHandler:
    def __init__(self, engine: PositioningEngine):
        self.engine = engine
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.connected = False

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"MQTT connected to {config.mqtt_broker}:{config.mqtt_port}")
            client.subscribe("uwb/distances")
            client.subscribe("uwb/range")
            self.connected = True
        else:
            logger.error(f"MQTT connection failed: rc={rc}")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())

            if msg.topic == "uwb/distances":
                # Combined format: {"A1": 3.45, "A2": 5.12, "ts": 12345}
                for key, val in payload.items():
                    if key.startswith("A") and isinstance(val, (int, float)):
                        self.engine.add_distance(key, float(val))

            elif msg.topic == "uwb/range":
                # Per-anchor format: {"anchor": "A1", "distance": 3.45, ...}
                anchor_id = payload.get("anchor")
                distance = payload.get("distance")
                rx_power = payload.get("rx_power", 0.0)
                if anchor_id and distance is not None:
                    self.engine.add_distance(anchor_id, float(distance), float(rx_power))

        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.error(f"MQTT message error: {e}")

    def connect(self):
        try:
            self.client.connect(config.mqtt_broker, config.mqtt_port, 60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"MQTT connection error: {e}")

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
