"""Thermometry: sensor map, calibration generations and converted readings.

Every published generation of the ``sensors`` scope freezes the full sensor
map together with the calibration of the sensor that moved. Readings are
stamped with the generation that was current when they were taken, so a
historical trace always interprets them with the mapping and calibration in
force at that time; only "current state" views follow the head generation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from ..core.clock import Clock
from ..core.config import ControlConfig, TemperatureEnvelope
from ..core.ids import validate_token
from ..errors import NotFoundError, StaleGenerationError, ValidationError
from ..persistence.store import DurableStore
from ..telemetry.readings import Reading, ReadingSeries
from ..versioning.generations import GenerationRegistry


@dataclass(frozen=True)
class Sensor:
    sensor_id: str
    position: str
    gain: float
    offset: float
    generation: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Sensor":
        return cls(
            sensor_id=str(value["sensor_id"]),
            position=str(value["position"]),
            gain=float(value["gain"]),
            offset=float(value["offset"]),
            generation=int(value["generation"]),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "position": self.position,
            "gain": self.gain,
            "offset": self.offset,
            "generation": self.generation,
        }


class Thermometry:
    """Keeps the sensor map and every calibration pinned to one generation scope."""

    scope = "sensors"
    document = "thermometry"

    def __init__(
        self,
        store: DurableStore,
        clock: Clock,
        config: ControlConfig,
        generations: GenerationRegistry,
        *,
        series_limit: int = 128,
    ) -> None:
        self.store = store
        self.clock = clock
        self.config = config
        self.generations = generations
        self._series_limit = series_limit
        self._sensors: dict[str, Sensor] = {}
        self._channels: list[str] = []
        self._series: dict[str, ReadingSeries] = {}
        self._load()
        self._replay()

    def _load(self) -> None:
        stored = self.store.try_read(self.document)
        if stored is None:
            return
        for item in stored.payload.get("sensors", []):
            sensor = Sensor.from_dict(item)
            self._sensors[sensor.sensor_id] = sensor
        self._channels = [str(item) for item in stored.payload.get("channels", [])]

    def _replay(self) -> None:
        """Rebuild channel lineage from the durable generation history.

        The generation journal is the single source of truth for past maps
        and calibrations, so a restart interprets old readings exactly as the
        shift that recorded them did.
        """

        self._history: dict[str, list[Sensor]] = {}
        positions: dict[str, str] = {}
        calibrations: dict[str, tuple[float, float]] = {}
        for revision in self.generations.revisions(self.scope):
            payload = revision.payload
            sensor_map = payload.get("sensor_map")
            sensor_id = payload.get("sensor")
            if not isinstance(sensor_map, dict) or sensor_id is None:
                continue
            key = str(sensor_id)
            # A revision freezes the whole map: learn positions of every
            # channel known at that time.
            for mapped_id, position in sensor_map.items():
                positions.setdefault(str(mapped_id), str(position))
            if "gain" in payload and "offset" in payload:
                calibrations[key] = (float(payload["gain"]), float(payload["offset"]))
                positions[key] = str(sensor_map.get(key, positions.get(key, "")))
            gain, offset = calibrations.get(key, (1.0, 0.0))
            sensor = Sensor(
                sensor_id=key,
                position=positions[key],
                gain=gain,
                offset=offset,
                generation=revision.generation,
            )
            self._history.setdefault(key, []).append(sensor)
            self._sensors[key] = sensor
            if key not in self._channels:
                self._channels.append(key)

    def persist(self) -> None:
        self.store.write(
            self.document,
            {
                "sensors": [self._sensors[key].as_dict() for key in sorted(self._sensors)],
                "channels": list(self._channels),
            },
        )

    # -- registration and calibration -------------------------------------

    def register_sensor(self, sensor_id: str, position: str, *, reason: str = "commissioning") -> Sensor:
        key = validate_token(sensor_id, field_name="sensor id")
        if key in self._sensors:
            raise ValidationError("sensor is already registered", sensor=key)
        return self._publish(key, position, gain=1.0, offset=0.0, reason=reason)

    def remap(self, sensor_id: str, position: str, *, reason: str = "sensor replacement") -> Sensor:
        existing = self.sensor(sensor_id)
        return self._publish(existing.sensor_id, position, gain=existing.gain, offset=existing.offset, reason=reason)

    def calibrate(self, sensor_id: str, gain: float, offset: float = 0.0, *, reason: str = "calibration") -> Sensor:
        existing = self.sensor(sensor_id)
        if float(gain) <= 0:
            raise ValidationError("calibration gain must be positive", sensor=existing.sensor_id)
        return self._publish(existing.sensor_id, existing.position, gain=float(gain), offset=float(offset), reason=reason)

    def _publish(self, sensor_id: str, position: str, *, gain: float, offset: float, reason: str) -> Sensor:
        label = validate_token(position, field_name="sensor position")
        sensor_map = {key: value.position for key, value in sorted(self._sensors.items())}
        sensor_map[sensor_id] = label
        revision = self.generations.publish(
            self.scope,
            {
                **asdict(self.config.temperature),
                "sensor_map": sensor_map,
                "sensor": sensor_id,
                "gain": gain,
                "offset": offset,
            },
            reason=reason,
        )
        sensor = Sensor(sensor_id=sensor_id, position=label, gain=gain, offset=offset, generation=revision.generation)
        self._sensors[sensor_id] = sensor
        self._history.setdefault(sensor_id, []).append(sensor)
        if sensor_id not in self._channels:
            self._channels.append(sensor_id)
        self.persist()
        return sensor

    def sensor(self, sensor_id: str) -> Sensor:
        key = str(sensor_id)
        if key not in self._sensors:
            raise NotFoundError("sensor is not registered", sensor=key)
        return self._sensors[key]

    def sensors(self) -> list[Sensor]:
        return [self._sensors[key] for key in sorted(self._sensors)]

    def sensor_history(self, sensor_id: str) -> list[Sensor]:
        """Every mapping/calibration revision ever published for the sensor."""

        self.sensor(sensor_id)
        return list(self._history.get(str(sensor_id), []))

    def _revision_sensor(self, sensor_id: str, generation: int) -> Sensor:
        """The sensor state (position, gain, offset) pinned at a generation.

        Each sensor keeps the state of the channel at every generation in
        which that channel moved; the state is constant between two moves, so
        the newest revision not later than the requested one applies.
        """

        key = str(sensor_id)
        self.sensor(key)
        wanted = int(generation)
        head = self.generations.generation(self.scope) if self.scope in self.generations.scopes() else 0
        if wanted > head or self.generations.revision(self.scope, wanted) is None:
            raise StaleGenerationError(
                "generation was never published for this scope",
                scope=self.scope,
                generation=wanted,
                current=head,
            )
        lineage = self._history.get(key, [])
        selected: Sensor | None = None
        for sensor in lineage:
            if sensor.generation <= wanted:
                selected = sensor
            else:
                break
        if selected is None:
            first = lineage[0].generation if lineage else None
            raise StaleGenerationError(
                "generation predates the sensor mapping and calibration",
                scope=self.scope,
                generation=wanted,
                earliest=first,
            )
        return selected

    def envelope_at(self, generation: int) -> "TemperatureEnvelope":
        """Reconstruct the temperature envelope frozen at a published generation."""

        revision = self.generations.revision(self.scope, int(generation))
        if revision is None:
            raise StaleGenerationError(
                "generation was never published for this scope",
                scope=self.scope,
                generation=int(generation),
                current=self.generations.generation(self.scope),
            )
        fields = TemperatureEnvelope.__dataclass_fields__
        values = {key: revision.payload[key] for key in fields if key in revision.payload}
        return TemperatureEnvelope(**values)

    def current_map(self) -> dict[str, str]:
        return {sensor.sensor_id: sensor.position for sensor in self.sensors()}

    def map_at(self, generation: int) -> dict[str, str]:
        """Reconstruct the sensor map exactly as it stood at a generation.

        Each channel contributes the position of its newest revision not
        later than ``generation``; channels commissioned afterwards are
        absent rather than shown at a position they did not yet hold.
        """

        wanted = int(generation)
        if self.generations.revision(self.scope, wanted) is None:
            raise StaleGenerationError(
                "generation was never published for this scope",
                scope=self.scope,
                generation=wanted,
                current=self.generations.generation(self.scope),
            )
        mapping: dict[str, str] = {}
        for sensor_id, lineage in sorted(self._history.items()):
            for sensor in reversed(lineage):
                if sensor.generation <= wanted:
                    mapping[sensor_id] = sensor.position
                    break
        return mapping

    def position_of(self, sensor_id: str, *, generation: int | None = None) -> str:
        if generation is None:
            return self.sensor(sensor_id).position
        return self._revision_sensor(sensor_id, generation).position

    # -- readings ----------------------------------------------------------

    def series(self, channel: str) -> ReadingSeries:
        key = str(channel)
        if key not in self._series:
            self._series[key] = ReadingSeries(self.store, self.clock, key, limit=self._series_limit)
        return self._series[key]

    def convert(self, sensor_id: str, raw_c: float, *, generation: int | None = None) -> float:
        """Apply the calibration of the requested generation to a raw value.

        With no generation the current calibration is used; pinning the
        generation reproduces the value the line would have seen back then.
        """

        if generation is None:
            selected = self.sensor(sensor_id)
        else:
            selected = self._revision_sensor(sensor_id, int(generation))
        return round(float(raw_c) * selected.gain + selected.offset, 6)

    def record_reading(self, sensor_id: str, raw_c: float, *, unit: str = "degC") -> Reading:
        sensor = self.sensor(sensor_id)
        value = self.convert(sensor_id, raw_c, generation=sensor.generation)
        return self.series(sensor.sensor_id).append(
            value,
            unit=unit,
            generation=sensor.generation,
            raw_value=float(raw_c),
        )

    def provenance(self, reading: Reading) -> dict[str, Any]:
        """Describe the exact mapping and calibration behind one stored reading.

        A historical trace uses only the generation stamped on the reading, so
        a later move or recalibration can never change what the shift saw.
        """

        sensor = self._revision_sensor(reading.channel, reading.generation)
        return {
            "sensor_id": sensor.sensor_id,
            "position": sensor.position,
            "generation": sensor.generation,
            "gain": sensor.gain,
            "offset": sensor.offset,
            "raw_c": reading.raw_value,
            "value_c": reading.value,
            "recorded_at": reading.timestamp,
            "current_position": self.sensor(reading.channel).position,
            "current_generation": self.sensor(reading.channel).generation,
            "mapping_current": sensor.generation == self.sensor(reading.channel).generation,
        }

    def history_as_recorded(self, sensor_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Replay a channel's readings under the mapping each one was taken with."""

        self.sensor(sensor_id)
        return [
            {"reading": reading.as_dict(), "provenance": self.provenance(reading)}
            for reading in self.history(sensor_id, limit=limit)
        ]

    def latest(self, sensor_id: str) -> Reading | None:
        return self.series(str(sensor_id)).latest()

    def history(self, sensor_id: str, *, limit: int | None = None) -> list[Reading]:
        return self.series(str(sensor_id)).all(limit)

__all__ = ["Sensor", "Thermometry"]
