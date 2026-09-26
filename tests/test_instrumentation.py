"""Temperature and flow instrumentation lineage."""

from __future__ import annotations

from pathlib import Path

import pytest

from uhtline.errors import NotFoundError, StaleGenerationError, ValidationError

from .support import STERILE_SENSOR, manual_runtime


def test_registering_the_same_sensor_twice_is_rejected(tmp_path: Path) -> None:
    runtime = manual_runtime(tmp_path)
    with pytest.raises(ValidationError):
        runtime.thermometry.register_sensor(STERILE_SENSOR, "duplicate")


def test_an_unknown_sensor_is_reported(tmp_path: Path) -> None:
    runtime = manual_runtime(tmp_path)
    with pytest.raises(NotFoundError):
        runtime.thermometry.sensor("TS-MISSING")


def test_calibration_is_applied_to_new_readings(tmp_path: Path) -> None:
    runtime = manual_runtime(tmp_path)
    assert runtime.thermometry.convert(STERILE_SENSOR, 100.0) == 100.0
    runtime.thermometry.calibrate(STERILE_SENSOR, 1.1, 0.5, reason="calibration")
    assert runtime.thermometry.convert(STERILE_SENSOR, 100.0) == 110.5


def test_conversion_uses_the_calibration_of_the_requested_generation(tmp_path: Path) -> None:
    runtime = manual_runtime(tmp_path)
    first = runtime.thermometry.sensor(STERILE_SENSOR).generation
    runtime.thermometry.calibrate(STERILE_SENSOR, 1.2, 0.0, reason="calibration")
    current = runtime.thermometry.sensor(STERILE_SENSOR).generation
    assert runtime.thermometry.convert(STERILE_SENSOR, 100.0, generation=first) == 100.0
    assert runtime.thermometry.convert(STERILE_SENSOR, 100.0, generation=current) == 120.0


def test_conversion_before_the_first_calibration_is_rejected(tmp_path: Path) -> None:
    runtime = manual_runtime(tmp_path)
    with pytest.raises(StaleGenerationError):
        runtime.thermometry.convert(STERILE_SENSOR, 100.0, generation=0)


def test_the_sensor_map_follows_the_current_position(tmp_path: Path) -> None:
    runtime = manual_runtime(tmp_path)
    previous = runtime.thermometry.sensor(STERILE_SENSOR).generation
    runtime.thermometry.remap(STERILE_SENSOR, "sterilization-inlet", reason="replacement")
    assert runtime.thermometry.position_of(STERILE_SENSOR) == "sterilization-inlet"
    assert runtime.thermometry.map_at(previous)[STERILE_SENSOR] == "sterilization-outlet"
    assert runtime.thermometry.current_map()[STERILE_SENSOR] == "sterilization-inlet"


def test_reading_history_is_bounded_by_the_series_limit(tmp_path: Path) -> None:
    runtime = manual_runtime(tmp_path)
    series = runtime.thermometry.series(STERILE_SENSOR)
    for index in range(series.limit + 10):
        series.append(137.0 + index * 0.01, unit="degC", generation=1)
    assert len(series.all()) == series.limit
    assert len(series.window(5)) == 5
    assert series.latest().value == pytest.approx(137.0 + (series.limit + 9) * 0.01)


def test_reading_statistics_summarise_the_series(tmp_path: Path) -> None:
    runtime = manual_runtime(tmp_path)
    runtime.control.record_temperature(STERILE_SENSOR, 136.0, reason="test")
    runtime.control.record_temperature(STERILE_SENSOR, 138.0, reason="test")
    statistics = runtime.thermometry.series(STERILE_SENSOR).statistics()
    assert statistics == {
        "channel": STERILE_SENSOR,
        "count": 2,
        "minimum": 136.0,
        "maximum": 138.0,
        "mean": 137.0,
        "latest": 138.0,
    }


def test_measurement_applies_the_calibrated_gain(tmp_path: Path) -> None:
    runtime = manual_runtime(tmp_path)
    runtime.flowmeter.calibrate(1.05, reason="calibration")
    reading = runtime.flowmeter.measure(10000.0)
    assert reading.gain == 1.05
    assert reading.litres_per_hour == 10500.0
    assert reading.generation == runtime.generations.generation("flow-calibration")


def test_flow_series_records_the_corrected_rate(tmp_path: Path) -> None:
    runtime = manual_runtime(tmp_path)
    runtime.flowmeter.calibrate(1.1, reason="calibration")
    reading = runtime.control.record_flow(12000.0)
    assert reading["value"] == pytest.approx(13200.0)
    assert runtime.flowmeter.series().latest().value == pytest.approx(13200.0)


def test_flow_envelope_check_flags_a_rate_outside_the_window(tmp_path: Path) -> None:
    runtime = manual_runtime(tmp_path)
    assert runtime.flowmeter.within_envelope(12000.0) is True
    assert runtime.flowmeter.within_envelope(400.0) is False


def test_flow_calibration_history_records_every_change(tmp_path: Path) -> None:
    runtime = manual_runtime(tmp_path)
    runtime.flowmeter.calibrate(1.05, reason="first")
    runtime.flowmeter.calibrate(1.08, 0.5, reason="second")
    history = runtime.flowmeter.calibrations()
    assert [item["gain"] for item in history] == [1.05, 1.08]
    assert history[-1]["offset"] == 0.5
    assert runtime.flowmeter.gain == 1.08


def test_temperature_history_records_every_reading(tmp_path: Path) -> None:
    runtime = manual_runtime(tmp_path)
    runtime.control.record_temperature(STERILE_SENSOR, 136.0, reason="test")
    runtime.control.record_temperature(STERILE_SENSOR, 137.0, reason="test")
    history = runtime.thermometry.history(STERILE_SENSOR)
    assert [item.value for item in history] == [136.0, 137.0]
    assert runtime.thermometry.latest(STERILE_SENSOR).value == 137.0


def test_old_readings_keep_the_position_and_calibration_they_were_taken_with(tmp_path: Path) -> None:
    runtime = manual_runtime(tmp_path)
    old_generation = runtime.thermometry.sensor(STERILE_SENSOR).generation
    runtime.control.record_temperature(STERILE_SENSOR, 100.0, reason="last week")
    runtime.control.remap_sensor(STERILE_SENSOR, "sterilization-inlet", reason="probe moved")
    runtime.control.recalibrate_temperature(STERILE_SENSOR, 1.2, 0.0, reason="post-move calibration")

    reading = runtime.thermometry.history(STERILE_SENSOR)[0]
    assert reading.generation == old_generation
    provenance = runtime.thermometry.provenance(reading)
    assert provenance["position"] == "sterilization-outlet"
    assert provenance["gain"] == 1.0
    assert provenance["mapping_current"] is False
    # The stored value is not recomputed under the new mapping or calibration.
    assert reading.value == 100.0
    assert reading.raw_value == 100.0


def test_replay_judges_a_window_with_the_envelope_of_the_pinned_generation(tmp_path: Path) -> None:
    runtime = manual_runtime(tmp_path)
    for raw in (136.0, 136.5, 137.0):
        runtime.control.record_temperature(STERILE_SENSOR, raw, reason="last week")
    old_generation = runtime.thermometry.sensor(STERILE_SENSOR).generation
    runtime.control.evaluate_window(STERILE_SENSOR, 3, reason="last week")
    runtime.control.remap_sensor(STERILE_SENSOR, "sterilization-inlet", reason="probe moved")

    replay = runtime.control.evaluate_window(
        STERILE_SENSOR,
        3,
        reason="historical trace",
        as_of_generation=old_generation,
    )
    assert replay["replay"] is True
    assert replay["position"] == "sterilization-outlet"
    assert replay["verdict"] == "pass"
    # A read-only replay never appends a new verdict.
    assert runtime.control.decision_history(kind="sterilization-window-replay") == []
    assert len(runtime.control.decision_history(kind="sterilization-window")) == 1


def test_reading_trace_and_lineage_survive_a_restart(tmp_path: Path) -> None:
    first = manual_runtime(tmp_path)
    old_generation = first.thermometry.sensor(STERILE_SENSOR).generation
    first.control.record_temperature(STERILE_SENSOR, 100.0, reason="last week")
    first.control.remap_sensor(STERILE_SENSOR, "sterilization-inlet", reason="probe moved")

    second = manual_runtime(tmp_path)
    trace = second.control.reading_trace(STERILE_SENSOR)
    assert trace["readings"][0]["provenance"]["generation"] == old_generation
    assert trace["readings"][0]["provenance"]["position"] == "sterilization-outlet"
    assert second.control.sensor_map_at(old_generation)["map"][STERILE_SENSOR] == "sterilization-outlet"
    assert second.control.sensor_lineage(STERILE_SENSOR)["current"]["position"] == "sterilization-inlet"


def test_a_channel_commissioned_later_is_absent_from_an_earlier_map(tmp_path: Path) -> None:
    runtime = manual_runtime(tmp_path)
    before = runtime.generations.generation("sensors")
    runtime.thermometry.register_sensor("TS-NEW", "inlet-a", reason="extension")
    assert "TS-NEW" not in runtime.thermometry.map_at(before)
    assert runtime.thermometry.map_at(runtime.generations.generation("sensors"))["TS-NEW"] == "inlet-a"

