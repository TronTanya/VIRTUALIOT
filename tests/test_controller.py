import pytest

from controller import (
    AutomationController,
    ConfigError,
    DeviceStates,
    HysteresisRule,
    ManualController,
    Mode,
    SensorReadings,
    Thresholds,
)

TH = Thresholds(
    heater_on_below=20.0, heater_off_above=22.0,
    fan_off_below=50.0, fan_on_above=60.0,
    lamp_on_below=250.0, lamp_off_above=350.0,
)


def readings(temp=21.0, hum=55.0, total=300.0, ambient=300.0):
    return SensorReadings(temperature=temp, humidity=hum, illuminance_total=total, illuminance_ambient=ambient)


@pytest.fixture
def ctrl():
    return AutomationController(TH)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"heater_on_below": 22.0, "heater_off_above": 22.0},
        {"heater_on_below": 23.0, "heater_off_above": 22.0},
        {"fan_off_below": 60.0, "fan_on_above": 55.0},
        {"lamp_on_below": 400.0, "lamp_off_above": 350.0},
        {"fan_on_above": 120.0},
        {"lamp_on_below": -10.0},
    ],
)
def test_invalid_thresholds_rejected(kwargs):
    from dataclasses import replace

    with pytest.raises(ConfigError):
        AutomationController(replace(TH, **kwargs))


def test_heater_transitions(ctrl):
    s, d = ctrl.evaluate(readings(temp=19.5), DeviceStates())
    assert s.heater and d[0].device == "heater" and d[0].new_state
    s, d = ctrl.evaluate(readings(temp=22.5), s)
    assert not s.heater and len(d) == 1


def test_fan_transitions(ctrl):
    s, _ = ctrl.evaluate(readings(hum=65.0), DeviceStates())
    assert s.fan
    s, _ = ctrl.evaluate(readings(hum=45.0), s)
    assert not s.fan


def test_lamp_transitions(ctrl):
    s, _ = ctrl.evaluate(readings(ambient=200.0, total=200.0), DeviceStates())
    assert s.lamp
    s, _ = ctrl.evaluate(readings(ambient=400.0, total=800.0), s)
    assert not s.lamp


@pytest.mark.parametrize("initial", [True, False])
def test_state_retained_inside_band(ctrl, initial):
    current = DeviceStates(heater=initial, fan=initial, lamp=initial)
    s, d = ctrl.evaluate(readings(temp=21.0, hum=55.0, ambient=300.0), current)
    assert s == current
    assert d == []


@pytest.mark.parametrize("initial", [True, False])
def test_exact_threshold_values_retain_state(ctrl, initial):
    current = DeviceStates(heater=initial, fan=initial, lamp=initial)
    for temp, hum, lux in ((20.0, 50.0, 250.0), (22.0, 60.0, 350.0)):
        s, d = ctrl.evaluate(readings(temp=temp, hum=hum, ambient=lux, total=lux), current)
        assert s == current and d == []


def test_rule_on_above_mode():
    rule = HysteresisRule(50.0, 60.0, Mode.ON_ABOVE)
    assert rule.next_state(False, 60.01)[0] is True
    assert rule.next_state(True, 49.99)[0] is False
    assert rule.next_state(True, 55.0) == (True, None)


def test_lamp_ignores_its_own_light(ctrl):
    # Лампа включена, общая освещённость высокая из-за самой лампы,
    # но естественный свет остаётся низким — лампа не выключается.
    s, d = ctrl.evaluate(readings(ambient=200.0, total=600.0), DeviceStates(lamp=True))
    assert s.lamp and d == []


def test_manual_controller_fixed_states():
    manual = ManualController(DeviceStates(heater=True))
    s, d = manual.evaluate(readings(temp=30.0), DeviceStates())
    assert s == DeviceStates(heater=True)
    assert [x.device for x in d] == ["heater"]
    s2, d2 = manual.evaluate(readings(temp=5.0), s)
    assert s2 == s and d2 == []
