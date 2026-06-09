import numpy as np
from anoship.detectors._ahsc.enhancement import AntiHabituation


def test_response_is_non_negative_and_shape_preserved():
    ah = AntiHabituation(dim=3, alpha=0.2, beta=0.1)
    out = ah.transform(np.array([[1.0, -2.0, 3.0], [0.5, 0.5, 0.5]]))
    assert out.shape == (2, 3)
    assert np.all(out >= 0.0)


def test_first_step_with_zero_weights_is_relu_of_input():
    ah = AntiHabituation(dim=2, alpha=0.3, beta=0.1)
    x = ah.step(np.array([2.0, -1.0]))
    assert np.allclose(x, [2.0, 0.0])  # max(s - 0, 0)


def test_repeated_stimulus_response_decreases_over_time():
    ah = AntiHabituation(dim=2, alpha=0.3, beta=0.1)
    s = np.array([1.0, 1.0])
    responses = [float(np.linalg.norm(ah.step(s))) for _ in range(15)]
    # habituation: the response to a recurring background is suppressed
    assert responses[-1] < responses[0]
    assert all(b <= a + 1e-9 for a, b in zip(responses, responses[1:]))
    # but it does not vanish entirely (stable fixed point > 0)
    assert responses[-1] > 0.0


def test_novel_pattern_retains_higher_response_than_habituated_background():
    ah = AntiHabituation(dim=3, alpha=0.3, beta=0.05)
    background = np.array([1.0, 1.0, 1.0])
    for _ in range(40):
        ah.step(background)
    habituated = np.linalg.norm(ah.step(background))
    novel = np.linalg.norm(ah.step(np.array([1.0, 1.0, 5.0])))
    assert novel > habituated


def test_reset_clears_state():
    ah = AntiHabituation(dim=2, alpha=0.3, beta=0.1)
    for _ in range(10):
        ah.step(np.array([1.0, 1.0]))
    ah.reset()
    assert np.allclose(ah.weights, 0.0)
    assert np.allclose(ah.step(np.array([2.0, 2.0])), [2.0, 2.0])
