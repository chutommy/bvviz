from typing import Dict, Any

import pytest
from matplotlib import pyplot as plt

from src.bvviz.engine import Engine, preprocess


@pytest.mark.parametrize('config', [
    {
        'backend_choice': 0,
        'shots': 10,
        'reset_err': 0.01,
        'meas_err': 0.1,
        'single_err': 0.1,
        'double_err': 0.1,
        'layout': 'trivial',
        'routing': 'basic',
        'translation': 'unroller',
        'optimization': 1,
        'approx': 0.9
    },
    {
        'backend_choice': 0,
        'shots': 100,
        'reset_err': 0.02,
        'meas_err': 0.13,
        'single_err': 0.16,
        'double_err': 0.11,
        'layout': 'noise_adaptive',
        'routing': 'lookahead',
        'translation': 'unroller',
        'optimization': 1,
        'approx': 0.93
    },
    {
        'backend_choice': 0,
        'shots': 1000,
        'reset_err': 0.032,
        'meas_err': 0.013,
        'single_err': 0.042,
        'double_err': 0.063,
        'layout': 'sabre',
        'routing': 'stochastic',
        'translation': 'unroller',
        'optimization': 2,
        'approx': 0.81
    },
    {
        'backend_choice': 0,
        'shots': 800,
        'reset_err': 0.013,
        'meas_err': 0.112,
        'single_err': 0.1625,
        'double_err': 0.1932,
        'layout': 'noise_adaptive',
        'routing': 'sabre',
        'translation': 'unroller',
        'optimization': 3,
        'approx': 0.7
    },
    {
        'backend_choice': 0,
        'shots': 800,
        'reset_err': 0.313,
        'meas_err': 0.712,
        'single_err': 0.8625,
        'double_err': 0.5932,
        'layout': 'trivial',
        'routing': 'basic',
        'translation': 'unroller',
        'optimization': 3,
        'approx': 0.4
    },
    {
        'backend_choice': 0,
        'shots': 2000,
        'reset_err': 0.713,
        'meas_err': 0.012,
        'single_err': 0.0625,
        'double_err': 0.0932,
        'layout': 'dense',
        'routing': 'basic',
        'translation': 'unroller',
        'optimization': 1,
        'approx': 0.321
    },
    {
        'backend_choice': 0,
        'shots': 1,
        'reset_err': 0,
        'meas_err': 0,
        'single_err': 0,
        'double_err': 0,
        'layout': 'trivial',
        'routing': 'basic',
        'translation': 'unroller',
        'optimization': 0,
        'approx': 0,
    },
    {
        'backend_choice': 0,
        'shots': 1,
        'reset_err': 1,
        'meas_err': 1,
        'single_err': 1,
        'double_err': 1,
        'layout': 'trivial',
        'routing': 'basic',
        'translation': 'unroller',
        'optimization': 1,
        'approx': 1,
    },
])
def test_engine(config: Dict[str, Any]):
    engine = Engine()
    engine.configure(config)
    assert engine.check_secret_size('110101')
    result = engine.run('110101')
    assert result is not None
    # pylint: disable=W0703
    try:
        mem = preprocess(result)
        assert mem is not None
    except Exception as exc:
        assert False, exc
    plt.close('all')
