from src.bvviz.config import Configuration


def test_reset():
    cfg = Configuration()

    cfg.transpiler_seed = None
    cfg.simulator_seed = None
    cfg.shot_count = None
    cfg.backend = None
    cfg.noise_config = None
    cfg.transpile_config = None

    cfg.reset()

    assert cfg.transpiler_seed == 1
    assert cfg.simulator_seed == 1
    assert cfg.shot_count == 1000
