from src.bvviz.config import Configuration


def test_reset():
    cfg = Configuration()
    cfg.transpiler_seed = 0
    cfg.simulator_seed = 0
    cfg.shot_count = 0

    cfg.reset()
    assert cfg.transpiler_seed == 1
    assert cfg.simulator_seed == 1
    assert cfg.shot_count == 1000
