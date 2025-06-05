import time

import fire

from plotting_benchmark.benchmark import PlottingBenchmark
from plotting_benchmark.custom_task_changer import TaskShortner


def main(
    limit: int | list[int] | None = None,
    cuda: int = 0,
    model: str = None,
    config: str = None
):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)

    benchmark = PlottingBenchmark(config_path=config)

    if model is not None:
        benchmark.model_names = [model] 
        print(f"[INFO] Using model: {benchmark.model_names[0]}")

    benchmark.run_benchmark(limit, reuse_results=False, load_intermediate=False, only_stats=False, skip_score=True)

if __name__ == "__main__":
    fire.Fire(main)
