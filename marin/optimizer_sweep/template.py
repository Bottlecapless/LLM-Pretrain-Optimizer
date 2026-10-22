"""Optimizer hyperparameter sweep template.

This module builds and launches a sweep of training runs to find strong
hyperparameters for a given optimizer and model size within Marin.

High-level flow:
- Computes a target token budget using Chinchilla scaling for the selected
  model size, scaled by a user-provided multiplier (`target_chinchilla`).
- Expands a `baseline_config` with `sweep_grids` into concrete training
  configurations and converts them to executable train configs.
- Skips configurations that have already been completed by consulting
  experiment tracking (e.g., Weights & Biases) via `check_baseline_run`.
- Optionally chooses a previously observed near-best baseline to minimize the
  number of remaining runs, then dispatches the sweep as Marin steps.

Data and models:
- Uses a local Route-B DCLM cache by default, configurable through
  `cache_root` / `ROUTE_B_CACHE_ROOT`.
- Maps `model_size` and `optimizer` tags to concrete model and optimizer
  configurations via `map_tag_to_model` and `map_tag_to_config`.

Entrypoint:
- `template(...)` orchestrates the sweep. When `DEBUG_MODE` is `False`, it
  creates and executes steps via the Marin executor. When `force_run` is
  `True`, only the baseline configuration is executed (no sweep).

Example:
    template(
        model_size="1.2b",
        target_chinchilla=1.0,
        optimizer="adamw",
        baseline_config={...},
        sweep_grids={...},
        gpu_type="H100",
        gpu_count=1,
        DEBUG_MODE=False,
        random_suffix=None,
        force_run=False,
    )
"""

# https://github.com/stanford-crfm/marin/issues/725
# Sweep to determine optimal hyperparameters for Adam on small scale
import os

from levanter.data.text import LMMixtureDatasetConfig, UrlDatasetSourceConfig

from experiments.optimizer_sweep.defaults import default_train
from marin.execution.executor import executor_main
from marin.optimizer_sweep.config import map_tag_to_config
from marin.optimizer_sweep.format import map_tag_to_format
from marin.optimizer_sweep.models import calculate_chinchilla, map_tag_to_model
from marin.optimizer_sweep.utils import (
    config_to_train_config,
    make_sweep_steps,
)
from marin.optimizer_sweep.utils_simp import approximate, check_baseline_run, create_configs, grab_best_run
from marin.resources import GpuConfig

# Sweep to determine optimal training config

DEFAULT_ROUTE_B_CACHE_ROOT = os.environ.get(
    "ROUTE_B_CACHE_ROOT", "/mnt/afs/250010233/data/marin_route_b/dclm_mix_5.37B"
)
DEFAULT_TOKENIZER = os.environ.get("ROUTE_B_TOKENIZER", "marin-community/marin-tokenizer")
DEFAULT_GPU_TYPE = os.environ.get("ROUTE_B_GPU_TYPE", "H100")
DEFAULT_GPU_COUNT = int(os.environ.get("ROUTE_B_GPU_COUNT", "1"))


def build_route_b_data_config(
    cache_root: str | None = None,
    tokenizer: str | None = None,
) -> LMMixtureDatasetConfig:
    """Build a training data config for a locally generated Route-B cache."""
    cache_root = os.path.abspath(os.path.expanduser(cache_root or DEFAULT_ROUTE_B_CACHE_ROOT))
    tokenizer = tokenizer or DEFAULT_TOKENIZER
    validation_root = os.path.join(cache_root, "validation", "paloma_c4_en")

    configs = {"route_b_dclm_mix": UrlDatasetSourceConfig(train_urls=[], validation_urls=[], cache_dir=cache_root)}
    train_weights = {"route_b_dclm_mix": 1.0}
    if os.path.isdir(os.path.join(validation_root, "validation")):
        configs["paloma/c4_en"] = UrlDatasetSourceConfig(
            train_urls=[], validation_urls=[], cache_dir=validation_root
        )
        train_weights["paloma/c4_en"] = 0.0

    return LMMixtureDatasetConfig(
        configs=configs,
        train_weights=train_weights,
        tokenizer=tokenizer,
        shuffle=False,
    )


def template(
    model_size,
    target_chinchilla,
    optimizer,
    baseline_config,
    sweep_grids,
    gpu_type=DEFAULT_GPU_TYPE,
    gpu_count=DEFAULT_GPU_COUNT,
    cache_root=DEFAULT_ROUTE_B_CACHE_ROOT,
    tokenizer=DEFAULT_TOKENIZER,
    tokenized_data=None,
    DEBUG_MODE=False,
    random_suffix=None,
    force_run=False,
):
    llama_model = map_tag_to_model[model_size]
    target_data = target_chinchilla * calculate_chinchilla(llama_model)
    data_size = f"{target_data//1_000_000_000}B"
    optimizer_config_generator = map_tag_to_config[optimizer]
    resources = GpuConfig(gpu_count=gpu_count, accelerator_type=gpu_type)
    tokenized_data = tokenized_data or build_route_b_data_config(cache_root, tokenizer)
    train_cache = os.path.join(os.path.abspath(os.path.expanduser(cache_root)), "train")
    if not os.path.isdir(train_cache):
        raise FileNotFoundError(f"Route-B train cache not found: {train_cache}")
    tags = (model_size, data_size, optimizer)
    approximate_best_config_list = []

    def optimal_run_set(baseline_config):
        target_steps, config_in_dict = create_configs(baseline_config, sweep_grids, target_data=target_data)
        train_configs = config_to_train_config(
            config_in_dict, target_steps, config_generator=optimizer_config_generator, resources=resources
        )
        # use wandb to avoid rerunning
        new_train_configs = []
        for train_config, config in zip(train_configs, config_in_dict, strict=False):
            if not check_baseline_run(config, tags):
                print(f"Unfinished: {config}")
                new_train_configs.append(train_config)
        return new_train_configs

    if force_run:
        # For force run, just create a single train config from baseline
        target_steps, _ = create_configs(baseline_config, {}, target_data=target_data)  # Empty sweep grid
        train_configs = config_to_train_config(
            [baseline_config], target_steps, config_generator=optimizer_config_generator, resources=resources
        )
        tags = ("debug", *tags)
    else:
        approximate_best_config_list = []
        if check_baseline_run(baseline_config, tags):
            current_best_config, approximate_best_config_list = grab_best_run(baseline_config.keys(), tags)
            baseline_config = current_best_config
            print(f"Current best config: {baseline_config}")
            print(f"Current Approximate best config: {approximate_best_config_list}")

        if len(approximate_best_config_list) <= 1:
            print("Dont have a choice")
            train_configs = optimal_run_set(baseline_config)
        else:
            train_configs = optimal_run_set(baseline_config)
            approximate_train_configs_len = [
                len(optimal_run_set(approx_baseline_config)) for approx_baseline_config in approximate_best_config_list
            ]
            if min(approximate_train_configs_len) <= len(train_configs) - 10:
                for i in range(len(approximate_best_config_list)):
                    if approximate_train_configs_len[i] == min(approximate_train_configs_len):
                        break
                baseline_config = approximate_best_config_list[i]
                train_configs = optimal_run_set(baseline_config)
            else:
                print("All configs are nearly equally close to finish")

    print(f"Choose: {baseline_config}")
    print(f"Closest to finish: {len(train_configs)}")
    if len(train_configs) > 0 and (not DEBUG_MODE):
        prefix = f"sweep-{model_size}-{data_size}-{optimizer}"
        if random_suffix is not None:
            prefix = prefix + random_suffix
        steps = make_sweep_steps(
            prefix=prefix,
            model_config=llama_model,
            train_configs=train_configs,
            tokenized_data=tokenized_data,
            format_train_config=map_tag_to_format[optimizer],
            default_train=default_train,
            tags=("llama", "dclm", *tags),
        )
        executor_main(steps)

    if not force_run:  # Only do final examination if not force_run
        current_best_config, approximate_best_config_list = grab_best_run(baseline_config.keys(), tags)
        in_side = False
        for approximate_best_config in approximate_best_config_list:
            if approximate(approximate_best_config, baseline_config):
                in_side = True
                break
        if in_side:
            print(f"Found: {baseline_config}")
            current_run_set = optimal_run_set(baseline_config)
            if len(current_run_set) > 0:
                print("Stupid Ray", flush=True)
            else:
                print("Succeed!")
        else:
            print("Stupid Ray", flush=True)
