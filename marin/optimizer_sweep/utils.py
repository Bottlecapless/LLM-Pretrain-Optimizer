import dataclasses
import logging
from collections.abc import Sequence

import ray
from levanter.models.llama import LlamaConfig

from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep

logger = logging.getLogger("ray")

def config_to_train_config(
    config_in_dict,
    target_steps,
    config_generator,
    resources,
):
    train_configs = []
    for config, target_step in zip(config_in_dict, target_steps, strict=False):
        new_config = config.copy()
        train_batch_size = new_config.pop("train_batch_size")
        learning_rate = new_config["learning_rate"]
        optimizer_config = config_generator(**new_config)
        train_configs.append(
            SimpleTrainConfig(
                resources=resources,
                steps_per_eval=min(1000, target_step - 1),
                num_train_steps=target_step,
                train_batch_size=train_batch_size,
                learning_rate=learning_rate,
                optimizer_config=optimizer_config,
            )
        )
    return train_configs


def _failure_ok_train(*args, **kwargs):
    """
    Wrapper to catch exceptions and log them, but not fail the whole sweep. We do this because some batch sizes are too
    big.
    """
    from marin.training.training import run_levanter_train_lm

    try:
        return ray.get(run_levanter_train_lm.remote(*args, **kwargs))
    except Exception as e:
        logger.exception("Failed to run training", exc_info=e)
        return None


def make_sweep_steps(
    prefix: str,
    model_config: LlamaConfig,
    train_configs,
    tokenized_data: ExecutorStep,
    format_train_config,
    default_train,
    tags: Sequence[str] = (),
):
    steps = []
    for train_config in train_configs:
        name = format_train_config(prefix, train_config)
        step = default_train(
            name=name,
            train_config=train_config,
            model_config=model_config,
            tokenized=tokenized_data,
            tags=tags,
            use_default_validation=False,
        )

        step = dataclasses.replace(step, fn=_failure_ok_train)

        steps.append(step)
    return steps
