"""Reproduce the first-row 130M results for AdamW, Adam-mini, and Muon.

The six scenarios are intentionally written out independently. In particular,
the Adam-mini 1x/2x epsilon and gradient-clipping values, and the Muon 1x/2x
momentum and learning rates, must not be inherited from a shared base config.
"""

import dataclasses
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CACHE_ROOT = "/mnt/afs/250010233/data/marin_route_b/dclm_mix_5.37B"
CACHE_ROOT = os.path.abspath(os.path.expanduser(os.environ.get("ROUTE_B_CACHE_ROOT", DEFAULT_CACHE_ROOT)))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(CACHE_ROOT, ".hf_home"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))

from levanter.optim import AdamConfig, MiniConfig, MuonConfig
from levanter.tracker.wandb import WandbConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.optimizer_sweep.defaults import default_train  # noqa: E402
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.optimizer_sweep.models import calculate_chinchilla, map_tag_to_model
from marin.optimizer_sweep.template import build_route_b_data_config
from marin.resources import GpuConfig

MODEL_TAG = "130m"
MODEL_CONFIG = map_tag_to_model[MODEL_TAG]
SEQ_LEN = 4096
BATCH_SIZE = 128
TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN

TOKENS_1X = 2_684_354_560
TOKENS_2X = 5_368_709_120
STEPS_1X = TOKENS_1X // TOKENS_PER_STEP
STEPS_2X = TOKENS_2X // TOKENS_PER_STEP

TOKENIZER = os.environ.get("ROUTE_B_TOKENIZER", "marin-community/marin-tokenizer")

DEFAULT_MARIN_PREFIX = "/mnt/afs/250010233/data/marin_optimizer_paper/130m_adamw_mini_muon"
MARIN_PREFIX = os.path.abspath(os.path.expanduser(os.environ.get("MARIN_PREFIX", DEFAULT_MARIN_PREFIX)))
os.environ.setdefault("MARIN_PREFIX", MARIN_PREFIX)

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "marin-community")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "optimizer-scaling")
WANDB_GROUP = os.environ.get("WANDB_GROUP", "paper130m-adamw-mini-muon")

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
COMMON_TAGS = ("paper130m", "130m", "route-b", "h100")


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    description: str
    optimizer_name: str
    chinchilla: int
    target_tokens: int
    expected_loss: float
    train_config: SimpleTrainConfig
    tags: tuple[str, ...]


def adamw_130m_1x() -> Scenario:
    optimizer = AdamConfig(
        learning_rate=0.008,
        weight_decay=0.1,
        min_lr_ratio=0.0,
        warmup=2000,
        decay=None,
        rewarmup=0.0,
        lr_schedule="cosine",
        beta1=0.9,
        beta2=0.98,
        epsilon=1e-20,
        max_grad_norm=1.0,
        nesterov=False,
    )
    train_config = SimpleTrainConfig(
        resources=GpuConfig(gpu_count=1, accelerator_type="H100"),
        train_batch_size=128,
        num_train_steps=STEPS_1X,
        learning_rate=0.008,
        steps_per_eval=1000,
        steps_per_export=5000,
        steps_per_hf_export=-1,
        optimizer_config=optimizer,
    )
    return Scenario(
        scenario_id="adamw_1x",
        description="AdamW, 130M, 1x Chinchilla, paper Table 36 row 1",
        optimizer_name="adamw",
        chinchilla=1,
        target_tokens=TOKENS_1X,
        expected_loss=3.529,
        train_config=train_config,
        tags=(*COMMON_TAGS, "adamw", "1x", "expected_loss=3.529"),
    )


def adamw_130m_2x() -> Scenario:
    optimizer = AdamConfig(
        learning_rate=0.008,
        weight_decay=0.1,
        min_lr_ratio=0.0,
        warmup=2000,
        decay=None,
        rewarmup=0.0,
        lr_schedule="cosine",
        beta1=0.9,
        beta2=0.98,
        epsilon=1e-20,
        max_grad_norm=1.0,
        nesterov=False,
    )
    train_config = SimpleTrainConfig(
        resources=GpuConfig(gpu_count=1, accelerator_type="H100"),
        train_batch_size=128,
        num_train_steps=STEPS_2X,
        learning_rate=0.008,
        steps_per_eval=1000,
        steps_per_export=5000,
        steps_per_hf_export=-1,
        optimizer_config=optimizer,
    )
    return Scenario(
        scenario_id="adamw_2x",
        description="AdamW, 130M, 2x Chinchilla, paper Table 37 row 1",
        optimizer_name="adamw",
        chinchilla=2,
        target_tokens=TOKENS_2X,
        expected_loss=3.409,
        train_config=train_config,
        tags=(*COMMON_TAGS, "adamw", "2x", "expected_loss=3.409"),
    )


def adam_mini_130m_1x() -> Scenario:
    optimizer = MiniConfig(
        learning_rate=0.008,
        weight_decay=0.1,
        min_lr_ratio=0.0,
        warmup=2000,
        decay=None,
        rewarmup=0.0,
        lr_schedule="cosine",
        beta1=0.9,
        beta2=0.98,
        epsilon=1e-15,
        max_grad_norm=1.0,
        nesterov=False,
    )
    train_config = SimpleTrainConfig(
        resources=GpuConfig(gpu_count=1, accelerator_type="H100"),
        train_batch_size=128,
        num_train_steps=STEPS_1X,
        learning_rate=0.008,
        steps_per_eval=1000,
        steps_per_export=5000,
        steps_per_hf_export=-1,
        optimizer_config=optimizer,
    )
    return Scenario(
        scenario_id="adam_mini_1x",
        description="Adam-mini, 130M, 1x Chinchilla, paper Table 64 row 1",
        optimizer_name="adam_mini",
        chinchilla=1,
        target_tokens=TOKENS_1X,
        expected_loss=3.542,
        train_config=train_config,
        tags=(*COMMON_TAGS, "adam_mini", "1x", "expected_loss=3.542"),
    )


def adam_mini_130m_2x() -> Scenario:
    optimizer = MiniConfig(
        learning_rate=0.008,
        weight_decay=0.1,
        min_lr_ratio=0.0,
        warmup=2000,
        decay=None,
        rewarmup=0.0,
        lr_schedule="cosine",
        beta1=0.9,
        beta2=0.98,
        epsilon=1e-20,
        max_grad_norm=2.0,
        nesterov=False,
    )
    train_config = SimpleTrainConfig(
        resources=GpuConfig(gpu_count=1, accelerator_type="H100"),
        train_batch_size=128,
        num_train_steps=STEPS_2X,
        learning_rate=0.008,
        steps_per_eval=1000,
        steps_per_export=5000,
        steps_per_hf_export=-1,
        optimizer_config=optimizer,
    )
    return Scenario(
        scenario_id="adam_mini_2x",
        description="Adam-mini, 130M, 2x Chinchilla, paper Table 65 row 1",
        optimizer_name="adam_mini",
        chinchilla=2,
        target_tokens=TOKENS_2X,
        expected_loss=3.416,
        train_config=train_config,
        tags=(*COMMON_TAGS, "adam_mini", "2x", "expected_loss=3.416"),
    )


def muon_130m_1x() -> Scenario:
    optimizer = MuonConfig(
        learning_rate=0.016,
        adam_lr=0.0032,
        weight_decay=0.1,
        min_lr_ratio=0.0,
        warmup=0,
        decay=0.8,
        rewarmup=0.0,
        lr_schedule="linear",
        momentum=0.95,
        nesterov=True,
        backend_steps=5,
        beta1=0.8,
        beta2=0.98,
        epsilon=1e-15,
        muon_epsilon=1e-5,
        max_grad_norm=1.0,
    )
    train_config = SimpleTrainConfig(
        resources=GpuConfig(gpu_count=1, accelerator_type="H100"),
        train_batch_size=128,
        num_train_steps=STEPS_1X,
        learning_rate=0.016,
        steps_per_eval=1000,
        steps_per_export=5000,
        steps_per_hf_export=-1,
        optimizer_config=optimizer,
    )
    return Scenario(
        scenario_id="muon_1x",
        description="Muon, 130M, 1x Chinchilla, paper Table 82 row 1",
        optimizer_name="muon",
        chinchilla=1,
        target_tokens=TOKENS_1X,
        expected_loss=3.464,
        train_config=train_config,
        tags=(*COMMON_TAGS, "muon", "1x", "expected_loss=3.464"),
    )


def muon_130m_2x() -> Scenario:
    optimizer = MuonConfig(
        learning_rate=0.008,
        adam_lr=0.0024,
        weight_decay=0.1,
        min_lr_ratio=0.0,
        warmup=0,
        decay=0.8,
        rewarmup=0.0,
        lr_schedule="linear",
        momentum=0.98,
        nesterov=True,
        backend_steps=5,
        beta1=0.8,
        beta2=0.98,
        epsilon=1e-15,
        muon_epsilon=1e-5,
        max_grad_norm=1.0,
    )
    train_config = SimpleTrainConfig(
        resources=GpuConfig(gpu_count=1, accelerator_type="H100"),
        train_batch_size=128,
        num_train_steps=STEPS_2X,
        learning_rate=0.008,
        steps_per_eval=1000,
        steps_per_export=5000,
        steps_per_hf_export=-1,
        optimizer_config=optimizer,
    )
    return Scenario(
        scenario_id="muon_2x",
        description="Muon, 130M, 2x Chinchilla, paper Table 83 row 1",
        optimizer_name="muon",
        chinchilla=2,
        target_tokens=TOKENS_2X,
        expected_loss=3.369,
        train_config=train_config,
        tags=(*COMMON_TAGS, "muon", "2x", "expected_loss=3.369"),
    )


ALL_SCENARIOS = {
    "adamw_1x": adamw_130m_1x,
    "adamw_2x": adamw_130m_2x,
    "adam_mini_1x": adam_mini_130m_1x,
    "adam_mini_2x": adam_mini_130m_2x,
    "muon_1x": muon_130m_1x,
    "muon_2x": muon_130m_2x,
}


def _selected_scenarios() -> list[Scenario]:
    requested = os.environ.get("SCENARIOS", "all").strip()
    if requested.lower() in {"", "all"}:
        return [factory() for factory in ALL_SCENARIOS.values()]
    names = [name.strip() for name in requested.split(",") if name.strip()]
    unknown = sorted(set(names) - set(ALL_SCENARIOS))
    if unknown:
        raise SystemExit(f"Unknown scenarios: {', '.join(unknown)}. Available: {', '.join(ALL_SCENARIOS)}")
    return [ALL_SCENARIOS[name]() for name in names]


def _validate_scenarios(scenarios: list[Scenario]) -> None:
    assert MODEL_CONFIG.seq_len == SEQ_LEN
    assert calculate_chinchilla(MODEL_CONFIG) == TOKENS_1X
    assert TOKENS_1X == STEPS_1X * TOKENS_PER_STEP
    assert TOKENS_2X == STEPS_2X * TOKENS_PER_STEP
    if len({scenario.scenario_id for scenario in scenarios}) != len(scenarios):
        raise ValueError("Scenario ids must be unique")

    train_cache = Path(CACHE_ROOT) / "train"
    validation_cache = Path(CACHE_ROOT) / "validation" / "paloma_c4_en" / "validation"
    manifest_path = Path(CACHE_ROOT) / "manifest.json"
    if not train_cache.is_dir():
        raise FileNotFoundError(f"Missing Route-B train cache: {train_cache}")
    if not validation_cache.is_dir():
        raise FileNotFoundError(f"Missing Paloma C4/EN validation cache: {validation_cache}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Route-B manifest: {manifest_path}")

    with manifest_path.open() as handle:
        manifest = json.load(handle)
    if int(manifest["actual_tokens"]) < TOKENS_2X:
        raise ValueError(
            f"Route-B cache has {manifest['actual_tokens']} tokens, but 2x requires {TOKENS_2X}: {CACHE_ROOT}"
        )
    if int(manifest["seq_len"]) != SEQ_LEN:
        raise ValueError(f"Cache seq_len is {manifest['seq_len']}, expected {SEQ_LEN}")

    for scenario in scenarios:
        config = scenario.train_config
        if config.train_batch_size != BATCH_SIZE:
            raise ValueError(f"{scenario.scenario_id}: batch size is {config.train_batch_size}, expected {BATCH_SIZE}")
        expected_steps = scenario.target_tokens // TOKENS_PER_STEP
        if config.num_train_steps != expected_steps:
            raise ValueError(
                f"{scenario.scenario_id}: {config.num_train_steps} steps, expected {expected_steps}"
            )
        if not isinstance(config.resources, GpuConfig):
            raise ValueError(f"{scenario.scenario_id}: expected GpuConfig")
        if config.resources.gpu_count != 1 or config.resources.accelerator_type != "H100":
            raise ValueError(f"{scenario.scenario_id}: expected exactly 1 H100")


def _manifest(scenarios: list[Scenario]) -> dict[str, object]:
    return {
        "model": MODEL_TAG,
        "model_seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "cache_root": CACHE_ROOT,
        "tokenizer": TOKENIZER,
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "scenarios": [
            {
                "id": scenario.scenario_id,
                "description": scenario.description,
                "optimizer": scenario.optimizer_name,
                "chinchilla": scenario.chinchilla,
                "target_tokens": scenario.target_tokens,
                "num_train_steps": scenario.train_config.num_train_steps,
                "expected_loss": scenario.expected_loss,
                "tags": list(scenario.tags),
                "optimizer_config": dataclasses.asdict(scenario.train_config.optimizer_config),
            }
            for scenario in scenarios
        ],
    }


def _write_manifest(manifest: dict[str, object]) -> Path:
    output_root = Path(MARIN_PREFIX)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "paper_130m_scenarios.json"
    temporary_path = path.with_suffix(".json.tmp")
    with temporary_path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(path)
    return path


def _configure_wandb(step, scenario: Scenario):
    config = step.config
    inner_config = config.train_config
    tracker = inner_config.trainer.tracker
    if not isinstance(tracker, WandbConfig):
        raise TypeError(f"Expected WandbConfig, got {type(tracker).__name__}")
    tracker = dataclasses.replace(
        tracker,
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=scenario.scenario_id,
        group=WANDB_GROUP,
    )
    trainer = dataclasses.replace(inner_config.trainer, tracker=tracker)
    inner_config = dataclasses.replace(inner_config, trainer=trainer)
    return dataclasses.replace(step, config=dataclasses.replace(config, train_config=inner_config))


def main() -> None:
    scenarios = _selected_scenarios()
    _validate_scenarios(scenarios)
    manifest = _manifest(scenarios)
    manifest_path = _write_manifest(manifest)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[paper130m] Scenario manifest: {manifest_path}")
    print(f"[paper130m] MARIN_PREFIX:      {MARIN_PREFIX}")
    print(f"[paper130m] W&B:                {WANDB_ENTITY}/{WANDB_PROJECT} group={WANDB_GROUP}")

    tokenized_data = build_route_b_data_config(CACHE_ROOT, TOKENIZER)
    steps = []
    for scenario in scenarios:
        step = default_train(
            name=f"paper130m/{scenario.scenario_id}",
            tokenized=tokenized_data,
            model_config=MODEL_CONFIG,
            train_config=scenario.train_config,
            tags=scenario.tags,
            use_default_validation=False,
            eval_harness_tasks=(),
        )
        steps.append(_configure_wandb(step, scenario))

    if DRY_RUN:
        for scenario, step in zip(scenarios, steps, strict=True):
            print(f"[paper130m] built {scenario.scenario_id}: {step.name}")
        print("[paper130m] DRY_RUN=1: not submitting training steps")
        return

    executor_main(
        steps=steps,
        description="130M AdamW vs Adam-mini vs Muon at 1x and 2x Chinchilla on Route-B data",
    )


if __name__ == "__main__":
    main()
