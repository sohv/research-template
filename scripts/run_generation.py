# generates one model response per prompt in a jsonl dataset and writes them with run metadata.
# uv run -m scripts.run_generation --dataset_path data/processed/prompts.jsonl --output_dir results/raw/250612_example_v1 --model_id claude-sonnet-4-6 --num_tasks 10 --seed 42

import asyncio
import json
import logging
from pathlib import Path

import anthropic
import simple_parsing

from src.generation.batch import run_batch
from src.metrics.io import write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


def load_prompts(dataset_path: str, num_tasks: int | None) -> list[dict]:
    lines = Path(dataset_path).read_text().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    return records[:num_tasks] if num_tasks else records


def main():
    config = simple_parsing.parse(Config, add_config_path_arg=True)
    assert config.dataset_path, "--dataset_path is required, there is no default"

    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    config_path = write_config_json(config, output_dir)
    LOGGER.info(f"wrote run metadata to {config_path}")

    records = load_prompts(config.dataset_path, config.num_tasks)
    LOGGER.info(f"loaded {len(records)} prompts from {config.dataset_path}")

    client = anthropic.AsyncAnthropic()
    responses = asyncio.run(run_batch(client, [r["prompt"] for r in records], config.model_id))
    outputs = [{"id": r["id"], "prompt": r["prompt"], "response": resp} for r, resp in zip(records, responses)]

    path = write_jsonl(outputs, output_dir / "outputs.jsonl")
    print(f"Results saved to {path}")
    print(
        f"Score with: uv run -m scripts.run_metrics --dataset_path {path} "
        f"--output_dir {output_dir} --model_id {config.model_id} --seed {config.seed}"
    )


main()
