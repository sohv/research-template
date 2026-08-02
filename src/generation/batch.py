import asyncio
import logging

from src.generation.cache import cached_llm_call

LOGGER = logging.getLogger(__name__)


async def run_batch(client, prompts: list[str], model: str, max_concurrent: int = 20) -> list[str]:
    sem = asyncio.Semaphore(max_concurrent)

    async def call(prompt: str) -> str:
        async with sem:
            return await cached_llm_call(client, model, [{"role": "user", "content": prompt}])

    LOGGER.info(f"dispatching {len(prompts)} calls to {model} at concurrency {max_concurrent}")
    return await asyncio.gather(*[call(p) for p in prompts])
