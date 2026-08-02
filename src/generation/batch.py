import asyncio
import logging

from src.generation.cache import cached_llm_call

LOGGER = logging.getLogger(__name__)


async def run_batch(client, prompts: list[str], model: str, max_concurrent: int = 20) -> list[str | None]:
    """Call the model once per prompt, at most max_concurrent in flight.

    Results line up with prompts by index. A call that fails comes back as None in its own slot
    rather than taking the batch down with it, so one bad response twelve hours into an overnight
    sweep doesn't discard everything before it. Callers either filter the Nones out or record them
    as failures — never treat a None as a response.
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def call(index: int, prompt: str) -> str | None:
        async with sem:
            try:
                return await cached_llm_call(client, model, [{"role": "user", "content": prompt}])
            except Exception as error:
                LOGGER.error(f"prompt {index} failed: {error!r}")
                return None

    LOGGER.info(f"dispatching {len(prompts)} calls to {model} at concurrency {max_concurrent}")
    results = await asyncio.gather(*[call(i, p) for i, p in enumerate(prompts)])
    n_failed = sum(1 for result in results if result is None)
    if n_failed:
        LOGGER.error(f"{n_failed}/{len(prompts)} calls failed, returned as None")
    return results
