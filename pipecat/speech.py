"""SLNG speech to text and text to speech in a Pipecat bot. Your LLM stays where it is.

The STT and TTS run on SLNG in your project's region. The LLM is Pipecat's OpenAI
service, called directly on your OpenAI key, with no Context Router.

Works with any SLNG API key, including a key from the "Call TTS/STT models API" path.

Run it, then open http://localhost:7860 and talk::

    uv run python speech.py -t webrtc
"""

import os
import uuid
from typing import Any

from dotenv import load_dotenv
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner
from pipecat_slng import SlngSTTService, SlngTTSService

load_dotenv()

# The region of the SLNG project your key belongs to, as a gateway code.
REGION = os.getenv("SLNG_REGION", "eu-north")
# One stable ID per agent. SLNG groups usage, calls, and cache by it.
AGENT_ID = os.getenv("SLNG_AGENT_ID", "wayne-concierge-v1")
MODEL = os.getenv("LLM_MODEL", "gpt-5.6-luna")

INSTRUCTIONS = (
    "You are the front desk of Wayne Enterprises. Keep replies short: they are spoken aloud."
)
GREETING = "Hi, how can I help you today?"

TRANSPORTS = {"webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)}


def services(session_id: str) -> tuple[SlngSTTService, OpenAILLMService, SlngTTSService]:
    """Build the STT, LLM, and TTS for one call.

    Args:
        session_id: One ID per call, the same on every turn of that call.
    """
    stt = SlngSTTService(
        api_key=os.environ["SLNG_API_KEY"],
        world_part=REGION,
        model="soniox/speech-ai:rt-v5",
        language=Language.EN,
        external_agent_id=AGENT_ID,
        external_session_id=session_id,
    )
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(model=MODEL, system_instruction=INSTRUCTIONS),
    )
    tts = SlngTTSService(
        api_key=os.environ["SLNG_API_KEY"],
        world_part=REGION,
        model="slng/fish/tts:s2.1-pro",
        voice="9a9cf47702da476aa4629e2506d4a857",  # Fish Audio reference_id
        language=Language.EN,
        external_agent_id=AGENT_ID,
        external_session_id=session_id,
    )
    return stt, llm, tts


async def bot(runner_args: RunnerArguments) -> None:
    """Run one call."""
    session_id = str(uuid.uuid4())  # one per call; your call or room ID works too
    transport = await create_transport(runner_args, TRANSPORTS)
    stt, llm, tts = services(session_id)
    aggregators = LLMContextAggregatorPair(
        LLMContext(), user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )
    worker = PipelineWorker(pipeline, conversation_id=session_id)

    @transport.event_handler("on_client_connected")
    async def greet(transport: Any, client: Any) -> None:
        await worker.queue_frame(TTSSpeakFrame(GREETING))

    @transport.event_handler("on_client_disconnected")
    async def hang_up(transport: Any, client: Any) -> None:
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
