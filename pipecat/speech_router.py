"""SLNG speech to text, text to speech, and the Context Router in a Pipecat bot.

The router answers with the model you chose in your project's Think section: an SLNG
model, or an LLM key you saved on the BYOK page. Nothing about the model is in this
code, so you can change it in the dashboard without a deploy.

Needs a key from a project on the Execution Layer ("Improve my agent" > Pipecat),
created in the same region as ``SLNG_REGION``.

Run it, then open http://localhost:7860 and talk::

    uv run python speech_router.py -t webrtc
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
# One stable ID per agent. It scopes the router cache: change it when you change
# the prompt in a meaningful way, for example wayne-concierge-v2.
AGENT_ID = os.getenv("SLNG_AGENT_ID", "wayne-concierge-v1")
# slng/auto answers with the model chosen in your project's Think section. Send it
# whenever the request carries no slng_config: the router owns the model choice.
MODEL = "slng/auto"

# {{company_name}} and {{caller_name}} are template variables: the router fills them
# from the template_variables each request sends. The cached answer keeps the
# placeholders, so callers with different names can share it.
INSTRUCTIONS = (
    "You are the front desk of {{company_name}}, speaking with {{caller_name}}. "
    "Keep replies short: they are spoken aloud."
)
GREETING = "Hi, how can I help you today?"

TRANSPORTS = {"webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)}


def call_variables(caller_name: str) -> dict[str, str]:
    """Return this call's values for the placeholders in INSTRUCTIONS.

    Send values that are spoken or echoed, like names. Never send a value that changes
    what the answer says, such as the reply language: write that into the prompt.
    A placeholder with no value makes the router answer 422.
    """
    return {"company_name": "Wayne Enterprises", "caller_name": caller_name}


def services(
    session_id: str, variables: dict[str, str]
) -> tuple[SlngSTTService, OpenAILLMService, SlngTTSService]:
    """Build the STT, LLM, and TTS for one call.

    Args:
        session_id: One ID per call, the same on every turn of that call.
        variables: This call's template variables, from call_variables().
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
        api_key=os.environ["SLNG_API_KEY"],
        base_url=f"https://{REGION}.context-router.slng.ai/v1",
        default_headers={
            "X-Slng-Agent-Id": AGENT_ID,  # cache scope
            "X-Slng-Session-Id": session_id,  # one per call
        },
        settings=OpenAILLMService.Settings(
            model=MODEL,
            system_instruction=INSTRUCTIONS,
            # Pipecat passes extra to the OpenAI client; extra_body adds body fields.
            extra={"extra_body": {"template_variables": variables}},
        ),
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
    stt, llm, tts = services(session_id, call_variables("Bruce"))  # from your CRM or dispatch
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
        await worker.queue_frame(TTSSpeakFrame(GREETING))  # never reaches the router

    @transport.event_handler("on_client_disconnected")
    async def hang_up(transport: Any, client: Any) -> None:
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
