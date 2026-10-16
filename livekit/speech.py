"""SLNG speech to text and text to speech in a LiveKit agent. Your LLM stays where it is.

The STT and TTS run on SLNG in your project's region. The LLM is LiveKit's OpenAI
plugin, called directly on your OpenAI key, with no Context Router.

Works with any SLNG API key, including a key from the "Call TTS/STT models API" path.

Run it against your microphone::

    uv run python speech.py console
"""

import os
import uuid

from dotenv import load_dotenv
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import openai, silero, slng

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


def build_session(session_id: str) -> AgentSession:
    """Build the voice pipeline for one call.

    Args:
        session_id: One ID per call, the same on every turn of that call.
    """
    stt = slng.STT(
        api_key=os.environ["SLNG_API_KEY"],
        slng_base_url=f"{REGION}.api.slng.ai",
        model="soniox/speech-ai:rt-v5",
        language="en",
        external_agent_id=AGENT_ID,
        external_session_id=session_id,
    )
    session = AgentSession(
        stt=stt,
        llm=openai.LLM(model=MODEL, api_key=os.environ["OPENAI_API_KEY"]),
        tts=slng.TTS(
            api_key=os.environ["SLNG_API_KEY"],
            slng_base_url=f"{REGION}.api.slng.ai",
            model="slng/fish/tts:s2.1-pro",
            voice="9a9cf47702da476aa4629e2506d4a857",  # Fish Audio reference_id
            language="en",
            external_agent_id=AGENT_ID,
            external_session_id=session_id,
        ),
        vad=silero.VAD.load(),
    )
    # Lets the STT finalize the caller's turn as soon as LiveKit detects its end.
    stt.attach_to_session(session)
    return session


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    """Run one call."""
    session_id = str(uuid.uuid4())  # one per call; your room or call ID works too
    session = build_session(session_id)
    await session.start(agent=Agent(instructions=INSTRUCTIONS), room=ctx.room)
    session.say(GREETING)


if __name__ == "__main__":
    cli.run_app(server)
