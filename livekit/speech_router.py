"""SLNG speech to text, text to speech, and the Context Router in a LiveKit agent.

The router answers with the model you chose in your project's Think section: an SLNG
model, or an LLM key you saved on the BYOK page. Nothing about the model is in this
code, so you can change it in the dashboard without a deploy.

Needs a key from a project on the Execution Layer ("Improve my agent" > LiveKit),
created in the same region as ``SLNG_REGION``.

Run it against your microphone::

    uv run python speech_router.py console
"""

import os
import uuid

from dotenv import load_dotenv
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import openai, silero, slng
from openai import AsyncOpenAI

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


def call_variables(caller_name: str) -> dict[str, str]:
    """Return this call's values for the placeholders in INSTRUCTIONS.

    Send values that are spoken or echoed, like names. Never send a value that changes
    what the answer says, such as the reply language: write that into the prompt.
    A placeholder with no value makes the router answer 422.
    """
    return {"company_name": "Wayne Enterprises", "caller_name": caller_name}


def build_session(session_id: str, variables: dict[str, str]) -> AgentSession:
    """Build the voice pipeline for one call.

    Args:
        session_id: One ID per call, the same on every turn of that call.
        variables: This call's template variables, from call_variables().
    """
    router = AsyncOpenAI(
        base_url=f"https://{REGION}.context-router.slng.ai/v1",
        api_key=os.environ["SLNG_API_KEY"],
        max_retries=0,  # the router fails over between models itself
    )
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
        llm=openai.LLM(
            client=router,
            model=MODEL,
            extra_headers={
                "X-Slng-Agent-Id": AGENT_ID,  # cache scope
                "X-Slng-Session-Id": session_id,  # one per call
            },
            extra_body={"template_variables": variables},
        ),
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
    session = build_session(session_id, call_variables("Bruce"))  # from your CRM or dispatch
    await session.start(agent=Agent(instructions=INSTRUCTIONS), room=ctx.room)
    session.say(GREETING)  # a fixed greeting never reaches the router


if __name__ == "__main__":
    cli.run_app(server)
