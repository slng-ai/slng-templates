"""Run one spoken turn through a template's pipeline, with no room and no microphone.

It builds the template's ``AgentSession``, then:

1. speaks a caller line with the session's TTS,
2. transcribes that audio with the session's STT,
3. asks the session's LLM for a reply, printing the router's ``x-slng-*`` headers,
4. speaks the reply with the session's TTS,
5. for a router template, asks the caller's name, which only template_variables give.

Usage::

    uv run python smoke.py speech
    uv run python smoke.py speech_router
    uv run python smoke.py speech_router_byok
"""

import asyncio
import importlib
import sys
import time
import uuid

from livekit import rtc
from livekit.agents import llm, stt
from livekit.agents.utils import http_context

CALLER_LINE = "Hello, what are your opening hours on Saturday?"
NAME_QUESTION = "What is my name? Answer with the name only."


async def speak(session_tts, text: str) -> list[rtc.AudioFrame]:  # noqa: ANN001
    """Return the audio the TTS makes for ``text``."""
    stream = session_tts.stream()
    stream.push_text(text)
    stream.end_input()
    frames = [event.frame async for event in stream]
    await stream.aclose()
    return frames


def seconds(frames: list[rtc.AudioFrame]) -> float:
    """Return how long the frames play for."""
    return sum(f.samples_per_channel / f.sample_rate for f in frames)


async def transcribe(session_stt, frames: list[rtc.AudioFrame]) -> str:  # noqa: ANN001
    """Stream the frames to the STT and return the final transcript."""
    resampler = rtc.AudioResampler(frames[0].sample_rate, 16000)
    stream = session_stt.stream()
    pieces: list[str] = []

    async def read() -> None:
        async for event in stream:
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT and event.alternatives:
                pieces.append(event.alternatives[0].text)

    reader = asyncio.create_task(read())
    silence = rtc.AudioFrame(b"\0\0" * 1600, 16000, 1, 1600)  # 100 ms
    for frame in frames:
        for out in resampler.push(frame):
            stream.push_frame(out)
            await asyncio.sleep(out.samples_per_channel / 16000)  # real time
    for out in resampler.flush():
        stream.push_frame(out)
    for _ in range(15):
        stream.push_frame(silence)
        await asyncio.sleep(0.1)
    stream.end_input()
    await asyncio.wait_for(reader, timeout=20)
    return " ".join(p.strip() for p in pieces if p.strip())


def watch_router(session_llm) -> dict[str, str]:  # noqa: ANN001
    """Record the x-slng-* headers of the LLM's responses, when it talks to the router."""
    seen: dict[str, str] = {}
    client = getattr(session_llm, "_client", None)
    hooks = getattr(getattr(client, "_client", None), "event_hooks", None)
    if hooks is not None:

        async def hook(response) -> None:  # noqa: ANN001
            seen.update({k: v for k, v in response.headers.items() if k.startswith("x-slng-")})

        hooks["response"].append(hook)
    return seen


async def main(name: str) -> None:
    """Run one turn through the template called ``name``."""
    template = importlib.import_module(name)
    http_context._new_session_ctx()
    try:
        extra = (template.call_variables("Bruce"),) if hasattr(template, "call_variables") else ()
        session = template.build_session(str(uuid.uuid4()), *extra)
        headers = watch_router(session.llm)

        caller_audio = await speak(session.tts, CALLER_LINE)
        print(f"caller audio: {seconds(caller_audio):.1f}s")

        started = time.perf_counter()
        heard = await transcribe(session.stt, caller_audio)
        print(f"STT heard:    {heard!r}")
        assert heard, "STT returned no transcript"

        chat = llm.ChatContext()
        chat.add_message(role="system", content=template.INSTRUCTIONS)
        chat.add_message(role="assistant", content=template.GREETING)
        chat.add_message(role="user", content=heard)
        started = time.perf_counter()
        reply = ""
        async with session.llm.chat(chat_ctx=chat) as stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    reply += chunk.delta.content
        print(f"LLM replied:  {reply!r} ({time.perf_counter() - started:.2f}s)")
        if headers:
            print(f"router:       {headers}")
        assert reply.strip(), "LLM returned no text"
        assert "{{" not in reply, "a template variable was not filled"

        reply_audio = await speak(session.tts, reply)
        print(f"TTS audio:    {seconds(reply_audio):.1f}s")
        assert reply_audio, "TTS returned no audio"

        if extra:
            chat.add_message(role="assistant", content=reply)
            chat.add_message(role="user", content=NAME_QUESTION)
            answer = ""
            async with session.llm.chat(chat_ctx=chat) as stream:
                async for chunk in stream:
                    if chunk.delta and chunk.delta.content:
                        answer += chunk.delta.content
            print(f"name check:   {answer!r}")
            assert "Bruce" in answer, "the router did not fill {{caller_name}}"
        print(f"OK: {name}")
    finally:
        await http_context._close_http_ctx()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "speech"))
