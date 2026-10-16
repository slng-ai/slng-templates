"""Run one spoken turn through a template's services, with no transport and no microphone.

It builds the template's STT, LLM, and TTS, then:

1. speaks a caller line with the TTS,
2. transcribes that audio with the STT,
3. asks the LLM for a reply, printing the router's ``x-slng-*`` headers,
4. speaks the reply with the TTS.

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

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

CALLER_LINE = "Hello, what are your opening hours on Saturday?"
RATE = 16000


class Collector(FrameProcessor):
    """Keep every frame that reaches the end of the pipeline, and note when one is done."""

    def __init__(self, done: type[Frame]) -> None:
        super().__init__()
        self.frames: list[Frame] = []
        self.finished = asyncio.Event()
        self._done = done

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        if isinstance(frame, self._done | ErrorFrame):
            self.finished.set()
        await self.push_frame(frame, direction)


async def run(
    service: FrameProcessor, frames: list[Frame], done: type[Frame], pace: bool = False
) -> list[Frame]:
    """Send frames through one service until a ``done`` frame comes out."""
    collector = Collector(done)
    worker = PipelineWorker(
        Pipeline([service, collector]),
        params=PipelineParams(audio_in_sample_rate=RATE, audio_out_sample_rate=RATE),
        idle_timeout_secs=None,
        enable_rtvi=False,
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    await asyncio.sleep(1)  # let the service connect
    for frame in frames:
        await worker.queue_frame(frame)
        if pace and isinstance(frame, InputAudioRawFrame):
            await asyncio.sleep(len(frame.audio) / 2 / RATE)  # real time
    try:
        await asyncio.wait_for(collector.finished.wait(), timeout=30)
    finally:
        await worker.cancel()
        await running
    errors = [f for f in collector.frames if isinstance(f, ErrorFrame)]
    assert not errors, errors[0].error
    return collector.frames


async def speak(tts: FrameProcessor, text: str) -> bytes:
    """Return the 16 kHz audio the TTS makes for ``text``."""
    frames = await run(tts, [TTSSpeakFrame(text)], TTSStoppedFrame)
    return b"".join(f.audio for f in frames if isinstance(f, TTSAudioRawFrame))


async def transcribe(stt: FrameProcessor, audio: bytes) -> str:
    """Stream 16 kHz audio to the STT, end the turn, and return the transcript."""
    chunk = RATE // 50 * 2  # 20 ms
    silence = b"\0" * chunk
    frames: list[Frame] = [VADUserStartedSpeakingFrame()]
    frames += [
        InputAudioRawFrame(audio[i : i + chunk], RATE, 1) for i in range(0, len(audio), chunk)
    ]
    frames += [InputAudioRawFrame(silence, RATE, 1)] * 25
    frames.append(VADUserStoppedSpeakingFrame())
    out = await run(stt, frames, TranscriptionFrame, pace=True)
    return " ".join(f.text for f in out if isinstance(f, TranscriptionFrame)).strip()


def watch_router(llm: FrameProcessor) -> dict[str, str]:
    """Record the x-slng-* headers of the LLM's responses, when it talks to the router."""
    seen: dict[str, str] = {}
    hooks = getattr(getattr(getattr(llm, "_client", None), "_client", None), "event_hooks", None)
    if hooks is not None:

        async def hook(response) -> None:  # noqa: ANN001
            seen.update({k: v for k, v in response.headers.items() if k.startswith("x-slng-")})

        hooks["response"].append(hook)
    return seen


async def main(name: str) -> None:
    """Run one turn through the template called ``name``."""
    template = importlib.import_module(name)
    session_id = str(uuid.uuid4())
    # A Pipecat service runs in one pipeline, so each step gets fresh services.
    stt, llm, tts = template.services(session_id)
    headers = watch_router(llm)

    caller_audio = await speak(tts, CALLER_LINE)
    print(f"caller audio: {len(caller_audio) / 2 / RATE:.1f}s")
    assert caller_audio, "TTS returned no audio"

    heard = await transcribe(stt, caller_audio)
    print(f"STT heard:    {heard!r}")
    assert heard, "STT returned no transcript"

    context = LLMContext(
        messages=[
            {"role": "assistant", "content": template.GREETING},
            {"role": "user", "content": heard},
        ]
    )
    started = time.perf_counter()
    out = await run(llm, [LLMContextFrame(context)], LLMFullResponseEndFrame)
    reply = "".join(f.text for f in out if isinstance(f, LLMTextFrame)).strip()
    print(f"LLM replied:  {reply!r} ({time.perf_counter() - started:.2f}s)")
    if headers:
        print(f"router:       {headers}")
    assert reply, "LLM returned no text"
    assert "{{" not in reply, "a template variable was not filled"

    reply_audio = await speak(template.services(session_id)[2], reply)
    print(f"TTS audio:    {len(reply_audio) / 2 / RATE:.1f}s")
    assert reply_audio, "TTS returned no audio"
    print(f"OK: {name}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "speech"))
