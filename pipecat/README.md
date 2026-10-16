# Pipecat templates

Three Pipecat bots on SLNG speech to text and text to speech. Each file is a complete bot with its own `bot()` function for the Pipecat runner, so copy the one you need.

| File | LLM |
| -- | -- |
| `speech.py` | Pipecat's `OpenAILLMService` on your `OPENAI_API_KEY`, no router. |
| `speech_router.py` | The Context Router, answering with the model in your project's **Think** section. |
| `speech_router_byok.py` | The Context Router, on the model and OpenAI key each request sends in `slng_config`. |

Requires Python 3.11 or later, `pipecat-ai` 1.12, and `pipecat-slng` 0.7.

## Run

```bash
cp .env.example .env
uv sync
uv run python smoke.py speech        # one spoken turn, no transport, no microphone
uv run python speech.py -t webrtc    # then open http://localhost:7860 and talk
```

## What to keep when you copy a template

- **`world_part`** pins the region on `SlngSTTService` and `SlngTTSService`, as a gateway code such as `eu-north`. Use the same code in the router's base URL, `https://{region}.context-router.slng.ai/v1`.
- **`external_agent_id` and `external_session_id`** on both speech services, and the `X-Slng-Agent-Id` and `X-Slng-Session-Id` headers on the LLM in `default_headers`. Use the same two values on all three.
- **`settings.extra={"extra_body": {...}}`** carries the router's own fields. Pipecat passes `extra` straight to the OpenAI client's `create()`, which only takes unknown body fields inside `extra_body`. `slng_config` is for inline BYOK, `template_variables` fills `{{name}}` placeholders in the prompt.
- **`TTSSpeakFrame(GREETING)`** on `on_client_connected` speaks a fixed greeting without a model request.
