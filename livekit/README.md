# LiveKit templates

Three LiveKit agents on SLNG speech to text and text to speech. Each file is a complete agent with its own `AgentServer`, so copy the one you need.

| File | LLM |
| -- | -- |
| `speech.py` | LiveKit's OpenAI plugin on your `OPENAI_API_KEY`, no router. |
| `speech_router.py` | The Context Router, answering with the model in your project's **Think** section. |
| `speech_router_byok.py` | The Context Router, on the model and OpenAI key each request sends in `slng_config`. |

Requires Python 3.10 or later, `livekit-agents` 1.8.8, and `livekit-plugins-slng` 1.8.8.

## Run

```bash
cp .env.example .env
uv sync
uv run python smoke.py speech        # one spoken turn, no room, no microphone
uv run python speech.py console      # talk to it in your terminal
uv run python speech.py dev          # connect to your LiveKit server (LIVEKIT_* in .env)
```

## What to keep when you copy a template

- **`slng_base_url`** pins the region: `f"{REGION}.api.slng.ai"`. Without it the plugin uses `api.slng.ai`. The plugin has no `base_url` parameter: an unknown keyword is passed on as a model option, so the region would silently not apply.
- **`external_agent_id` and `external_session_id`** on `slng.STT` and `slng.TTS`, and the `X-Slng-Agent-Id` and `X-Slng-Session-Id` headers on the LLM. Use the same two values on all three.
- **`stt.attach_to_session(session)`** lets the STT finalize the caller's turn as soon as LiveKit detects its end.
- **`max_retries=0`** on the router's `AsyncOpenAI` client. The router fails over between models itself.
- **`extra_body`** carries the router's own fields: `slng_config` for inline BYOK, `template_variables` for `{{name}}` placeholders in the prompt.
- **`session.say(GREETING)`** speaks a fixed greeting without a model request.
