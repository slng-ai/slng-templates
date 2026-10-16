# SLNG templates

Working voice agents that keep your own orchestrator, LiveKit or Pipecat, and run speech to text, text to speech, and the Context Router on SLNG. Copy the one that matches your setup.

| Template | LiveKit | Pipecat | LLM | Key it needs |
| -- | -- | -- | -- | -- |
| SLNG speech | [`livekit/speech.py`](livekit/speech.py) | [`pipecat/speech.py`](pipecat/speech.py) | Your OpenAI key, called directly | Any SLNG key |
| SLNG speech + Context Router, project model | [`livekit/speech_router.py`](livekit/speech_router.py) | [`pipecat/speech_router.py`](pipecat/speech_router.py) | The model chosen in your project's **Think** section | Execution Layer project key |
| SLNG speech + Context Router, inline BYOK | [`livekit/speech_router_byok.py`](livekit/speech_router_byok.py) | [`pipecat/speech_router_byok.py`](pipecat/speech_router_byok.py) | Your OpenAI key, sent per request in `slng_config` | Execution Layer project key |

Every template runs Soniox for speech to text and Fish Audio for text to speech in `eu-north` by default. The full walkthrough is in the docs: [Set up a project for your own code](https://docs.slng.ai/guides/get-started/create-a-project/your-own-code).

## Pick the right API key

SLNG has three kinds of API key. They are not interchangeable.

| Key | Where you create it | What it can call |
| -- | -- | -- |
| Models only | **New project** > **Call TTS/STT models API** > **Generate key** | Speech to text and text to speech models. Not the Context Router. |
| Agents managed by SLNG | **New project** > **Build an agent**, then the project's **Project Vault** | Your agent deployed on SLNG. |
| Your own agents | **New project** > **Improve my agent** > LiveKit, Pipecat, or Custom, then **Project Vault** > **SLNG API key** | Speech models and the Context Router, for the layers your project includes. |

These templates are for the third kind. `speech.py` also works with a models-only key. The two router templates need a key from an Execution Layer project, or the router answers `403 project_mode_forbidden`.

## Set up the project

1. In [app.slng.ai](https://app.slng.ai), click **New project**, choose **Improve my agent**, and pick **LiveKit** or **Pipecat**.
2. Choose what runs on SLNG. **Voice in, voice out** runs speech to text and text to speech on SLNG and keeps your LLM. **Use your orchestrator, we handle the rest** adds SLNG LLMs.
3. Pick the **Region**. The project's API keys only work there, so set `SLNG_REGION` to the same region's code. For example, **Netherlands** is `eu-north` and **United Kingdom** is `gb`. Codes are on the [regions map](https://docs.slng.ai/guides/regions/regions-map).
4. For `speech_router.py`, open **Execution Layer** > **Think** and choose a model: one from the **SLNG models** tab, or your own from the **BYOK** tab (**Add new key** registers it). `speech_router_byok.py` skips this step: it sends the model and your key with every request.
5. Open **Project Vault** > **SLNG API key** > **Create API key**. Copy the key: you only see it once.

## Agent ID and session ID

Every template sends two IDs with each request, to speech to text, the LLM, and text to speech:

- **Agent ID** (`SLNG_AGENT_ID`): one stable value per agent, such as `wayne-concierge-v1`. It scopes the Context Router cache. Give the agent a new ID when you change its prompt in a meaningful way, or answers cached under the old prompt stay in scope.
- **Session ID**: one value per call, the same on every turn of that call. The templates make a UUID when the call starts. Your room, call, or conversation ID works too.

The router rejects a request without them with `400 missing_client_id`. On the speech plugins they are `external_agent_id` and `external_session_id`; on the router they are the `X-Slng-Agent-Id` and `X-Slng-Session-Id` headers.

## Run one

```bash
cd livekit            # or: cd pipecat
cp .env.example .env  # then fill in SLNG_API_KEY, SLNG_REGION, and OPENAI_API_KEY
uv sync
uv run python smoke.py speech   # one spoken turn, no microphone needed
```

`smoke.py` builds the template's pipeline and runs one turn: it speaks a caller line, transcribes it, asks the LLM, and speaks the reply. Pass `speech_router` or `speech_router_byok` to check the others. Then talk to it:

```bash
uv run python speech.py console     # LiveKit, in your terminal
uv run python speech.py -t webrtc   # Pipecat, then open http://localhost:7860
```

See [`livekit/README.md`](livekit/README.md) and [`pipecat/README.md`](pipecat/README.md) for each framework's details.

## Errors you might see

| Error | Cause |
| -- | -- |
| `403 project_mode_forbidden` | The key is not from an Execution Layer project. Create one under **Improve my agent**. |
| `401` | `SLNG_REGION` is not the region the key's project was created in. |
| `400 missing_client_id` or `invalid_client_id` | The agent or session ID is missing, or has a space, comma, pipe, or brace. |
| `422 missing_template_variables` | The prompt has a `{{name}}` placeholder with no value in `template_variables`. |
