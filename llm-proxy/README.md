# LLM proxy

**Architecture:** Python app → this proxy → Duke API. The app never talks to Duke directly; it only calls this server. The Duke URL and API key live only here. Only the machine running this proxy needs to reach Duke (Duke VPN on that machine if required).

## Quick start

```bash
npm install
npm start
```

Listens on **http://localhost:3001**. Then run the emotion interface with:

```bash
python emotion_interface.py --dukegpt-url http://localhost:3001
```

## Env (optional)

- `PORT` — default `3001`
- `DUKE_LLM_URL` — default `https://litellm.oit.duke.edu/v1/chat/completions`
- `DUKE_LLM_KEY` — Duke API key (default set in code)

## Endpoint

- **POST /proxy/llm** — body: `{ "model", "messages" }` (OpenAI chat format). Proxies to Duke API and returns the same JSON.
