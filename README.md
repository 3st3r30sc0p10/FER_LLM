# Emotion-Driven Generative Language Interface

A real-time system that turns **webcam emotion** into **poetic language** via structural mapping. Emotion becomes syntax; language becomes embodied.

## Architecture

```
[ Webcam ] → [ Emotion Recognition ] → [ Emotion → Language Mapper ] → [ LLM ] → [ Text Output ]
     OpenCV         DeepFace                  Jakobsonian / POS              DukeGPT / OpenAI    OpenCV window
```

**DukeGPT:** This app never talks to Duke directly. Flow is **app → proxy → Duke API**.

| Role | Talks to | Needs Duke VPN? |
|------|----------|-----------------|
| This app (Python) | Proxy only (`http://localhost:3001`) | No |
| Proxy (`llm-proxy`) | Duke LiteLLM API | Only if Duke restricts by network |

The proxy holds the Duke URL and API key; it forwards requests and returns Duke’s response. So you only need Duke VPN (or Duke network) on the **machine running the proxy**, not on the machine running the emotion app.

Each module is independent and replaceable.

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd FER_LLM
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. LLM backend (choose one)

- **DukeGPT (default)**  
  The app calls the **proxy** only (default `http://localhost:3001`). The proxy talks to Duke’s API and holds the API key. You do **not** need Duke VPN on the machine running the emotion app; only the machine running the proxy must be able to reach Duke (use Duke VPN there if required).

  1. **Start the proxy** (in its own terminal; keep it running):
     ```bash
     cd llm-proxy && npm install && npm start
     ```
  2. **Run the emotion app**:
     ```bash
     python emotion_interface.py
     ```
     No `--dukegpt-url` needed when using default localhost proxy.

- **OpenAI**  
  Set your API key and use `--backend openai`:
  ```bash
  export OPENAI_API_KEY=sk-...
  python emotion_interface.py --backend openai
  ```

## Run

Default: Jakobsonian function mapping + DukeGPT via proxy. Start the proxy first (see above), then:

```bash
python emotion_interface.py
```

Options:

- `--buffer-size 5` — number of recent emotions in the sequence (default: 5)
- `--backend dukegpt|openai` — which LLM to use
- `--dukegpt-url URL` — proxy base URL (default `http://localhost:3001`; use if proxy runs elsewhere)
- `--grammar` — use part-of-speech mapping instead of Jakobsonian functions
- `--camera 0` — camera device ID

Examples:

```bash
# Default: DukeGPT, Jakobsonian
python emotion_interface.py

# OpenAI, part-of-speech structure
python emotion_interface.py --backend openai --grammar

# Longer emotional sequence
python emotion_interface.py --buffer-size 7
```

Press **ESC** to quit.

## Project layout

| File | Role |
|------|------|
| `emotion_interface.py` | Main pipeline: webcam → emotion buffer → prompt → LLM → display |
| `webcam_capture.py` | OpenCV webcam capture |
| `emotion_mapper.py` | Emotion → grammar (POS) and Emotion → Jakobsonian function mapping + prompt builder |
| `llm_generator.py` | DukeGPT and OpenAI text generation |

## Emotion mappings

**Jakobsonian (default)**  
Emotions map to communicative functions that shape the sentence (poetic, emotive, conative, referential, phatic, metalingual, syntagmatic).

**Grammar (--grammar)**  
Emotions map to parts of speech (adjective, verb, noun, adverb, etc.); the LLM is asked to produce a sentence following that structure.

## Face / emotion detection

- The app uses **MediaPipe** by default for face detection. If no face or emotion is detected, try:
  - Good lighting and face the camera clearly.
  - Use OpenCV detector: `python emotion_interface.py --detector opencv`.
  - Check the bottom of the window (or terminal) for the reported error.
- First run may download DeepFace emotion models; ensure you have internet and wait for it to finish.

## Requirements

- Python 3.9+
- Webcam
- 8GB RAM recommended; GPU optional. DeepFace runs on CPU.

## Future ideas

- Flask or projection UI
- Temporal emotion tracking and intensity
- ONNX FER model (e.g. for edge devices)
- Multi-person detection
