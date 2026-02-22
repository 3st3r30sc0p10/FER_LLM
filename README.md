# Emotion-Driven Generative Language Interface

A real-time system that turns **webcam emotion** into **poetic language** via structural mapping. Emotion becomes syntax; language becomes embodied.

## Architecture

```
[ Webcam ] → [ Emotion Recognition ] → [ Emotion → Language Mapper ] → [ LLM ] → [ Text Output ]
     OpenCV         DeepFace                  Jakobsonian / POS              Ollama / OpenAI    OpenCV window
```

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

- **Ollama (local)**  
  Install [Ollama](https://ollama.com), then:
  ```bash
  ollama run mistral
  ```
  Keep it running in another terminal.

- **OpenAI**  
  Set your API key and use `--backend openai`:
  ```bash
  export OPENAI_API_KEY=sk-...
  python emotion_interface.py --backend openai
  ```

## Run

Default: Jakobsonian function mapping + Ollama.

```bash
python emotion_interface.py
```

Options:

- `--buffer-size 5` — number of recent emotions in the sequence (default: 5)
- `--backend ollama|openai` — which LLM to use
- `--grammar` — use part-of-speech mapping instead of Jakobsonian functions
- `--camera 0` — camera device ID

Examples:

```bash
# Local Mistral, Jakobsonian (default)
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
| `llm_generator.py` | Ollama and OpenAI text generation |

## Emotion mappings

**Jakobsonian (default)**  
Emotions map to communicative functions that shape the sentence (poetic, emotive, conative, referential, phatic, metalingual, syntagmatic).

**Grammar (--grammar)**  
Emotions map to parts of speech (adjective, verb, noun, adverb, etc.); the LLM is asked to produce a sentence following that structure.

## Requirements

- Python 3.9+
- Webcam
- 8GB RAM recommended; GPU optional. DeepFace runs on CPU.

## Future ideas

- Flask or projection UI
- Temporal emotion tracking and intensity
- ONNX FER model (e.g. for edge devices)
- Multi-person detection
