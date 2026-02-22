"""
Module 3: Emotion → Language Structure Mapping
Maps emotions to grammatical roles and to Jakobsonian communicative functions.
Emotion becomes syntax; language becomes embodied.
"""

from typing import List

# Part-of-speech mapping (original plan)
EMOTION_TO_GRAMMAR = {
    "happy": "adjective",
    "sad": "verb",
    "angry": "noun",
    "fear": "adverb",
    "surprise": "interjection",
    "neutral": "article",
    "disgust": "preposition",
}

# Jakobsonian structural mapping (communicative functions)
EMOTION_TO_FUNCTION = {
    "happy": "poetic",
    "sad": "emotive",
    "angry": "conative",
    "fear": "referential",
    "surprise": "phatic",
    "disgust": "metalingual",
    "neutral": "syntagmatic",
}

FUNCTION_INSTRUCTIONS = {
    "poetic": "foreground metaphor and rhythm",
    "emotive": "include subjective interior tone",
    "conative": "address someone directly or use imperative",
    "referential": "describe concrete context",
    "phatic": "include an interruption or opening marker",
    "metalingual": "reflect on language itself",
    "syntagmatic": "maintain grammatical coherence",
}

# DeepFace emotion set (canonical order for validation)
SUPPORTED_EMOTIONS = frozenset(EMOTION_TO_GRAMMAR.keys())


def emotion_to_grammar_sequence(emotions: List[str]) -> List[str]:
    """Convert a list of emotions to a list of grammatical roles."""
    return [EMOTION_TO_GRAMMAR.get(e, "adjective") for e in emotions if e in EMOTION_TO_GRAMMAR]


def emotion_to_function_sequence(emotions: List[str]) -> List[str]:
    """Convert a list of emotions to Jakobsonian function sequence."""
    return [EMOTION_TO_FUNCTION.get(e, "syntagmatic") for e in emotions if e in EMOTION_TO_FUNCTION]


def build_prompt_from_grammar(structure_sequence: List[str]) -> str:
    """Build LLM prompt from grammatical structure (article, adjective, noun, ...)."""
    structure_str = " ".join(structure_sequence) if structure_sequence else "adjective noun verb"
    return f"""Create a poetic sentence using this grammatical structure:
{structure_str}

Rules:
- Respect grammatical order.
- Be poetic.
- Be abstract.
Output only the sentence, no explanation."""


def build_prompt_from_functions(function_sequence: List[str]) -> str:
    """Build LLM prompt from Jakobsonian function sequence (recommended)."""
    if not function_sequence:
        function_sequence = ["syntagmatic"]
    rules = "\n".join(f"- {FUNCTION_INSTRUCTIONS.get(f, f'apply: {f}')}" for f in function_sequence)
    return f"""Create one coherent poetic sentence.
Follow these structural rules (in order):
{rules}

Output only the sentence, no explanation."""
