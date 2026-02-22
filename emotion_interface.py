"""
Emotion-Driven Generative Language Interface — Main pipeline.
Continuous loop: Webcam → Emotion → Buffer → Grammar/Functions → LLM → Display.
"""

import argparse
import time
import cv2
from collections import deque

from webcam_capture import open_camera, read_frame, release_camera
from emotion_mapper import (
    emotion_to_function_sequence,
    build_prompt_from_functions,
    build_prompt_from_grammar,
    emotion_to_grammar_sequence,
)
from llm_generator import generate


# Default buffer size for emotional flow over time
DEFAULT_BUFFER_SIZE = 5


def analyze_emotion(frame) -> str:
    """Run DeepFace emotion analysis on a single frame. Returns dominant emotion."""
    from deepface import DeepFace

    result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
    return result[0]["dominant_emotion"]


def run_pipeline(
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    backend: str = "ollama",
    use_grammar: bool = False,
    device_id: int = 0,
) -> None:
    """
    Run the full pipeline in a continuous loop.
    - buffer_size: number of recent emotions to form structure sequence
    - backend: "ollama" or "openai"
    - use_grammar: if True use POS mapping; if False use Jakobsonian functions
    """
    cap = open_camera(device_id)
    emotion_buffer = deque(maxlen=buffer_size)
    current_text = "Looking for a face..."
    last_emotion_time = 0
    emotion_interval = 0.5  # seconds between emotion analyses (avoid overload)
    last_llm_time = 0
    llm_interval = 2.0  # seconds between LLM calls

    try:
        while True:
            ret, frame = read_frame(cap)
            if not ret or frame is None:
                break

            now = time.monotonic()

            # Run emotion detection at fixed interval
            if now - last_emotion_time >= emotion_interval:
                last_emotion_time = now
                try:
                    emotion = analyze_emotion(frame)
                    emotion_buffer.append(emotion)
                except Exception as e:
                    # Keep previous text on failure
                    pass

            # Build structure and call LLM at longer interval
            if emotion_buffer and (now - last_llm_time >= llm_interval):
                last_llm_time = now
                try:
                    if use_grammar:
                        structure = emotion_to_grammar_sequence(list(emotion_buffer))
                        prompt = build_prompt_from_grammar(structure)
                    else:
                        functions = emotion_to_function_sequence(list(emotion_buffer))
                        prompt = build_prompt_from_functions(functions)
                    current_text = generate(prompt, backend=backend)
                except Exception as e:
                    current_text = f"(generation error: {e})"

            # Overlay text on frame (wrap long lines for readability)
            display_frame = frame.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            color = (255, 255, 255)
            outline = (0, 0, 0)
            y0 = 40
            line_height = 28
            max_width = display_frame.shape[1] - 80

            # Word wrap
            words = current_text.split()
            lines = []
            current_line = []
            for w in words:
                test = " ".join(current_line + [w])
                (tw, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
                if tw <= max_width:
                    current_line.append(w)
                else:
                    if current_line:
                        lines.append(" ".join(current_line))
                    current_line = [w]
            if current_line:
                lines.append(" ".join(current_line))

            for i, line in enumerate(lines[:4]):  # max 4 lines
                y = y0 + i * line_height
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    cv2.putText(
                        display_frame, line, (51 + dx, y + dy), font, font_scale, outline, thickness + 1
                    )
                cv2.putText(display_frame, line, (50, y), font, font_scale, color, thickness)

            # Show emotion buffer
            if emotion_buffer:
                buf_str = " → ".join(emotion_buffer)
                cv2.putText(
                    display_frame, buf_str, (50, display_frame.shape[0] - 30),
                    font, 0.5, (200, 200, 200), 1
                )

            cv2.imshow("Emotion → Language", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    finally:
        release_camera(cap)


def main():
    parser = argparse.ArgumentParser(
        description="Emotion-Driven Generative Language Interface"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=DEFAULT_BUFFER_SIZE,
        help="Number of emotions in sequence (default: 5)",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "openai"],
        default="ollama",
        help="LLM backend (default: ollama)",
    )
    parser.add_argument(
        "--grammar",
        action="store_true",
        help="Use part-of-speech mapping instead of Jakobsonian functions",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)",
    )
    args = parser.parse_args()
    run_pipeline(
        buffer_size=args.buffer_size,
        backend=args.backend,
        use_grammar=args.grammar,
        device_id=args.camera,
    )


if __name__ == "__main__":
    main()
