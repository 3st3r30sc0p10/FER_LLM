"""
Emotion-Driven Generative Language Interface — Main pipeline.
Continuous loop: Webcam → Emotion → Buffer → Grammar/Functions → LLM → Display.
"""

import argparse
import sys
import time
import cv2
from collections import deque
from typing import Optional

from webcam_capture import open_camera, read_frame, release_camera
from emotion_mapper import (
    emotion_to_function_sequence,
    build_prompt_from_functions,
    build_prompt_from_grammar,
    emotion_to_grammar_sequence,
)
from llm_generator import generate, get_dukegpt_url


# Default buffer size for emotional flow over time
DEFAULT_BUFFER_SIZE = 5


def analyze_emotion(frame, detector_backend: str = "opencv"):
    """
    Run DeepFace emotion analysis on a single frame. Returns dominant emotion.
    Expects BGR frame (OpenCV); converts to RGB for DeepFace.
    Default detector is opencv (works without MediaPipe; use --detector mediapipe if you have mediapipe<0.10.31).
    """
    from deepface import DeepFace

    # OpenCV captures BGR; DeepFace models typically expect RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = DeepFace.analyze(
        frame_rgb,
        actions=["emotion"],
        enforce_detection=False,
        detector_backend=detector_backend,
    )
    return result[0]["dominant_emotion"]


def run_pipeline(
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    backend: str = "dukegpt",
    use_grammar: bool = False,
    device_id: int = 0,
    detector_backend: str = "opencv",
    dukegpt_url: Optional[str] = None,
) -> None:
    """
    Run the full pipeline in a continuous loop.
    - buffer_size: number of recent emotions to form structure sequence
    - backend: "dukegpt" or "openai"
    - use_grammar: if True use POS mapping; if False use Jakobsonian functions
    """
    if backend == "dukegpt":
        url = get_dukegpt_url(dukegpt_url)
        print("DukeGPT backend: connecting to", url, file=sys.stderr)
    cap = open_camera(device_id)
    emotion_buffer = deque(maxlen=buffer_size)
    current_text = "Looking for a face..."
    last_emotion_error = None  # show user why detection might be failing
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
                    emotion = analyze_emotion(frame, detector_backend=detector_backend)
                    emotion_buffer.append(emotion)
                    last_emotion_error = None
                except Exception as e:
                    last_emotion_error = str(e)
                    print("Emotion detection error:", e, file=sys.stderr)
                    # Still allow pipeline to run with previous buffer if any

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
                    gen_kwargs = {"backend": backend}
                    if backend == "dukegpt" and dukegpt_url is not None:
                        gen_kwargs["dukegpt_url"] = dukegpt_url
                    current_text = generate(prompt, **gen_kwargs)
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

            # Show emotion buffer or status
            if emotion_buffer:
                buf_str = " → ".join(emotion_buffer)
                cv2.putText(
                    display_frame, buf_str, (50, display_frame.shape[0] - 30),
                    font, 0.5, (200, 200, 200), 1
                )
            elif last_emotion_error:
                # Short error hint so user knows why no face is detected
                err_short = (last_emotion_error[:60] + "..") if len(last_emotion_error) > 60 else last_emotion_error
                cv2.putText(
                    display_frame, err_short, (50, display_frame.shape[0] - 30),
                    font, 0.45, (0, 180, 255), 1
                )
            else:
                cv2.putText(
                    display_frame, "Position your face in frame", (50, display_frame.shape[0] - 30),
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
        choices=["dukegpt", "openai"],
        default="dukegpt",
        help="LLM backend: dukegpt (default) or openai",
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
    parser.add_argument(
        "--detector",
        choices=["opencv", "mediapipe", "ssd", "retinaface", "mtcnn"],
        default="opencv",
        help="Face detector backend (default: opencv; mediapipe needs mediapipe<0.10.31)",
    )
    parser.add_argument(
        "--dukegpt-url",
        type=str,
        default=None,
        metavar="URL",
        help="DukeGPT proxy base URL (e.g. http://localhost:3001 or http://server:3001). Default from DUKEGPT_API_URL.",
    )
    args = parser.parse_args()
    run_pipeline(
        buffer_size=args.buffer_size,
        backend=args.backend,
        use_grammar=args.grammar,
        device_id=args.camera,
        detector_backend=args.detector,
        dukegpt_url=args.dukegpt_url,
    )


if __name__ == "__main__":
    main()
