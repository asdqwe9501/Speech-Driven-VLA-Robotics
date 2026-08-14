"""Mic -> faster-whisper -> English task string -> current_task.txt

Push-to-talk: press Enter to START recording, speak your command, press Enter
again to STOP. The clip is transcribed, the recognized color is normalized into
the exact task string the policy was trained on, and written to current_task.txt
so voice_run.py / lerobot-record can pick it up live. Enter-gated recording
avoids background noise and robot sounds triggering false commands.
"""

import argparse
import queue
import string
import sys
import threading
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
BLOCK_S = 0.03  # audio block size fed to the callback (30 ms)
MAX_RECORD_S = 12.0  # safety auto-stop if the user forgets to press Enter
MIN_RMS = 0.005  # below this the clip is treated as silence (avoids Whisper hallucinations)
COLORS = ("red", "black")  # the two colors the policy was trained on
# Exact trained task strings (verified from dataset6 meta) - must match verbatim
TASK_TEMPLATE = "grab the {color} ball and put it in the box"
STOP_TASK = "__STOP__"  # sentinel: record loop returns the arm home and waits
COMPLETE_TASK = "__COMPLETE__"  # sentinel: record loop ends the whole session (full shutdown)
DEFAULT_TASK_FILE = Path(__file__).parent / "current_task.txt"
FUZZY_CUTOFF = 0.8  # min string-similarity to accept a mispronounced word (high = fewer false matches)

# Word (English mishears + Korean) -> canonical color. match_color lowercases first.
ALIASES = {
    # English
    "red": "red", "read": "red", "rad": "red", "bread": "red", "rate": "red",
    "black": "black", "block": "black", "back": "black", "blak": "black",
    "blag": "black", "plaque": "black", "blackball": "black",
    # Korean (say these - far more accurate for a Korean speaker)
    "빨강": "red", "빨간": "red", "빨간색": "red", "빨강색": "red", "빨간공": "red",
    "빨강공": "red", "빨간볼": "red", "레드": "red",
    "검정": "black", "검은": "black", "검정색": "black", "검은색": "black", "까만": "black",
    "까망": "black", "검은공": "black", "검정공": "black", "검은볼": "black", "블랙": "black",
}

# stop / complete command words (English + Korean)
STOP_ALIASES = {"stop", "stopp", "stahp", "정지", "멈춰", "멈춤", "스톱", "스탑", "그만", "중지"}
COMPLETE_ALIASES = {"complete", "completed", "komplete", "완료", "끝", "끝내", "종료", "컴플리트", "컴플릿"}


def match_color(text: str) -> tuple[str | None, float, str, str]:
    """Decide the intended color from a (possibly noisy) transcript.

    Returns (color, score, matched_word, reason). reason is "ok" for a single
    confident color, "conflict" if both colors are heard (ambiguous -> refuse),
    or "none" if nothing clears FUZZY_CUTOFF. color is None unless reason=="ok".
    """
    words = [w.strip(string.punctuation) for w in text.lower().split()]
    hits: dict[str, tuple[float, str]] = {}  # color -> (best_score, word)
    for w in words:
        if not w:
            continue
        if w in ALIASES:  # exact word or known mishear
            c = ALIASES[w]
            if hits.get(c, (0.0, ""))[0] < 1.0:
                hits[c] = (1.0, w)
            continue
        for color in COLORS:  # fuzzy fallback
            score = SequenceMatcher(None, w, color).ratio()
            if score > hits.get(color, (0.0, ""))[0]:
                hits[color] = (score, w)

    confident = {c: v for c, v in hits.items() if v[0] >= FUZZY_CUTOFF}
    if len(confident) == 1:
        c, (s, w) = next(iter(confident.items()))
        return (c, s, w, "ok")
    if len(confident) >= 2:  # both red and black heard -> ambiguous, refuse
        return (None, 1.0, ",".join(confident), "conflict")
    best_word = max(hits.values(), key=lambda v: v[0])[1] if hits else ""
    return (None, 0.0, best_word, "none")


def is_stop(text: str) -> bool:
    """True if the transcript contains a 'stop' command (English/Korean, mishear tolerant)."""
    for w in text.lower().split():
        w = w.strip(string.punctuation)
        if w in STOP_ALIASES or SequenceMatcher(None, w, "stop").ratio() >= 0.86:
            return True
    return False


def is_complete(text: str) -> bool:
    """True if the transcript contains a 'complete' command (English/Korean, mishear tolerant)."""
    for w in text.lower().split():
        w = w.strip(string.punctuation)
        if w in COMPLETE_ALIASES or SequenceMatcher(None, w, "complete").ratio() >= 0.8:
            return True
    return False


def to_task(text: str) -> str | None:
    """Map a raw transcript to 'grab the {color} ball', or None if no color."""
    color, _, _, _ = match_color(text)
    return TASK_TEMPLATE.format(color=color) if color else None


def write_task(task: str, task_file: Path) -> None:
    tmp = task_file.with_suffix(".tmp")
    tmp.write_text(task, encoding="utf-8")
    tmp.replace(task_file)  # atomic swap so the reader never sees a half write


def process_utterance(audio: np.ndarray, model: WhisperModel, task_file: Path, lang: str) -> bool:
    """Transcribe one utterance and write the task/sentinel.

    Returns True if the session should end ("complete" heard), False otherwise.
    """
    peak = float(np.max(np.abs(audio)))  # peak-normalize so a low-gain mic still feeds Whisper a loud signal
    if peak > 0:
        audio = audio / peak * 0.95
    segments, _ = model.transcribe(
        audio,
        language=lang,
        beam_size=5,
        condition_on_previous_text=False,  # each command is independent
        # Prime the domain vocabulary so accented 'red'/'black' aren't heard as other words.
        initial_prompt="Grab the red ball or the black ball and put it in the box. Stop. Complete.",
    )
    text = " ".join(s.text for s in segments).strip()
    print(f"\n  heard : {text!r}")

    if is_complete(text):  # complete overrides everything: end the whole session
        write_task(COMPLETE_TASK, task_file)
        print("  -> COMPLETE: ending session.")
        return True

    if is_stop(text):  # stop overrides color: robot returns home and waits
        write_task(STOP_TASK, task_file)
        print("  -> STOP: robot returning home, waiting for next command.")
        return False

    color, score, matched, reason = match_color(text)
    print(f"  match : color={color} score={score:.2f} word={matched!r} ({reason})")

    if reason == "conflict":
        print(f"  -> both colors heard ({matched}) - ambiguous, not writing. say ONE color.")
        return False
    if color is None:
        print(f"  -> no confident color (need one of {COLORS}) - not writing.")
        return False

    task = TASK_TEMPLATE.format(color=color)
    write_task(task, task_file)
    print(f"  -> task written: {task}")
    return False


def record_until_enter(stream_q: queue.Queue) -> np.ndarray | None:
    """Record audio until the user presses Enter again (or MAX_RECORD_S elapses).
    Returns the captured audio, or None if nothing usable was recorded."""
    while not stream_q.empty():  # drop audio buffered before recording started
        stream_q.get_nowait()
    stop = threading.Event()
    threading.Thread(target=lambda: (sys.stdin.readline(), stop.set()), daemon=True).start()
    print("  recording... speak, then press Enter to STOP.", flush=True)

    frames: list[np.ndarray] = []
    max_frames = int(MAX_RECORD_S / BLOCK_S)
    while not stop.is_set() and len(frames) < max_frames:
        try:
            frames.append(stream_q.get(timeout=0.1))
        except queue.Empty:
            pass
    if not frames:
        return None
    return np.concatenate(frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium", help="faster-whisper size (tiny/base/small/medium/large-v3)")
    ap.add_argument("--lang", default="en", help="spoken language: 'en' (default) or 'ko'")
    ap.add_argument("--task-file", type=Path, default=DEFAULT_TASK_FILE)
    args = ap.parse_args()

    print(f"loading whisper '{args.model}' on cuda...", flush=True)
    model = WhisperModel(args.model, device="cuda", compute_type="float16")

    block = int(SAMPLE_RATE * BLOCK_S)
    stream_q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        stream_q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=block,
                        dtype="float32", callback=callback):
        print(f"task file -> {args.task_file}", flush=True)

        while True:  # push-to-talk: Enter to start, Enter to stop
            try:
                input("\n[Enter] to START recording (Ctrl+C to quit) ")
            except EOFError:
                break
            audio = record_until_enter(stream_q)
            if audio is None or float(np.sqrt(np.mean(audio ** 2))) < MIN_RMS:
                print("  (too quiet / nothing recorded - try again)")
                continue
            if process_utterance(audio, model, args.task_file, args.lang):
                print("session complete. bye")
                return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
