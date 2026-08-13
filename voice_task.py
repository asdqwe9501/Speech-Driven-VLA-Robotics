"""Mic -> faster-whisper -> English task string -> current_task.txt

Hands-free: the mic stays open, an energy VAD auto-detects when you start and
stop speaking, and each utterance is transcribed. The recognized color is
normalized into the exact task string the policy was trained on and written to
current_task.txt so lerobot-record can pick it up live. No key presses.
"""

import argparse
import collections
import queue
import string
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
COLORS = ("red", "black")  # the two colors the policy was trained on
# Exact trained task strings (verified from dataset6 meta) - must match verbatim
TASK_TEMPLATE = "grab the {color} ball and put it in the box"
STOP_TASK = "__STOP__"  # sentinel: record loop returns the arm home and waits
COMPLETE_TASK = "__COMPLETE__"  # sentinel: record loop ends the whole session (full shutdown)
DEFAULT_TASK_FILE = Path(__file__).parent / "current_task.txt"
FUZZY_CUTOFF = 0.6  # min string-similarity to accept a mispronounced word

# Common Whisper mishears -> canonical color (poor pronunciation / accents)
ALIASES = {
    "red": "red", "read": "red", "rad": "red", "bread": "red", "rate": "red",
    "black": "black", "block": "black", "back": "black", "blak": "black",
    "blag": "black", "plaque": "black", "blackball": "black",
}

# Gentle bias toward the expected words without prompt-echo hallucination
HOTWORDS = "red black ball stop complete"

# VAD (energy-based) tuning
BLOCK_S = 0.03            # audio block size fed to the callback (30 ms)
PREROLL_S = 0.3          # audio kept before speech onset so we don't clip starts
END_SILENCE_S = 0.6     # trailing silence that ends an utterance
MIN_SPEECH_S = 0.3      # ignore blips shorter than this
NOISE_MULT = 1.8        # start threshold = ambient RMS * this


def rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block ** 2)))


def calibrate(stream_q: queue.Queue, blocks: int) -> float:
    """Measure ambient noise floor and return a speech-onset threshold."""
    floor = 0.0
    for _ in range(blocks):
        floor = max(floor, rms(stream_q.get()))
    thresh = max(floor * NOISE_MULT, 0.0035)
    print(f"  noise floor={floor:.4f} -> speech threshold={thresh:.4f}", flush=True)
    return thresh


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
    """True if the transcript contains a 'stop' command (with mishear tolerance)."""
    for w in text.lower().split():
        w = w.strip(string.punctuation)
        if w in ("stop", "stopp", "stahp") or SequenceMatcher(None, w, "stop").ratio() >= 0.86:
            return True
    return False


def is_complete(text: str) -> bool:
    """True if the transcript contains a 'complete' command (with mishear tolerance)."""
    for w in text.lower().split():
        w = w.strip(string.punctuation)
        if w in ("complete", "completed", "komplete") or SequenceMatcher(None, w, "complete").ratio() >= 0.8:
            return True
    return False


def has_ball(text: str) -> bool:
    """True if the utterance mentions 'ball' (mishear tolerant) - the command must
    be the full task sentence, not a bare color word."""
    for w in text.lower().split():
        w = w.strip(string.punctuation)
        if w == "ball" or SequenceMatcher(None, w, "ball").ratio() >= 0.75:
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


def process_utterance(audio: np.ndarray, model: WhisperModel, task_file: Path) -> bool:
    """Transcribe one utterance and write the task/sentinel.

    Returns True if the session should end ("complete" heard), False otherwise.
    """
    segments, _ = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        hotwords=HOTWORDS,
        condition_on_previous_text=False,  # each command is independent
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
    if not has_ball(text):  # require the full ball command, not a bare color word
        print("  -> heard a color but not the ball command - say the full sentence.")
        return False

    task = TASK_TEMPLATE.format(color=color)
    write_task(task, task_file)
    print(f"  -> task written: {task}")
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", help="faster-whisper size (tiny/base/small/medium)")
    ap.add_argument("--task-file", type=Path, default=DEFAULT_TASK_FILE)
    args = ap.parse_args()

    print(f"loading whisper '{args.model}' on cuda...", flush=True)
    model = WhisperModel(args.model, device="cuda", compute_type="float16")

    block = int(SAMPLE_RATE * BLOCK_S)
    stream_q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        stream_q.put(indata[:, 0].copy())

    end_silence_blocks = int(END_SILENCE_S / BLOCK_S)
    preroll = collections.deque(maxlen=int(PREROLL_S / BLOCK_S))
    min_speech_blocks = int(MIN_SPEECH_S / BLOCK_S)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=block,
                        dtype="float32", callback=callback):
        print("calibrating ambient noise (stay quiet ~1s)...", flush=True)
        thresh = calibrate(stream_q, blocks=int(1.0 / BLOCK_S))
        print(f"listening. task file -> {args.task_file}  (Ctrl+C to quit)", flush=True)

        speech: list[np.ndarray] = []
        in_speech = False
        silence = 0
        while True:
            data = stream_q.get()
            loud = rms(data) > thresh
            if not in_speech:
                preroll.append(data)
                if loud:
                    in_speech, silence = True, 0
                    speech = list(preroll) + [data]
                    print("  * speech onset", flush=True)
            else:
                speech.append(data)
                silence = 0 if loud else silence + 1
                if silence >= end_silence_blocks:  # utterance finished
                    in_speech = False
                    preroll.clear()
                    dur = (len(speech) - silence) * BLOCK_S
                    if len(speech) - silence >= min_speech_blocks:
                        if process_utterance(np.concatenate(speech), model, args.task_file):
                            print("session complete. bye")
                            return
                    else:
                        print(f"  * utterance too short ({dur:.2f}s), ignored", flush=True)
                    speech = []


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
