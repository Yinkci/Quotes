"""Your own voice-over instead of an AI voice.

The recording (voice, often with music under it) plays unchanged from start to end as the video's
soundtrack. Whisper (faster-whisper, already on this PC) writes down every word with its time;
that turns the recording into scenes (cut at sentence ends, a few seconds each) and times the captions. Each scene keeps
`rec_start`, the second in the recording where it begins; it ends where the next scene begins.
"""
import hashlib

import re
import subprocess

import numpy as np
import soundfile as sf

from .config import CONFIG, WORKSPACE
from .captions import build_captions
from .tts import SAMPLE_RATE, _align_words

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
DECODED_DIR = WORKSPACE / "cache" / "recordings"
DECODED_DIR.mkdir(parents=True, exist_ok=True)
WHISPER_MODELS = ("small.en", "tiny.en")      # already downloaded on this PC; small.en is the accurate one
MIN_SCENE, MAX_SCENE = 2.0, 8.0               # seconds on screen per scene
SENTENCE_END = re.compile(r"[.!?…][\"'”’)\]]*$")
SOFT_END = re.compile(r"[,;:—–-][\"'”’)\]]*$")


class RecordingError(RuntimeError):
    pass


def of(project):
    """The project's recording, or None when it uses an AI voice."""
    rec = project.get("recording") or {}
    return rec if rec.get("file") else None


def _key(path):
    from pathlib import Path
    st = Path(path).stat()
    return hashlib.sha1(f"{path}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]


def decoded(path):
    """The recording as mono float audio at SAMPLE_RATE (cached), for slicing per scene."""
    from pathlib import Path
    if not Path(path).is_file():
        raise RecordingError(f"Your recording was not found ({Path(path).name}). Choose it again in Settings → Voice.")
    wav = DECODED_DIR / f"{_key(path)}.wav"
    if not wav.exists():
        tmp = wav.with_suffix(".tmp.wav")
        proc = subprocess.run([CONFIG["ffmpeg"], "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-i", str(path),
                               "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le", str(tmp)],
                              capture_output=True, text=True, creationflags=NO_WINDOW)
        if proc.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RecordingError("Could not read your recording: " + (proc.stderr.strip()[-300:] or "unknown error"))
        tmp.replace(wav)
    audio, _ = sf.read(str(wav), dtype="float32")
    return audio


def transcribe(path, job):
    """Every spoken word with its time: {"duration", "words": [[text, start, end], ...]}."""
    duration = len(decoded(path)) / SAMPLE_RATE
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RecordingError("Whisper (faster-whisper) is not installed in this Python.") from None
    job.update("Loading Whisper…", 0.02)
    model = None
    for name in WHISPER_MODELS:
        try:
            model = WhisperModel(name, device="cpu", compute_type="int8", local_files_only=True)
            break
        except Exception:      # not downloaded: try the next one
            continue
    if model is None:
        raise RecordingError("No Whisper model was found on this PC (small.en or tiny.en).")
    # Without a punctuated prompt Whisper often leaves out full stops, and scenes are cut at sentence ends.
    segments, _ = model.transcribe(str(path), language="en", word_timestamps=True, beam_size=5,
                                   condition_on_previous_text=False,
                                   initial_prompt="Hello. This is a story, told with full punctuation.")
    words = []
    for seg in segments:
        if job.cancelled():
            from .tts import Cancelled
            raise Cancelled()
        job.update(f"Listening to your recording… {seg.end:.0f}s of {duration:.0f}s", 0.05 + 0.9 * min(1, seg.end / (duration or 1)))
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append([text, round(float(w.start), 3), round(float(max(w.start, w.end)), 3)])
    if not words:
        raise RecordingError("No speech was found in the recording.")
    return {"duration": round(duration, 3), "words": refine_words(words, decoded(path))}


FRAME = 0.02            # seconds per loudness frame
CAPTION_PAUSE = 0.35    # a silence this long between two words always starts a new caption
LEAD_QUIET = 0.15       # a quiet stretch this long at the start of a word is not part of it
VOICE_BAND = (300, 11000)  # Hz: speech incl. s/f/t sounds; below it are the bass and low notes of music under the voice


def refine_words(words, audio):
    """Whisper often gives a word the silence before it ("and" 8.08–9.64 s when it is spoken at 9.6 s),
    so its caption would appear while nothing is said. For a word that lasts far longer than it
    takes to say, find where the voice really is in that stretch: the last loud part, followed back
    to where it starts (short dips inside a word are allowed). Works with music under the voice,
    because it compares loudness within the word's own stretch.
    Loudness is measured twice: over all frequencies, and only over the voice's frequencies. Music
    can fool either one (a held bass note is loud overall but quiet in the voice band; soft music
    in a pause can be as loud in the voice band as a low word like "and"), and a fooled one only
    finds the voice too early, never too late, so the later start of the two wins."""
    hop = int(SAMPLE_RATE * FRAME)
    n = len(audio) // hop
    if not n:
        return words
    frames = audio[:n * hop].reshape(n, hop)
    spec = np.abs(np.fft.rfft(frames * np.hanning(hop), axis=1))
    freqs = np.fft.rfftfreq(hop, 1 / SAMPLE_RATE)
    measures = [np.sqrt((frames ** 2).mean(axis=1)),
                np.sqrt((spec[:, (freqs >= VOICE_BAND[0]) & (freqs <= VOICE_BAND[1])] ** 2).sum(axis=1))]
    measures = [(rms, float(np.percentile(rms, 15))) for rms in measures]
    return [[text, *max((_refine_word(text, s, e, rms, floor) for rms, floor in measures), key=lambda t: t[0])]
            for text, s, e in words]


def _refine_word(text, s, e, rms, floor):
    """(start, end) of one word where the voice really is, by one loudness measure (see refine_words)."""
    n = len(rms)
    expected = 0.1 + 0.06 * len(re.sub(r"\W", "", text))
    a, b = int(s / FRAME), min(n, int(np.ceil(e / FRAME)))
    if e - s > expected * 1.6 + 0.2 and b - a > 2:
        span = rms[a:b]
        peak = float(span.max())
        if peak > floor * 1.6:          # there is a voice in this stretch
            last = a + int(np.flatnonzero(span >= 0.5 * peak)[-1])
            first, quiet = last, 0
            while first - 1 >= a and quiet <= 4:
                first -= 1
                quiet = quiet + 1 if rms[first] < 0.5 * peak else 0
            first += quiet
            s, e = round(max(s, first * FRAME), 3), round(min(e, max((last + 1) * FRAME, first * FRAME + 0.1)), 3)
    # Also a word of normal length can start with silence (often the first word after a pause):
    # skip a quiet stretch of LEAD_QUIET or more at its start, so its caption waits for the voice.
    a, b = int(s / FRAME), min(n, int(np.ceil(e / FRAME)))
    if b - a > 2:
        span = rms[a:b]
        loud = np.flatnonzero(span >= max(floor * 1.3, 0.35 * float(span.max())))
        if len(loud) and loud[0] * FRAME >= LEAD_QUIET:
            s = round(min(a + int(loud[0]), b - 1) * FRAME, 3)
            e = round(max(e, s + 0.1), 3)
    return s, e


def make_scenes(words, duration=None, lo=MIN_SCENE, hi=MAX_SCENE):
    """Cut the recording into scenes that each stay on screen between lo and hi seconds, as closely
    as possible. Cuts go at sentence ends where they can, else after a comma, else at a pause; out of
    all ways to cut, the one with the lowest total cost wins. Returns [{"script", "start"}]; a scene
    starts halfway through the pause before its first word and ends where the next one starts."""
    if not words:
        return []
    end = max(duration or 0, words[-1][2])
    # Possible cuts before word k: (k, time, cost of cutting there).
    cuts = [(0, 0.0, 0.0)]
    for k in range(1, len(words)):
        before, gap = words[k - 1][0], words[k][1] - words[k - 1][2]
        # Inside a sentence only where the voice pauses (a comma read straight on is no place to cut);
        # costly enough that one cut in a long sentence beats moving every other scene off its sentence.
        cost = 0.0 if SENTENCE_END.search(before) else None if gap < 0.3 else \
            (1.0 if SOFT_END.search(before) else 1.5) - 0.5 * min(gap, 1.0)
        if cost is not None:
            cuts.append((k, (words[k - 1][2] + words[k][1]) / 2, cost))
    cuts.append((len(words), end, 0.0))

    def scene_cost(length):
        under, over = max(0.0, lo - length), max(0.0, length - hi)
        return 2 * under ** 2 + 4 * over ** 2 + (100 if length < MIN_SCENE else 0)

    best = [0.0] + [float("inf")] * (len(cuts) - 1)
    prev = [0] * len(cuts)
    for q in range(1, len(cuts)):
        for p in range(q - 1, -1, -1):
            length = cuts[q][1] - cuts[p][1]
            if length > 3 * hi and p < q - 1:
                break
            c = best[p] + scene_cost(length) + (cuts[q][2] if q < len(cuts) - 1 else 0)
            if c < best[q]:
                best[q], prev[q] = c, p
    chosen, q = [], len(cuts) - 1
    while q:
        chosen.append(q)
        q = prev[q]
    bounds = [0] + chosen[::-1]
    return [{"script": " ".join(w[0] for w in words[cuts[a][0]:cuts[b][0]]), "start": round(cuts[a][1], 3)}
            for a, b in zip(bounds, bounds[1:])]


def bounds(scenes, duration):
    """(start, end) in the recording for every scene. A missing or out-of-order rec_start (a scene
    added or moved by hand) is spread evenly between its neighbours. Same rule as the page."""
    n = len(scenes)
    starts = [None] * n
    last = 0.0
    for i, sc in enumerate(scenes):
        v = sc.get("rec_start")
        if i == 0:
            starts[i] = 0.0
        elif isinstance(v, (int, float)) and last < v < duration:
            starts[i] = last = float(v)
    i = 0
    while i < n:
        if starts[i] is None:
            p = i - 1
            q = i
            while q < n and starts[q] is None:
                q += 1
            a, b = starts[p], (starts[q] if q < n else duration)
            for k in range(i, q):
                starts[k] = a + (b - a) * (k - p) / (q - p)
            i = q
        i += 1
    return [(starts[i], starts[i + 1] if i + 1 < n else duration) for i in range(n)]


def scene_captions(script, words, start, end, words_per_caption):
    """Caption events (seconds from the scene start) for this scene's script, timed by the words
    Whisper heard between start and end. The captions always show the script as typed."""
    ours = (script or "").split()
    if not ours:
        return []
    heard = [w for w in words if start <= (w[1] + w[2]) / 2 < end]
    # Times relative to the first heard word, so if the text no longer matches (edited a lot) the
    # words are spread over the time actually spoken, not over the silence before it.
    base = heard[0][1] if heard else start
    total = (heard[-1][2] - base) if heard else end - start
    times = _align_words(ours, [w[0] for w in heard], [w[1] - base for w in heard], [w[2] - base for w in heard],
                         max(0.1, total))
    shift = base - start
    times = [(a + shift, b + shift) for a, b in times]
    # A pause in the voice always ends a caption, so a caption never shows words not yet spoken.
    events, run = [], [0]
    for k in range(1, len(ours) + 1):
        if k == len(ours) or times[k][0] - times[k - 1][1] > CAPTION_PAUSE:
            events += build_captions([ours[j] for j in run], [times[j] for j in run], words_per_caption)
            run = []
        run.append(k)
    return events
