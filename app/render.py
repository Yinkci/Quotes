"""Project -> video.

Pipeline: narration per scene (cached) -> frame-exact timeline -> ASS subtitles -> one still
clip per scene (cached) -> a single FFmpeg pass that joins clips with hard cuts, burns subtitles
and muxes the narration track. Images are static: no movement, transitions or fades.

Timing rule: every scene lasts lead-in + its measured narration + a pause, rounded UP to whole
frames, so audio and video can never drift apart.
"""
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import CONFIG, EXPORTS_DIR, LUTS_DIR, MY_DEFAULTS_FILE, OVERLAYS_DIR, SOUNDS_DIR, WORKSPACE
from .tts import SAMPLE_RATE, Cancelled, ElevenLabs
from .captions import split_sentences, write_ass
from . import recording

SEGMENT_VERSION = 4
QUALITY = {"standard": ("medium", 21), "high": ("medium", 18), "max": ("slow", 16)}
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

SEGMENT_DIR = WORKSPACE / "cache" / "segments"
AUDIO_DIR = WORKSPACE / "cache" / "audio"
PREVIEW_DIR = WORKSPACE / "preview"
JOBS_DIR = WORKSPACE / "jobs"
for _d in (SEGMENT_DIR, AUDIO_DIR, PREVIEW_DIR, JOBS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

def builtin_looks():
    """Names of the built-in color grades (the .cube files in luts/)."""
    return sorted(p.stem for p in LUTS_DIR.glob("*.cube"))


DEFAULT_LOOK = "Anime 03" if "Anime 03" in builtin_looks() else ""

DEFAULT_SETTINGS = {
    "voice": {"voice": "am_liam", "speed": 1.0, "style": "story",
              "effect": "website", "soften_s": False,     # sound effect on the narration (VOICE_EFFECTS)
              "settle_end": False,                        # last word slower + lower (settle_ending)
              "volume": 40},                              # % (voice_loudness): 100 = as loud as TikTok / Reels
    # Pauses after a scene, by how it ends (like quote reels: ~1.6 s heard after a full stop, ~0.7 s after
    # a comma or "..."; the voice clips add ~0.3 s of their own). Lead-in 0: the picture cuts on the first word.
    "timing": {"lead_in": 0.0, "pause_sentence": 1.25, "pause_open": 0.4, "sentence_pause": 0.2,
               "end_hold": 3.0,   # music plays on after the last line, then the video ends
               "silent_scene": 3.0},       # seconds a scene with an image but no narration is shown
    "subtitles": {"enabled": True, "font": "Arial Black", "size": 84, "color": "#FFFFFF",
                  "outline_color": "#000000", "outline": 6, "shadow": 2, "bold": True, "italic": False,
                  "uppercase": False, "position": 26, "position_x": 50, "max_width": 84,
                  "words_per_caption": 5,
                  "first_at_start": True,      # the first caption is on screen from the first frame (the hook)
                  "flourish": False},          # a thin curved swoosh under the captions
    "imagegen": {"style": "", "no_text": True, "references": [], "continuity": True,
                 "places": "new"},        # "new": every scene is a different place; "story": stay until the story moves
    "story": {"continues_from": "",        # project file of the episode this one continues
              "world": ""},                # what the world looks like now (e.g. "a week into a zombie outbreak")
    "watermark": {"enabled": False, "text": "", "font": "Arial", "size": 40, "color": "#FFFFFF",
                  "opacity": 70, "outline": 2, "outline_color": "#000000", "shadow": 1, "bold": True, "italic": False,
                  "anchor": "br", "position_x": 96, "position_y": 96},
    "music": {"enabled": False, "file": "", "name": "", "volume": 15, "duck": True, "fade_out": 2.0,
              "radio": True},       # the music sounds like it comes out of the radio too (MUSIC_RADIO)
    # A sound at the very start of the video (default: pressing Play on a cassette). The voice starts
    # voice_at s into it (0.7: right after the click; the rest of it plays on under the voice), or when it
    # ends if that's sooner. length (s) is only for the app's duration estimates. tape: the cassette keeps
    # running (hiss, motor) under the whole video after it.
    "intro": {"enabled": False, "file": str(SOUNDS_DIR / "cassette_play.wav"), "name": "Cassette play",
              "volume": 70, "length": 2.4, "voice_at": 0.7, "tape": True},
    # A picture on top of the whole video, under the captions (default: the animated radio cassette GIF,
    # looping). size = % of the video width; position_x / position_y = where its center is, % of the frame.
    # pulse: how much it moves with the voice; pulse_mode: "off" (it stays still), "classic" (the first version: the
    # whole picture pulses from its center), "whole" (it bounces from its bottom edge, shadow too) or
    # "speakers" (only its speaker cones push, when it has speaker info); shadow: a soft shadow under it;
    # graded: the color grade is applied to it too, so it sits in the picture.
    "overlay": {"enabled": False, "file": str(OVERLAYS_DIR / "radio_cassette.gif"), "name": "Radio cassette",
                "size": 60, "position_x": 50, "position_y": 16, "opacity": 100,
                "pulse": 3, "pulse_mode": "off", "shadow": True, "graded": True},
    # The channel's name, always in the same spot (like a radio display). x / y: its center, % of the frame.
    "station": {"enabled": True, "text": "DEEP TALK RADIO · 104.5 FM", "font": "Consolas", "size": 38,
                "color": "#FFB347", "glow": True, "position_x": 50, "position_y": 5,
                "lines": True,                 # thin lines on both sides of it
                "on_air": True},               # a red "ON AIR" light under it
    # The end: after the hold, the radio clicks off (the music and tape stop with it), and a line
    # appears under the last caption once the voice is done.
    "outro": {"click": True, "text_enabled": True, "text": "Tune in tomorrow · Deep Talk Radio"},
    # Film grain and a soft vignette over the picture (0 = off), under the captions.
    "texture": {"grain": 20, "vignette": 30},
    # Bars that move with the voice, with thin lines on both sides. width: % of the video width;
    # position_x / position_y: its center, % of the frame.
    "visualizer": {"enabled": True, "width": 30, "position_x": 50, "position_y": 93, "color": "#FFE9C7"},
    # Color grade over the whole video: a built-in Look (luts/<look>.cube), or your own .cube file (lut wins).
    "color": {"look": DEFAULT_LOOK, "lut": "", "lut_name": "", "strength": 100},
    "motion": {"fit": "contain"},      # images are always static; only how they fit the frame is a choice
    "export": {"output_dir": str(EXPORTS_DIR), "filename": "", "resolution": "1080x1920",
               "fps": 30, "quality": "high"},
}
BUILTIN_SETTINGS = json.loads(json.dumps(DEFAULT_SETTINGS))
# Settings that belong to one episode only, never to "my defaults".
NOT_DEFAULTS = {"story": None, "export": {"filename", "output_dir"}, "imagegen": {"references"}}


def defaults_from(settings):
    """The part of a project's settings that can become the starting point for new projects."""
    out = {}
    for section, builtin in BUILTIN_SETTINGS.items():
        skip = NOT_DEFAULTS.get(section, set())
        if skip is None:
            continue
        given = settings.get(section) or {}
        out[section] = {k: given[k] for k in builtin if k in given and k not in skip}
    return out


def apply_my_defaults(mine=None):
    """DEFAULT_SETTINGS = built-in values, overridden by my_defaults.json (or `mine`)."""
    if mine is None:
        try:
            mine = json.loads(MY_DEFAULTS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            mine = {}
    for section, builtin in BUILTIN_SETTINGS.items():
        DEFAULT_SETTINGS[section] = {**json.loads(json.dumps(builtin)), **(mine.get(section) or {})}


apply_my_defaults()


class RenderError(Exception):
    pass


def merged_settings(project):
    """Project settings with defaults filled in for anything missing."""
    given = project.get("settings") or {}
    out = {}
    for section, defaults in DEFAULT_SETTINGS.items():
        out[section] = {**defaults, **(given.get(section) or {})}
    return out


# ---- jobs ------------------------------------------------------------------------------------
class Job:
    def __init__(self, kind):
        self.id = uuid.uuid4().hex[:10]
        self.kind = kind
        self.state = "running"          # running | done | error | cancelled
        self.stage = "Starting…"
        self.progress = 0.0
        self.error = None
        self.result = None
        self.started = time.time()
        self.proc = None
        self._cancel = threading.Event()

    def update(self, stage=None, progress=None):
        if stage is not None:
            self.stage = stage
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))

    def cancel(self):
        self._cancel.set()
        proc = self.proc
        if proc and proc.poll() is None:
            proc.kill()

    def cancelled(self):
        return self._cancel.is_set()

    def to_dict(self):
        return {"id": self.id, "kind": self.kind, "state": self.state, "stage": self.stage,
                "progress": round(self.progress, 4), "error": self.error, "result": self.result,
                "elapsed": round(time.time() - self.started, 1)}


def run_ffmpeg(args, job, cwd=None, duration=None, on_progress=None):
    cmd = [CONFIG["ffmpeg"], "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
           "-progress", "pipe:1", "-nostats", *args]
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
    job.proc = proc
    errors = []
    reader = threading.Thread(target=lambda: errors.extend(proc.stderr.readlines()), daemon=True)
    reader.start()
    for line in proc.stdout:
        if job.cancelled():
            proc.kill()
            break
        if duration and on_progress and line.startswith("out_time_us="):
            try:
                on_progress(min(1.0, int(line.split("=")[1]) / 1e6 / duration))
            except ValueError:
                pass
    proc.wait()
    reader.join(timeout=5)
    job.proc = None
    if job.cancelled():
        raise Cancelled()
    if proc.returncode != 0:
        detail = "".join(errors[-12:]).strip() or f"exit code {proc.returncode}"
        raise RenderError(f"FFmpeg failed: {detail}")


_probe_cache = {}


def image_size(path):
    st = Path(path).stat()
    key = (str(path), st.st_mtime, st.st_size)
    if key not in _probe_cache:
        out = subprocess.run([CONFIG["ffprobe"], "-v", "error", "-select_streams", "v:0",
                              "-show_entries", "stream=width,height", "-of", "json", str(path)],
                             capture_output=True, text=True, creationflags=NO_WINDOW)
        try:
            stream = json.loads(out.stdout)["streams"][0]
            _probe_cache[key] = (int(stream["width"]), int(stream["height"]))
        except (ValueError, KeyError, IndexError):
            raise RenderError(f"Could not read image: {Path(path).name}. Is it a valid PNG/JPG/WebP?")
    return _probe_cache[key]


# ---- narration + timeline --------------------------------------------------------------------
def validate(project):
    problems = []
    scenes = project.get("scenes") or []
    if not scenes:
        problems.append("Add at least one scene.")
    for i, scene in enumerate(scenes, 1):
        image = scene.get("image")
        if image and not Path(image).exists():      # no image at all is fine: that scene is black
            problems.append(f"Scene {i}: image not found ({Path(image).name}). Use Replace to locate it.")
        if not (scene.get("script") or "").strip() and not image and not recording.of(project):
            problems.append(f"Scene {i} is empty: add narration or an image.")   # an image alone is a silent scene
    lut = lut_path(merged_settings(project)["color"])
    if lut and not Path(lut).is_file():
        problems.append(f"Color grade file not found ({Path(lut).name}). Choose it again (Settings → Color grade).")
    rec = recording.of(project)
    if rec and not Path(rec["file"]).is_file():
        problems.append(f"Your recording was not found ({Path(rec['file']).name}). Choose it again in Settings → Voice.")
    elif rec and not rec.get("words"):
        problems.append("Your recording hasn't been listened to yet: click 🎧 Make scenes from recording (Settings → Voice).")
    music = merged_settings(project)["music"]
    if music["enabled"] and not rec:      # a recording brings its own music
        if not music["file"]:
            problems.append("Background music is on but no music file is chosen (Settings → Background music).")
        elif not Path(music["file"]).is_file():
            problems.append(f"Background music file not found ({Path(music['file']).name}). Choose it again.")
    intro = merged_settings(project)["intro"]
    if intro["enabled"] and not rec and not Path(intro["file"] or "").is_file():
        problems.append("The intro sound file was not found. Choose it again (Settings → Intro sound).")
    overlay = merged_settings(project)["overlay"]
    if overlay["enabled"] and not Path(overlay["file"] or "").is_file():
        problems.append("The overlay picture was not found. Choose it again (Settings → Overlay).")
    if problems:
        raise RenderError("\n".join(problems))


def scene_audio_key(script, settings, around=("", ""), final=None, settled=False):
    v, t = settings["voice"], settings["timing"]
    parts = [v["voice"], round(float(v["speed"]), 3), float(t["sentence_pause"]), script]
    if v.get("style", "story") != "story":       # left out for storytelling so existing caches stay valid
        parts.append(v["style"])
    if str(v["voice"]).startswith("eleven:"):    # a new model gives new audio, so a new file name
        parts.append(ElevenLabs.model)
        if any(around):                          # ElevenLabs reads differently with the neighbouring scenes
            parts.append(list(around))
        if final:                                # read as part of the video's last sentence (final_group)
            parts.append(["final", final])
    if settled:                                  # the video's last word, settled down (settle_ending)
        parts.append(["settled", SETTLE_DROP, SETTLE_STRETCH])
    raw = json.dumps(parts)
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def scene_context(scenes, i):
    """(script of the nearest spoken scene before scene i, ... after it) — "" at the start / end."""
    said = lambda k: (scenes[k].get("script") or "").strip()
    before = next((said(k) for k in range(i - 1, -1, -1) if said(k)), "")
    after = next((said(k) for k in range(i + 1, len(scenes)) if said(k)), "")
    return before, after


def final_group(scenes):
    """The video's last sentence, when it starts in earlier scenes that run on into it (no . ! ? at their end):
    ([scene index per fragment], [fragment text], text before it). None when the last scene is one sentence."""
    said = [k for k, sc in enumerate(scenes) if (sc.get("script") or "").strip()]
    if not said:
        return None
    units = lambda k: split_sentences(scenes[k]["script"])
    last = said[-1]
    if len(units(last)) != 1:
        return None
    idx, frags, before = [last], [units(last)[0]], ""
    for k in reversed(said[:-1]):
        if not runs_on(scenes[k]["script"]):
            break
        parts = units(k)
        idx.insert(0, k)
        frags.insert(0, parts[-1])
        if len(parts) > 1:          # the sentence starts inside this scene: its earlier sentences come before
            before = " ".join(parts[:-1])
            break
    if len(idx) == 1:
        return None
    if not before:
        before = scene_context(scenes, idx[0])[0]
    return idx, frags, before


# The video's last word: AI voices often lift the end of it, which sounds like more is coming. Its pitch glides
# down by SETTLE_DROP semitones and it is stretched a little, so the voice lands (measured: "happy" went from
# -2 -> 0 semitones at its end to -2 -> -4).
SETTLE_DROP = 4.0
SETTLE_STRETCH = 1.1


def settle_ending(audio, word):
    """`audio` with its word at `word` = (start, end) s gliding down in pitch and slowed a little (FFmpeg
    rubberband, voice character kept). Everything before the word is untouched."""
    sr, pre = SAMPLE_RATE, 0.3          # the 0.3 s before the word goes through too, then is cross-faded away
    ws, we = word
    start = max(0.0, ws - pre)
    w0, w1 = ws - start, we - start
    ramp = max(0.05, (w1 - w0) * 0.9)
    cmds = [f"{w0:.3f} rubberband@rb tempo {1 / SETTLE_STRETCH:.4f};"]
    cmds += [f"{t:.3f} rubberband@rb pitch {2 ** (-SETTLE_DROP * (t - w0) / ramp / 12):.4f};"
             for t in np.arange(w0, w0 + ramp, 0.02)]
    tmp = Path(tempfile.mkdtemp(prefix="settle_"))
    try:
        sf.write(str(tmp / "in.wav"), audio[int(start * sr):], sr)
        (tmp / "cmds.txt").write_text("\n".join(cmds))
        subprocess.run([CONFIG["ffmpeg"], "-y", "-v", "error", "-i", "in.wav", "-af",
                        "asendcmd=f=cmds.txt,rubberband@rb=tempo=1:pitch=1:formant=preserved:pitchq=quality",
                        "out.wav"], check=True, cwd=str(tmp), creationflags=NO_WINDOW)
        out = sf.read(str(tmp / "out.wav"), dtype="float32")[0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if out.ndim > 1:
        out = out.mean(axis=1)
    # 40 ms cross-fade from the original into the processed audio, ending 20 ms before the word starts
    j, xf = int(start * sr) + int((w0 - 0.06) * sr), int(0.04 * sr)
    k = j - int(start * sr)
    if k < 0 or k + xf > len(out):
        return audio
    fade = np.linspace(0, 1, xf, dtype=np.float32)
    mix = audio[j:j + xf] * (1 - fade) + out[k:k + xf] * fade
    return np.concatenate([audio[:j], mix, out[k + xf:]]).astype(np.float32)


def narrate(project, narrator, job, progress_span=(0.0, 1.0), only=None):
    """Generate (or load cached) narration for every scene (or just scene `only`). Returns per-scene dicts."""
    if recording.of(project):
        return narrate_recording(project, job, progress_span)
    s = merged_settings(project)
    voice, speed = s["voice"]["voice"], float(s["voice"]["speed"])
    scenes = project["scenes"]
    eleven = str(voice).startswith("eleven:")
    # The last sentence is read in one go so it ends like an ending (ElevenLabs, storytelling style only).
    group = final_group(scenes) if eleven and s["voice"].get("style", "story") == "story" else None
    pieces = None
    last_said = max((k for k, sc in enumerate(scenes) if (sc.get("script") or "").strip()), default=-1)
    out = []
    a, b = progress_span
    for i, scene in enumerate(scenes):
        if only is not None and i != only:
            continue
        job.update(f"Narration: scene {i + 1} of {len(scenes)}", a + (b - a) * i / len(scenes))
        script = (scene.get("script") or "").strip()
        if script:
            # Only the scene before: given the next one, ElevenLabs runs on into it and the clip ends mid-word.
            around = (scene_context(scenes, i)[0], "") if eleven else ("", "")
            final = None
            if group and i in group[0]:
                if pieces is None:
                    pieces = narrator.final_sentence(group[1], voice, speed, group[2])
                final = group[1]
            audio, captions, fresh, last_word = narrator.scene(
                scene["script"], voice, speed, float(s["timing"]["sentence_pause"]),
                int(s["subtitles"]["words_per_caption"]), cancelled=job.cancelled,
                style=s["voice"].get("style", "story"), around=around,
                last_unit=pieces[group[0].index(i)] if final else None)
            if not len(audio):
                raise RenderError(f"Scene {i + 1}: narration produced no audio. Check the script text.")
            # The video's last words: the voice settles down on the final word, like the end of a talk.
            settled = (s["voice"].get("settle_end", False) and i == last_said and not runs_on(script)
                       and last_word is not None)
            key = scene_audio_key(scene["script"], s, around, final, settled)
            if settled and not (AUDIO_DIR / f"{key}.wav").exists():
                audio = settle_ending(audio, last_word)
            elif settled:
                audio = sf.read(str(AUDIO_DIR / f"{key}.wav"), dtype="float32")[0]
        else:   # image only: silence for the chosen length, no captions
            hold = max(0.5, float(s["timing"].get("silent_scene", 3.0)))
            audio, captions = np.zeros(int(round(hold * SAMPLE_RATE)), dtype=np.float32), []
            key = scene_audio_key(f"<silent {hold}>", s)
        wav = AUDIO_DIR / f"{key}.wav"
        if not wav.exists():
            sf.write(str(wav), audio, SAMPLE_RATE)
        out.append({"id": scene.get("id"), "audio": audio, "captions": captions,
                    "speech": len(audio) / SAMPLE_RATE, "audio_url": f"/api/audio/{key}.wav"})
    return out


def narrate_recording(project, job, progress_span=(0.0, 1.0)):
    """Your own recording: each scene is its stretch of the recording, captions timed by Whisper."""
    rec = recording.of(project)
    s = merged_settings(project)
    job.update("Reading your recording", progress_span[0])
    audio = recording.decoded(rec["file"])
    duration = len(audio) / SAMPLE_RATE
    words, out = recording.refine_words(rec.get("words") or [], audio), []     # also fixes projects made before
    for scene, (start, end) in zip(project["scenes"], recording.bounds(project["scenes"], duration)):
        piece = audio[int(round(start * SAMPLE_RATE)):int(round(end * SAMPLE_RATE))]
        key = hashlib.sha1(json.dumps([rec["file"], round(start, 3), round(end, 3)]).encode()).hexdigest()[:20]
        wav = AUDIO_DIR / f"rec_{key}.wav"
        if not wav.exists():
            sf.write(str(wav), piece, SAMPLE_RATE)
        out.append({"id": scene.get("id"), "audio": piece, "speech": end - start, "rec_start": start, "rec_end": end,
                    "captions": recording.scene_captions(scene.get("script"), words, start, end,
                                                         int(s["subtitles"]["words_per_caption"])),
                    "audio_url": f"/api/audio/rec_{key}.wav"})
    return out


def ends_sentence(script):
    """True when the scene ends a sentence (. ! ?), False when it runs on (comma, "...", nothing)."""
    end = (script or "").rstrip().rstrip("\"'”’)")
    return end.endswith((".", "!", "?")) and not end.endswith(("..", "…"))


def runs_on(script):
    """True when the scene visibly continues into the next one (a comma, dash, "..." at its end). A line
    with no punctuation at all (a quote written one line per scene) is a line of its own."""
    end = (script or "").rstrip().rstrip("\"'”’)")
    return end.endswith((",", ";", ":", "—", "–", "-", "..", "…"))


def build_timeline(project, narration):
    """Frame-exact scene durations, the full narration track and global caption events."""
    s = merged_settings(project)
    fps = int(s["export"]["fps"])
    t = s["timing"]
    lead_in = max(0.0, float(t["lead_in"]))
    n = len(narration)
    rec = recording.of(project)
    intro = None if rec else intro_audio(s)
    # The voice waits this long for the intro sound (right after the click, not the whole sound).
    intro_len = (min(len(intro) / SAMPLE_RATE, max(0.0, float(s["intro"].get("voice_at", 0.7))))
                 if intro is not None else 0.0)
    outro = None if rec else outro_audio(s)      # the radio clicks off at the very end
    scenes, frame = [], 0
    if rec:      # the recording sets the clock: scenes start exactly where they start in it
        lead_in = 0.0
        edges = [0] + [round(item["rec_start"] * fps) for item in narration[1:]]
        edges.append(math.ceil(narration[-1]["rec_end"] * fps - 1e-6))
        for i, item in enumerate(narration):
            frames = max(1, edges[i + 1] - edges[i])
            scenes.append({**item, "start_frame": frame, "frames": frames, "start": frame / fps, "duration": frames / fps})
            frame += frames
    for i, item in ([] if rec else enumerate(narration)):
        last = i == n - 1
        if last:
            tail = max(0.2, float(t["end_hold"]))
            music = s["music"]       # the music plays on through the hold, THEN fades out
            if music["enabled"] and music["file"] and float(music["volume"]) > 0:
                tail += max(0.0, float(music["fade_out"]))
            if outro is not None:
                tail += OUTRO_TAIL
        else:     # breathe after a finished sentence, move on quickly when the thought continues
            done = ends_sentence(project["scenes"][i].get("script"))
            tail = max(0.0, float(t["pause_sentence"] if done else t["pause_open"]))
        pre = lead_in + (intro_len if i == 0 else 0.0)
        frames = math.ceil((pre + item["speech"] + tail) * fps - 1e-6)
        scenes.append({**item, "start_frame": frame, "frames": frames, "pre": pre,
                       "start": frame / fps, "duration": frames / fps})
        frame += frames
    total_frames = frame
    total = total_frames / fps

    track = None if rec else np.zeros(int(round(total * SAMPLE_RATE)) + 1, dtype=np.float32)
    events = []
    for i, sc in enumerate(scenes):
        if track is not None:
            at = int(round((sc["start"] + sc["pre"]) * SAMPLE_RATE))
            track[at:at + len(sc["audio"])] = sc["audio"][:len(track) - at]
        visual_end = scenes[i + 1]["start"] if i < n - 1 else total
        base = sc["rec_start"] if rec else sc["start"] + sc["pre"]     # captions follow the recording's clock
        for c in sc["captions"]:
            events.append({"scene": i, "start": base + c["start"], "end": base + c["end"], "text": c["text"],
                           "limit": visual_end})
    # The hook: the first caption is already there on the first frame, before the voice reaches it.
    if events and s["subtitles"].get("first_at_start", True):
        events[0]["start"] = 0.0
    # A caption stays on screen until the next one appears (also through a pause in the voice); the
    # last caption of a scene stays until the next scene starts (the video's end for the last).
    for j, ev in enumerate(events):
        nxt = events[j + 1] if j + 1 < len(events) else None
        if nxt and nxt["scene"] == ev["scene"]:
            ev["end"] = nxt["start"]
        else:
            ev["end"] = min(ev["limit"], nxt["start"] if nxt else total)
        ev["end"] = max(ev["end"], ev["start"] + 0.2)
    last = scenes[-1] if scenes and not rec else None
    # With the cassette click at the end, the last caption goes a moment after the last word, well before
    # the click (not through the whole hold and music fade).
    if events and last and outro is not None:
        voice_end = last["start"] + last["pre"] + last["speech"]
        click = total - OUTRO_TAIL
        ev = events[-1]
        ev["end"] = max(ev["start"] + 0.2, min(ev["end"], voice_end + LAST_CAPTION_HOLD, click - 0.3))
    return {"fps": fps, "scenes": scenes, "total": total, "total_frames": total_frames,
            "track": track, "events": events, "intro": intro, "outro": outro,
            "outro_at": total - OUTRO_TAIL if outro is not None else total,      # the click; sound stops here
            "voice_end": last["start"] + last["pre"] + last["speech"] if last else total,
            "audio_file": rec["file"] if rec else None}      # a recording is used as it is, at full quality


# ---- per-scene still clips -------------------------------------------------------------------
def _fit_mode(fit):
    # Older projects saved "auto"/"blur"; those now use contain so artwork is never cropped.
    return "cover" if fit == "cover" else "contain"


NEAR_FIT = 0.12      # aspect-ratio difference small enough to crop rather than letterbox


def _contain_box(size, cw, ch, scale=1.0):
    """Position (x, y, w, h) of the whole image fitted inside a cw x ch canvas, centered."""
    iw, ih = size
    f = min(cw * scale / iw, ch * scale / ih)
    bw, bh = max(2, int(iw * f) // 2 * 2), max(2, int(ih * f) // 2 * 2)
    return (cw - bw) // 2, (ch - bh) // 2, bw, bh


def segment_filter(frames, width, height, fps, fit, size=None):
    """FFmpeg graph that holds one still image for `frames` frames."""
    hold = f"zoompan=z=1:x=0:y=0:d={frames}"
    # Near-9:16 images (the generator sometimes returns e.g. 981x1602) fill the frame with a slight trim
    # instead of getting thin black bars.
    if fit == "contain" and size and abs((size[0] / size[1]) / (width / height) - 1) <= NEAR_FIT:
        fit = "cover"
    if fit == "contain":
        bx, by, bw, bh = _contain_box(size, width, height)
        return (f"[0:v]scale={bw}:{bh}:flags=lanczos,setsar=1,format=rgb24,{hold}:s={bw}x{bh}:fps={fps},"
                f"pad={width}:{height}:{bx}:{by}:color=black,scale=out_range=tv,format=yuv420p[v]")
    return (f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={width}:{height},setsar=1,{hold}:s={width}x{height}:fps={fps},"
            f"scale=out_range=tv,format=yuv420p[v]")


def render_segment(image, frames, width, height, fps, fit, preview, job):
    """Render (or reuse) one scene clip."""
    if not image:
        return render_blank(frames, width, height, fps, preview, job)
    st = Path(image).stat()
    fit = _fit_mode(fit)
    graph = segment_filter(frames, width, height, fps, fit, image_size(image))
    key = hashlib.sha1(json.dumps([SEGMENT_VERSION, str(image), st.st_mtime, st.st_size, graph,
                                   frames, preview]).encode())
    out = SEGMENT_DIR / f"{key.hexdigest()[:24]}.mp4"
    if out.exists():
        out.touch()
        return out
    # Intermediate clips: re-encoded once more in the final pass, so CRF 15 is visually lossless.
    preset, crf = ("veryfast", 23) if preview else ("veryfast", 15)
    tmp = out.with_name(out.stem + ".part.mp4")
    try:
        run_ffmpeg(["-i", str(image), "-filter_complex", graph,
                    "-map", "[v]", "-frames:v", str(frames), "-an", "-c:v", "libx264",
                    "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", str(tmp)], job)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


# Sound effects for the narration (id -> (label, FFmpeg audio filters)), named like ElevenLabs' audio effects.
# Each gets its character from which frequencies it keeps, with only gentle boosts and no hard clipping:
# loud boosts around 2-5 kHz (where the ear is most sensitive) made earlier versions painful to listen to.
VOICE_EFFECTS = {
    "none": ("None — clean voice", ""),
    # Measured: a download from the ElevenLabs website vs. the same voice, text and model from the API differ by this
    # curve (less boom, clearer mids, softer top).
    "website": ("ElevenLabs website sound", "highpass=f=130:poles=2,highpass=f=130:poles=2,"
                                            "equalizer=f=1100:t=q:w=0.6:g=8,lowpass=f=9000:poles=2"),
    # Fitted to the quote sample's voice (third-octave curve, within ~1 dB from 5-8 kHz): the clean ElevenLabs voice has
    # +11 to +18 dB more at 5-10 kHz, which sounds sharp / piercing ("matining"); this takes it down and keeps body.
    "soft": ("Soft (like the quote sample)", "highpass=f=100:poles=2,equalizer=f=190:t=q:w=1:g=-5,"
                                             "equalizer=f=400:t=q:w=2:g=3,equalizer=f=1600:t=q:w=3:g=-3,"
                                             "equalizer=f=3150:t=q:w=3:g=2,equalizer=f=5000:t=q:w=3:g=-18,"
                                             "highshelf=f=7000:t=q:w=0.7:g=-11"),
    # Soft's curve first (no piercing top), then a milder Old Radio band: warm, a bit nostalgic, still clear.
    "soft_radio": ("Soft + Old Radio", "highpass=f=100:poles=2,equalizer=f=190:t=q:w=1:g=-5,"
                                       "equalizer=f=400:t=q:w=2:g=3,equalizer=f=1600:t=q:w=3:g=-3,"
                                       "equalizer=f=3150:t=q:w=3:g=2,equalizer=f=5000:t=q:w=3:g=-18,"
                                       "highshelf=f=7000:t=q:w=0.7:g=-11,"
                                       "highpass=f=280:poles=2,highpass=f=280:poles=2,lowpass=f=4200:poles=2,"
                                       "lowpass=f=4200:poles=2,equalizer=f=1200:t=q:w=0.8:g=2,"
                                       "acompressor=threshold=0.15:ratio=2.5:attack=5:release=100,"
                                       "asoftclip=type=atan:param=0.6"),
    "radio": ("Old Radio", "highpass=f=400:poles=2,highpass=f=400:poles=2,lowpass=f=3500:poles=2,lowpass=f=3500:poles=2,"
                           "equalizer=f=1200:t=q:w=0.8:g=3,acompressor=threshold=0.15:ratio=3:attack=5:release=100,"
                           "asoftclip=type=atan:param=1"),
    "robot": ("Robot", "afftfilt=real='hypot(re,im)':imag='0':win_size=512:overlap=0.75,highpass=f=120,lowpass=f=6000"),
    "cheap_mic": ("Cheap Microphone", "highpass=f=250:poles=2,lowpass=f=6000:poles=2,equalizer=f=500:t=q:w=1:g=-4,"
                                      "equalizer=f=2500:t=q:w=1:g=2,acompressor=threshold=0.15:ratio=3:attack=5:release=100"),
    "phone": ("Phone", "highpass=f=300:poles=2,highpass=f=300:poles=2,lowpass=f=3400:poles=2,lowpass=f=3400:poles=2,"
                       "acompressor=threshold=0.15:ratio=3:attack=5:release=80"),
    "low_quality_phone": ("Low Quality Phone", "highpass=f=500:poles=2,highpass=f=500:poles=2,lowpass=f=2700:poles=2,"
                                               "lowpass=f=2700:poles=2,acrusher=bits=10:mode=log:aa=1:samples=2:mix=0.5,"
                                               "lowpass=f=3000,acompressor=threshold=0.15:ratio=3:attack=5:release=80"),
    "bright_phone": ("Bright Phone", "highpass=f=600:poles=2,highpass=f=600:poles=2,lowpass=f=4500:poles=2,"
                                     "equalizer=f=3000:t=q:w=1:g=2,acompressor=threshold=0.15:ratio=3:attack=5:release=80"),
    "warm": ("Warm low-cut (like the quote sample)", "highpass=f=150:poles=2"),
}
# Voice volume 100% = this loudness (LUFS), like reels on TikTok / Reels / Shorts (the quote sample: -14.1);
# 50% = 6 dB quieter, and so on. Default 40% (about -22 LUFS, a little louder than an ElevenLabs download).
LOUDNESS = -14.0


def voice_loudness(voice):
    """Target loudness (LUFS) for the voice volume slider (5-100 %)."""
    pct = max(5.0, min(100.0, float(voice.get("volume") if voice.get("volume") is not None else 40)))
    return LOUDNESS + 20 * math.log10(pct / 100)
# Tames sharp "s" sounds: cuts only the hiss band (FFmpeg's deesser also dulled and quietened the whole voice).
SOFTEN_S = "equalizer=f=6500:t=q:w=1.2:g=-6,equalizer=f=9000:t=q:w=1:g=-4"


def voice_filter(voice):
    """FFmpeg filter chain for the narration's sound effect, or "" for a clean voice."""
    chain = [SOFTEN_S] if voice.get("soften_s") else []
    fx = VOICE_EFFECTS.get(voice.get("effect") or "none", VOICE_EFFECTS["none"])[1]
    return ",".join(chain + ([fx] if fx else []))


def _lufs(path):
    """Integrated loudness (LUFS) as the ear hears it, or None if FFmpeg can't measure it."""
    err = subprocess.run([CONFIG["ffmpeg"], "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128", "-f", "null", "-"],
                         capture_output=True, text=True, creationflags=NO_WINDOW).stderr
    found = re.findall(r"I:\s+(-?[\d.]+) LUFS", err)
    return float(found[-1]) if found and float(found[-1]) > -70 else None


def voice_master_tag(voice):
    """Short id of how the narration is processed (effect + loudness), for cached file names."""
    return hashlib.sha1(f"{voice_filter(voice)}|{voice_loudness(voice):.2f}".encode()).hexdigest()[:8]


def apply_voice_effect(src, dst, voice):
    """Write `src` with the narration's sound effect to `dst`, at the voice volume's loudness (LUFS, as the ear
    hears it), with peaks kept 1.5 dB under full scale by a limiter (no clipping). Returns dst."""
    fx = voice_filter(voice) or "anull"
    target = voice_loudness(voice)
    wet = Path(dst).with_suffix(".wet.wav")
    try:
        subprocess.run([CONFIG["ffmpeg"], "-y", "-v", "error", "-i", str(src), "-af", fx,
                        "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le", str(wet)], check=True, creationflags=NO_WINDOW)
        moved = _lufs(wet)
        gain = min(target - moved, 30.0) if moved is not None else 0.0
        for _ in range(2):     # the limiter takes back a little loudness: a 2nd pass makes up the difference
            subprocess.run([CONFIG["ffmpeg"], "-y", "-v", "error", "-i", str(wet), "-af",
                            f"volume={gain:.2f}dB,alimiter=limit=0.84:attack=5:release=80:level=0",
                            "-c:a", "pcm_s16le", str(dst)], check=True, creationflags=NO_WINDOW)
            got = _lufs(dst)
            if got is None or got >= target - 0.5:
                break
            gain = min(gain + (target - got), 30.0)
    finally:
        wet.unlink(missing_ok=True)
    return dst


def _sound(path, vol):
    """A sound file as mono float32 at SAMPLE_RATE, times vol (0..1), or None if it's silent or missing."""
    if not vol or not Path(path or "").is_file():
        return None
    raw = subprocess.run([CONFIG["ffmpeg"], "-v", "error", "-i", str(path), "-ac", "1",
                          "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"], capture_output=True, check=True,
                         creationflags=NO_WINDOW).stdout
    audio = np.frombuffer(raw, np.float32) * vol
    return audio if len(audio) else None


def _sfx_volume(s):
    return max(0.0, min(100.0, float(s["intro"]["volume"]))) / 100


def intro_audio(s):
    """The intro sound (mono, SAMPLE_RATE, at its volume), or None when it's off or missing."""
    return _sound(s["intro"]["file"], _sfx_volume(s)) if s["intro"]["enabled"] else None


TAPE_FILE = SOUNDS_DIR / "cassette_run.wav"    # the tape running after Play: a seamless loop
TAPE_XFADE = 0.7                                 # s: the loop comes in while the intro's own hiss fades out
OUTRO_FILE = SOUNDS_DIR / "radio_off.wav"      # the radio clicked off at the end
LAST_CAPTION_HOLD = 1.0                          # s the last caption stays after the last word
OUTRO_TAIL = 0.5                                 # s the click gets at the very end of the video
# Built-in intro sounds: (file, name, length s, when the voice starts s).
INTRO_SOUNDS = [(SOUNDS_DIR / "cassette_play.wav", "Cassette play", 2.4, 0.7),
                (SOUNDS_DIR / "radio_tuning.wav", "Radio tuning in", 1.0, 0.6)]


def tape_audio(s):
    """The running-tape loop (mono, SAMPLE_RATE, at the intro volume), or None when it's off."""
    return _sound(TAPE_FILE, _sfx_volume(s)) if s["intro"].get("tape") else None


def outro_audio(s):
    """The radio's off click, or None when the sign-off click is off."""
    return _sound(OUTRO_FILE, _sfx_volume(s)) if s["outro"].get("click") else None


def add_sounds(src, dst, intro=None, tape=None, outro=None, stop_at=None, fade_end=1.0):
    """`src` (the finished narration) with the intro sound at its start, the tape loop running from the
    end of the intro until `stop_at` s (fading out over the last `fade_end` s before it), and the off
    click at `stop_at`. Returns dst."""
    voice, sr = sf.read(str(src), dtype="float32")
    stop = len(voice) if stop_at is None else min(len(voice), int(round(stop_at * sr)))
    if intro is not None:
        n = min(len(voice), len(intro))
        voice[:n] += intro[:n]
    if tape is not None and intro is not None:
        start = max(0, len(intro) - int(TAPE_XFADE * sr))
        run = np.resize(tape, max(0, stop - start))      # repeats the loop as often as needed
        fin = min(len(run), int(TAPE_XFADE * sr))
        run[:fin] *= np.sin(np.linspace(0, np.pi / 2, fin))
        fout = min(len(run), int(max(0.1, fade_end) * sr))
        run[len(run) - fout:] *= np.linspace(1, 0, fout)
        voice[start:stop] += run
    if outro is not None:
        n = min(len(voice) - stop, len(outro))
        voice[stop:stop + n] += outro[:n]
    sf.write(str(dst), np.clip(voice, -0.98, 0.98), sr)
    return dst


PULSE_ATTACK = 0.04       # s: eased rise on a word (not an instant jump, which looks like twitching)
PULSE_RELEASE = 0.18      # s: soft settle after it
PULSE_GATE = 0.25         # below this share of the voice's loud level it rests, so it clearly bounces
SPEAKER_PUSH = 5.0        # speakers mode: the cones move this many times the pulse setting (3 -> 15 %)
# Whole mode: a bass thump, wider more than taller (squash and stretch), planted on its bottom edge.
WHOLE_PUSH_W, WHOLE_PUSH_H = 2.0, 1.1     # x the pulse setting: 3 -> 6 % wider, 3.3 % taller


def overlay_speakers(file):
    """Speaker discs of an overlay picture [(x, y, r) in its own pixels], from the .json next to it
    (e.g. overlays/radio_cassette.json); [] when it has none."""
    side = Path(file).with_suffix(".json")
    try:
        return [(float(s["x"]), float(s["y"]), float(s["r"])) for s in json.loads(side.read_text(encoding="utf-8"))["speakers"]]
    except (OSError, ValueError, KeyError, TypeError):
        return []


CLASSIC_RELEASE = 0.12    # classic mode: the first version (instant rise, this settle, no gate)


def write_pulse(path, track, fps, total, targets, classic=False):
    """FFmpeg sendcmd file making each target [(scale filter name, (w, h) px, growth % wide, growth % tall)]
    grow with the voice in `track`, per video frame: an eased rise on a word (PULSE_ATTACK), a soft settle
    (PULSE_RELEASE). False when there's no voice."""
    if track is None or not targets:
        return False
    hop = SAMPLE_RATE / fps
    frames = int(math.ceil(total * fps))
    rms = np.array([np.sqrt(np.mean(track[int(i * hop):int((i + 1) * hop)] ** 2) + 1e-12) for i in range(frames)])
    loud = np.percentile(rms[rms > 1e-4], 95) if (rms > 1e-4).any() else 0
    if not loud:
        return False
    env, level = np.zeros(frames), 0.0
    rise, fall = 1 - math.exp(-1 / (PULSE_ATTACK * fps)), math.exp(-1 / (PULSE_RELEASE * fps))
    # Quiet parts (between words, soft endings) rest at 0 and stressed syllables reach the full push: a
    # clear bounce instead of hovering half-pushed through a whole sentence.
    if classic:
        fall = math.exp(-1 / (CLASSIC_RELEASE * fps))
        for i, v in enumerate(np.clip(rms / loud, 0, 1)):
            level = v if v > level else level * fall
            env[i] = level
    for i, v in ([] if classic else enumerate(np.clip((rms / loud - PULSE_GATE) / (1 - PULSE_GATE), 0, 1) ** 0.8)):
        level = level + (v - level) * rise if v > level else level * fall
        env[i] = level
    lines, last = [], None
    for i, e in enumerate(env):
        sizes = tuple((max(2, round(w * (1 + pw / 100 * e) / 2) * 2), max(2, round(h * (1 + ph / 100 * e) / 2) * 2))
                      for _, (w, h), pw, ph in targets)
        if sizes != last:
            cmds = ", ".join(f"scale@{t[0]} w {sw}, scale@{t[0]} h {sh}" for t, (sw, sh) in zip(targets, sizes))
            lines.append(f"{i / fps:.4f} {cmds};")
            last = sizes
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def overlay_layout(ov, width, height, src_size):
    """(w, h, x, y) of the overlay in the video: its size and the center point, in pixels."""
    w = max(2, round(width * max(1.0, min(100.0, float(ov["size"]))) / 100 / 2) * 2)
    sw, sh = src_size or (w, w)
    h = max(2, round(w * sh / sw / 2) * 2)
    x = width * max(0.0, min(100.0, float(ov["position_x"]))) / 100
    y = height * max(0.0, min(100.0, float(ov["position_y"]))) / 100
    return w, h, x, y


def pulse_speakers(ov, src_size):
    """The speaker discs to pump, when the pulse is in speakers mode and the picture has speaker info."""
    return overlay_speakers(ov["file"]) if ov.get("pulse_mode") == "speakers" and src_size else []


def pulse_targets(ov, width, height, src_size):
    """What grows with the voice: the whole picture and its shadow (scale@pulse, scale@pulsesh), or in
    speakers mode each speaker disc (scale@spk0, scale@spk1, ...)."""
    pct = max(0.0, float(ov.get("pulse", 0)))
    if pct <= 0 or ov.get("pulse_mode", "off") == "off":
        return []
    w, h, _, _ = overlay_layout(ov, width, height, src_size)
    speakers = pulse_speakers(ov, src_size)
    if speakers:
        k = w / src_size[0]
        return [(f"spk{i}", (max(2, round(2 * r * k / 2) * 2),) * 2, pct * SPEAKER_PUSH, pct * SPEAKER_PUSH)
                for i, (_, _, r) in enumerate(speakers)]
    if ov.get("pulse_mode", "off") == "classic":
        return [("pulse", (w, h), pct, pct)]
    targets = [("pulse", (w, h), pct * WHOLE_PUSH_W, pct * WHOLE_PUSH_H)]
    if ov.get("shadow", True):
        targets.append(("pulsesh", (w, h), pct * WHOLE_PUSH_W, pct * WHOLE_PUSH_H))
    return targets


def overlay_filters(video, src, ov, width, height, out, fps=30, pulse=None, grade="", k=1.0, src_size=None):
    """Put input `src` (a picture or looping GIF, transparency kept) over `video`: `size` % of the frame
    width, centered at (position_x, position_y) % of the frame, with a soft shadow under it and the color
    grade on it (when on). With the sendcmd file `pulse`, it bounces with the voice from its bottom edge
    (its shadow with it), or in speakers mode only its speaker cones push out. Ends with the video.
    Output label [out]."""
    w, h, x, y = overlay_layout(ov, width, height, src_size)
    alpha = max(0.0, min(100.0, float(ov["opacity"]))) / 100
    speakers = pulse_speakers(ov, src_size) if pulse else []
    classic = bool(pulse) and ov.get("pulse_mode", "off") == "classic"
    whole = bool(pulse) and not speakers and not classic
    # Resized-by-command pictures go straight into their overlay: any filter after the scale would make
    # FFmpeg add a converter there, which quietly scales every frame back to the first size.
    # (Classic mode is the first version, kept as it was: resized at the start, before the grade.)
    resize = f"sendcmd=f={pulse},scale@pulse=w={w}:h={h}:flags=bicubic" if classic else f"scale={w}:{h}:flags=lanczos"
    parts = [f"[{src}:v]fps={fps},format=rgba,{resize}[ov0]"]
    last = "ov0"
    if grade and ov.get("graded", True):      # grade the colors, keep the transparency
        mix = (f"[ovc]format=gbrp,split[ovo][ovg];[ovg]{grade}[ovgg];"
               f"[ovo][ovgg]blend=all_expr='A*{1 - k:.3f}+B*{k:.3f}',format=gbrp[ovmix]" if k < 0.999 else
               f"[ovc]format=gbrp,{grade},format=gbrp[ovmix]")
        parts.append(f"[{last}]split[ovc][ova];[ova]format=rgba,alphaextract[ovalpha];{mix};[ovmix][ovalpha]alphamerge,format=rgba[ov1]")
        last = "ov1"
    if alpha < 1:
        parts.append(f"[{last}]colorchannelmixer=aa={alpha:.3f}[ov2]")
        last = "ov2"
    if speakers:          # copies of the radio for the speaker discs, cut out below
        parts.append(f"[{last}]format=rgba,split={len(speakers) + 1}[ovr]" + "".join(f"[spc{i}]" for i in range(len(speakers))))
        last = "ovr"
    # A bouncing picture keeps its bottom edge where it is (it stands on something): it grows upward.
    top = f"{y + h / 2:.1f}-h" if whole else f"{y:.1f}-h/2"
    base = video
    if ov.get("shadow", True):        # a soft dark copy, a little lower, under it (bouncing with it)
        blur = max(2, round(w * 0.012))
        drop = round(h * 0.035)
        stop = f"{y + h / 2 + drop:.1f}-h" if whole else f"{y + drop:.1f}-h/2"
        bounce = ",scale@pulsesh=w={w}:h={h}:flags=bicubic".format(w=w, h=h) if whole else ""
        parts.append(f"[{last}]format=rgba,split[ovs][ovm];[ovs]colorchannelmixer=rr=0:rg=0:rb=0:gr=0:gg=0:gb=0:br=0:bg=0:bb=0:"
                     f"aa={0.55 * alpha:.3f},format=yuva420p,boxblur=luma_radius={blur}:chroma_radius={max(1, blur // 2)}:"
                     f"alpha_radius={blur}{bounce}[ovsh];"
                     f"{video}[ovsh]overlay=x={x:.1f}-w/2:y={stop}:shortest=1:format=auto[vsh]")
        base, last = "[vsh]", "ovm"
    final = "vrad" if speakers else out
    if whole:
        parts.append(f"[{last}]sendcmd=f={pulse},scale@pulse=w={w}:h={h}:flags=bicubic[ovp]")
        last = "ovp"
    parts.append(f"{base}[{last}]overlay=x={x:.1f}-w/2:y={top}:shortest=1:format=auto[{final}]")
    # Speakers mode: each disc cut out of the radio as a soft-edged circle, scaled from its own center
    # with the voice, and laid back exactly over its speaker.
    kx = w / src_size[0] if speakers else 1
    left, upper = x - w / 2, y - h / 2
    cur = final
    for i, (sx, sy, r) in enumerate(speakers):
        d = max(2, round(2 * r * kx / 2) * 2)
        cx, cy = sx * kx, sy * kx
        cmd = f"sendcmd=f={pulse}," if i == 0 else ""
        mask = "alpha(X,Y)*clip((W/2-1.5-hypot(X-W/2,Y-H/2))/1.5,0,1)"
        nxt = out if i == len(speakers) - 1 else f"vspk{i}"
        parts.append(f"[spc{i}]crop={d}:{d}:{max(0, round(cx - d / 2))}:{max(0, round(cy - d / 2))},format=rgba,"
                     f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='{mask}',{cmd}"
                     f"scale@spk{i}=w={d}:h={d}:flags=bicubic[spk{i}];"
                     f"[{cur}][spk{i}]overlay=x={left + cx:.1f}-w/2:y={upper + cy:.1f}-h/2:shortest=1:format=auto[{nxt}]")
        cur = nxt
    return parts


# The music as if it came out of the radio too: less deep bass and no airy top, a little mid push.
MUSIC_RADIO = ("highpass=f=150:poles=2,lowpass=f=5000:poles=2,lowpass=f=5000:poles=2,"
               "equalizer=f=1200:t=q:w=1:g=2")


def music_filters(nar, mus, music, total, end=None):
    """Mix looping background music under the narration: set volume, lower it while the narrator
    speaks (optional), fade it out so it's gone at `end` s (the radio's off click; default the video's
    end). Output label [aout], exactly as long as the video."""
    fmt = "aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"
    end = total if end is None else min(end, total)
    vol = max(0.0, min(100.0, float(music["volume"]))) / 100
    fade = max(0.0, min(float(music["fade_out"]), end))
    radio = f"{MUSIC_RADIO}," if music.get("radio") else ""
    parts = [f"{nar}{fmt},asplit=2[nar][key]" if music["duck"] else f"{nar}{fmt}[nar]",
             f"[{mus}:a]{fmt},{radio}volume={vol:.3f}[m0]"]
    last = "m0"
    if music["duck"]:
        parts.append(f"[m0][key]sidechaincompress=threshold=0.03:ratio=4:attack=30:release=600[m1]")
        last = "m1"
    if fade > 0 or end < total:
        fade = max(fade, 0.03)
        parts.append(f"[{last}]afade=t=out:st={end - fade:.3f}:d={fade:.3f}[m2]")
        last = "m2"
    # The music adds to the voice's loudness: the limiter keeps the peaks of the mix from clipping.
    parts.append(f"[nar][{last}]amix=inputs=2:duration=first:normalize=0,"
                 f"alimiter=limit=0.84:attack=5:release=80:level=0[aout]")
    return parts


def visualizer_filters(video, src, vis, width, height, fps, out):
    """The voice from input `src` drawn as thin vertical bars (FFmpeg showwaves), tallest in the middle
    and fading out to both ends, centered at (position_x, position_y) % of the frame. Output label [out]."""
    bar = max(2, round(width / 360))          # px per bar (3 at 1080p), then a 1 px gap
    step = bar + 1
    w = max(16, round(width * max(5.0, min(90.0, float(vis["width"]))) / 100 / step) * step)
    h = max(8, round(w * 0.22 / 2) * 2)
    x = width * max(0.0, min(100.0, float(vis["position_x"]))) / 100
    y = height * max(0.0, min(100.0, float(vis["position_y"]))) / 100
    color = (vis.get("color") or "#FFE9C7").lstrip("#")
    # Drawn at one column per bar, widened to whole bars, then cut into bars with gaps; tallest in the
    # middle, fading out to both ends.
    return [f"[{src}:a]aformat=channel_layouts=mono,volume=2,"
            f"showwaves=s={w // step}x{h}:mode=cline:rate={fps}:colors=0x{color}:scale=sqrt:draw=full,"
            f"scale={w}:{h}:flags=neighbor,format=rgba,"
            f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='alpha(X,Y)*0.92*sin(PI*X/W)*lt(mod(X,{step}),{bar})'[wv];"
            f"{video}[wv]overlay=x={x:.1f}-w/2:y={y:.1f}-h/2:shortest=1:format=auto[{out}]"]


def texture_filter(texture, width):
    """Film grain + vignette for the picture ('' when both are off)."""
    grain = max(0.0, min(100.0, float(texture.get("grain", 0))))
    vig = max(0.0, min(100.0, float(texture.get("vignette", 0))))
    out = []
    if vig > 0:
        out.append(f"vignette=angle={math.pi / 2.2 * vig / 100:.4f}")
    if grain > 0:
        out.append(f"noise=alls={grain * 0.25 * width / 1080:.1f}:allf=t")
    return ",".join(out)


def render_blank(frames, width, height, fps, preview, job):
    """A black clip for a scene that has no image yet, so the video can be exported anyway."""
    key = hashlib.sha1(json.dumps([SEGMENT_VERSION, "blank", frames, width, height, fps, preview]).encode())
    out = SEGMENT_DIR / f"{key.hexdigest()[:24]}.mp4"
    if out.exists():
        out.touch()
        return out
    tmp = out.with_name(out.stem + ".part.mp4")
    try:
        run_ffmpeg(["-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}",
                    "-frames:v", str(frames), "-an", "-c:v", "libx264", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", "-color_range", "tv", str(tmp)], job)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


# ---- full render -----------------------------------------------------------------------------
def render(project, narrator, job, preview):
    """Render the project. Preview = half resolution, fast encode, same timeline."""
    if not CONFIG.get("ffmpeg"):
        raise RenderError("FFmpeg was not found. Install it or set 'ffmpeg' in config.json.")
    validate(project)
    s = merged_settings(project)
    narration = narrate(project, narrator, job, (0.0, 0.25))
    tl = build_timeline(project, narration)
    fps = tl["fps"]
    width, height = parse_resolution(s["export"]["resolution"])
    if preview:
        width, height = width // 2, height // 2
        width, height = width - width % 2, height - height % 2
    work = JOBS_DIR / job.id
    work.mkdir(parents=True, exist_ok=True)
    final_tmp = None
    try:
        if tl["audio_file"] is None:
            sf.write(str(work / "narration.wav"), tl["track"], SAMPLE_RATE)
        subs, wm, station, outro = s["subtitles"], s["watermark"], s["station"], s["outro"]
        # The sign-off line: under the last caption from when the voice is done until the end.
        sign_off = ({"text": outro["text"].strip(), "start": tl["voice_end"], "end": tl["total"]}
                    if outro.get("text_enabled") and (outro.get("text") or "").strip() and not tl["audio_file"] else None)
        vis = s["visualizer"] if s["visualizer"].get("enabled") else None
        burn = (subs["enabled"] or (wm["enabled"] and wm["text"].strip()) or sign_off or vis
                or (station["enabled"] and station["text"].strip()))
        if burn:
            write_ass(work / "subs.ass", tl["events"] if subs["enabled"] else [], subs, width, height,
                      watermark=wm, duration=tl["total"], station=station, sign_off=sign_off, visualizer=vis)

        clips, n = [], len(tl["scenes"])
        for i, sc in enumerate(tl["scenes"]):
            if job.cancelled():
                raise Cancelled()
            job.update(f"Rendering scene {i + 1} of {n}", 0.25 + 0.35 * i / n)
            clips.append(render_segment(project["scenes"][i]["image"], sc["frames"], width, height, fps,
                                        s["motion"]["fit"], preview, job))

        args = []
        for clip in clips:
            args += ["-i", str(clip)]
        narration = tl["audio_file"] or str(work / "narration.wav")
        narration = str(apply_voice_effect(narration, work / "narration_fx.wav", s["voice"]))   # voice effect, level-matched
        music = s["music"]
        use_music = music["enabled"] and music["file"] and float(music["volume"]) > 0 and not tl["audio_file"]
        if tl["intro"] is not None or tl["outro"] is not None:   # after the voice effect: the sounds stay as they are
            narration = str(add_sounds(narration, work / "narration_sfx.wav", tl["intro"], tape_audio(s),
                                       tl["outro"], stop_at=tl["outro_at"],
                                       fade_end=float(music["fade_out"]) if use_music else 1.0))
        args += ["-i", narration]
        if use_music:
            args += ["-stream_loop", "-1", "-i", str(music["file"])]      # loops for the whole video
        ov = s["overlay"]
        ov_input = None
        if ov["enabled"] and Path(ov["file"] or "").is_file() and float(ov["opacity"]) > 0:
            ov_input = n + 1 + (1 if use_music else 0)
            loop = ["-ignore_loop", "0"] if Path(ov["file"]).suffix.lower() == ".gif" else ["-loop", "1"]
            args += loop + ["-i", str(ov["file"])]       # a GIF keeps looping, a still picture stays
        vis_input = None
        if vis:                       # the narration once more, drawn as moving bars
            vis_input = n + 1 + (1 if use_music else 0) + (1 if ov_input is not None else 0)
            args += ["-i", narration]
        # The color grade is applied to every scene clip before they are joined, ending in YUV again.
        grade = color_grade_filter(s["color"], work)
        parts = []
        for i in range(n):
            if grade:
                parts.append(f"[{i}:v]settb=AVTB,setpts=PTS-STARTPTS,format=gbrp,split[go{i}][gs{i}];"
                             f"[gs{i}]{grade}[gg{i}];[go{i}][gg{i}]blend=all_expr='A*{1 - grade_k(s['color']):.3f}+"
                             f"B*{grade_k(s['color']):.3f}',format=yuv420p[v{i}]"
                             if grade_k(s["color"]) < 0.999 else
                             f"[{i}:v]settb=AVTB,setpts=PTS-STARTPTS,format=gbrp,{grade},format=yuv420p[v{i}]")
            else:
                parts.append(f"[{i}:v]settb=AVTB,setpts=PTS-STARTPTS,format=yuv420p[v{i}]")
        if n == 1:
            last = "v0"
        else:
            parts.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vc]")
            last = "vc"
        if ov_input is not None:     # on top of the pictures, under the captions
            size = image_size(ov["file"])
            pulse = "pulse.txt" if write_pulse(work / "pulse.txt", tl["track"], fps, tl["total"],
                                               pulse_targets(ov, width, height, size),
                                               classic=ov.get("pulse_mode", "off") == "classic") else None
            parts += overlay_filters(f"[{last}]", ov_input, ov, width, height, "vov", fps=fps, pulse=pulse,
                                     grade=grade, k=grade_k(s["color"]), src_size=size)
            last = "vov"
        if vis_input is not None:     # bars that move with the voice
            parts += visualizer_filters(f"[{last}]", vis_input, vis, width, height, fps, "vvis")
            last = "vvis"
        texture = texture_filter(s["texture"], width)
        if texture:                  # over the picture and the overlay, under the captions
            parts.append(f"[{last}]{texture}[vtx]")
            last = "vtx"
        sub_filter = "ass=subs.ass," if burn else ""
        parts.append(f"[{last}]{sub_filter}scale=out_range=tv,format=yuv420p[out]")  # JPGs are full-range; output standard TV range
        audio_map = f"{n}:a"
        if use_music:
            parts += music_filters(f"[{n}:a]", n + 1, music, tl["total"], end=tl["outro_at"])
            audio_map = "[aout]"
        if preview:
            preset, crf = "ultrafast", 26
        else:
            preset, crf = QUALITY.get(s["export"]["quality"], QUALITY["high"])
        enc = ["-filter_complex", ";".join(parts), "-map", "[out]", "-map", audio_map,
               "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "high",
               "-pix_fmt", "yuv420p", "-color_range", "tv", "-r", str(fps), "-c:a", "aac", "-b:a", "192k",
               "-ar", "48000", "-ac", "2", "-t", f"{tl['total']:.6f}", "-movflags", "+faststart"]

        if preview:
            for old in PREVIEW_DIR.glob("preview_*.mp4"):
                old.unlink(missing_ok=True)
            final = PREVIEW_DIR / f"preview_{job.id}.mp4"
        else:
            out_dir = Path(s["export"]["output_dir"] or EXPORTS_DIR)
            out_dir.mkdir(parents=True, exist_ok=True)
            base = _safe_name(s["export"]["filename"] or project.get("name") or "StoryCraft")
            final = out_dir / f"{base}.mp4"
            k = 2
            while final.exists():
                final = out_dir / f"{base} ({k}).mp4"
                k += 1
        final_tmp = final.with_name(final.stem + ".partial.mp4")
        job.update("Assembling video and subtitles", 0.6)
        run_ffmpeg(args + enc + [str(final_tmp)], job, cwd=str(work), duration=tl["total"],
                   on_progress=lambda f: job.update(f"Assembling video: {int(f * 100)}%",
                                                    0.6 + 0.4 * f))
        final_tmp.replace(final)
        final_tmp = None
        return {"path": str(final), "duration": round(tl["total"], 3), "width": width,
                "height": height, "scenes": scene_summary(tl)}
    finally:
        if final_tmp is not None:
            Path(final_tmp).unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)


RESOLUTIONS = {"1080x1920", "720x1280"}     # always vertical 9:16 (Shorts / Reels)


def lut_kind(path):
    """'lut3d' or 'lut1d' for a .cube file, None if it isn't one."""
    try:
        head = Path(path).read_text(encoding="utf-8", errors="replace")[:20000].upper()
    except OSError:
        return None
    return "lut3d" if "LUT_3D_SIZE" in head else "lut1d" if "LUT_1D_SIZE" in head else None


def lut_path(color):
    """The .cube file to use: your own file, else the built-in Look, else '' (original colors)."""
    if color.get("lut"):
        return color["lut"]
    return str(LUTS_DIR / f"{color['look']}.cube") if color.get("look") else ""


def grade_k(color):
    """Color grade strength, 0..1 (below 1 the graded picture is mixed with the original)."""
    return max(0.0, min(float(color.get("strength", 100)), 100.0)) / 100


def color_grade_filter(color, work):
    """The LUT filter for the chosen color grade ('' for none). Captions are added later, so they keep
    their own colors."""
    lut = lut_path(color)
    kind = lut_kind(lut) if lut else None
    if not kind or grade_k(color) <= 0:
        return ""
    shutil.copyfile(lut, work / "grade.cube")      # a plain relative name: no Windows path escaping in the filter
    return f"{kind}=file=grade.cube"


def parse_resolution(value):
    """'1080x1920' -> (1080, 1920). A 16:9 value from an older project is turned upright;
    anything unknown falls back to 1080x1920."""
    w, _, h = str(value).partition("x")
    value = f"{h}x{w}" if f"{h}x{w}" in RESOLUTIONS else value
    value = value if value in RESOLUTIONS else "1080x1920"
    w, h = value.split("x")
    return int(w), int(h)


def scene_summary(tl):
    return [{"id": sc["id"], "duration": round(sc["duration"], 3), "speech": round(sc["speech"], 3),
             "start": round(sc["start"], 3), "audio_url": sc["audio_url"],
             "captions": [{"start": round(c["start"], 3), "end": round(c["end"], 3), "text": c["text"]}
                          for c in sc["captions"]]} for sc in tl["scenes"]]


def _safe_name(name):
    cleaned = "".join(ch for ch in name if ch not in '<>:"/\\|?*').strip().rstrip(".")
    return cleaned or "StoryCraft"


SEGMENT_CACHE_MB = 300
SEGMENT_MAX_AGE_DAYS = 7


def trim_segments(max_mb=SEGMENT_CACHE_MB, max_age_days=SEGMENT_MAX_AGE_DAYS):
    """Keep the clip cache small: drop clips older than a week, then the least recently used ones
    until the cache is under max_mb. Clips used in the last 10 minutes (the current render) stay."""
    now = time.time()
    files = []
    for f in SEGMENT_DIR.glob("*.mp4"):
        try:
            st = f.stat()
        except OSError:
            continue
        if f.name.endswith(".part.mp4") or st.st_mtime < now - max_age_days * 86400:
            f.unlink(missing_ok=True)
        else:
            files.append((st.st_mtime, st.st_size, f))
    total = sum(size for _, size, _ in files)
    for mtime, size, f in sorted(files):            # oldest use first
        if total <= max_mb * 1024 * 1024 or mtime > now - 600:
            break
        f.unlink(missing_ok=True)
        total -= size


def _dir_size(d, pattern="*"):
    return sum(f.stat().st_size for f in d.glob(pattern) if f.is_file())


def cache_info():
    return {"clips_mb": round(_dir_size(SEGMENT_DIR) / 1048576, 1),
            "previews_mb": round(_dir_size(PREVIEW_DIR) / 1048576, 1),
            "narration_mb": round((_dir_size(AUDIO_DIR) + _dir_size(WORKSPACE / "cache" / "tts")) / 1048576, 1)}


def clear_video_cache():
    """Delete cached clips and old previews (rebuilt automatically). Narration cache is kept."""
    for f in list(SEGMENT_DIR.glob("*.mp4")) + list(PREVIEW_DIR.glob("*.mp4")):
        f.unlink(missing_ok=True)


def cleanup_cache():
    """At startup: trim the clip cache and remove temp folders left by an interrupted render."""
    trim_segments()
    for d in JOBS_DIR.iterdir():
        shutil.rmtree(d, ignore_errors=True)
