"""Split narration scripts into sentences and short caption phrases, and write ASS subtitles.

The script text is never rewritten: captions show the words exactly as typed.
"""
import re

ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "etc", "prof", "mt", "no",
                 "approx", "e.g", "i.e", "a.m", "p.m", "u.s", "lt", "sgt", "capt", "gen"}
SOFT_BREAK = re.compile(r"[,;:—–…、，；：]+[\"'”’)\]」』）]*$|-{2,}$")
HAS_WORD = re.compile(r"\w")

# Japanese / Chinese: no spaces between words, and their own punctuation (。！？ end a sentence, 、 is a comma).
CJK = re.compile(r"[぀-ヿ㐀-鿿豈-﫿ｦ-ﾟ]")
CJK_CHUNK = 3          # characters per "word" for spaceless text (captions and timing work on these)
CJK_CAPTION = 9        # about this many characters per caption line for spaceless text


def split_words(text):
    """Words of a sentence. Normal text: split at spaces. Spaceless text (Japanese, Chinese): short chunks
    of CJK_CHUNK characters, cut after each 、。！？ so punctuation stays with the chunk before it.
    Splitting text joined with spaces gives the same words as splitting each part (final_sentence relies on it)."""
    out = []
    for run in (text or "").split():
        if not CJK.search(run):
            out.append(run)
            continue
        for clause in re.findall(r"[^、，。！？!?…]+[、，。！？!?…」』）\"”’)]*|[、，。！？!?…」』）\"”’)]+", run):
            m = re.match(r"(.*?)([、，。！？!?…」』）\"”’)]*)$", clause)
            body, punct = m.group(1), m.group(2)
            pieces = [body[k:k + CJK_CHUNK] for k in range(0, len(body), CJK_CHUNK)] or [""]
            if len(pieces) > 1 and len(pieces[-1]) == 1:       # no lone 1-character chunk at the end
                pieces[-2] += pieces.pop()
            pieces[-1] += punct
            if pieces[-1] and not body and out and CJK.search(out[-1]):
                out[-1] += pieces[-1]                           # punctuation alone joins the chunk before
            else:
                out += [p for p in pieces if p]
    return out


def join_words(words):
    """Inverse of split_words for display: no space between two Japanese/Chinese chunks."""
    text = ""
    for w in words:
        text += w if text and _cjk_edge(text[-1], w[0]) else (" " + w if text else w)
    return text


def _cjk_edge(a, b):
    """True when the boundary between characters a and b is inside spaceless text."""
    cjk_or_punct = lambda c: bool(CJK.search(c)) or c in "、，。！？」』）「『（…"  # noqa: E731
    return cjk_or_punct(a) and cjk_or_punct(b)


def split_sentences(text, lines=False):
    """Return a list of sentences. Blank lines and sentence punctuation both end a sentence;
    with lines=True (poems) every line break does too."""
    sentences = []
    blocks = re.split(r"\n\s*\n|\r\n\s*\r\n", text.strip())
    if lines:
        blocks = [line for block in blocks for line in block.splitlines()]
    for paragraph in blocks:
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        # Cut after . ! ? or an ellipsis (plus any closing quotes) followed by whitespace.
        parts, start = [], 0
        for match in re.finditer(r"[.!?…]+[\"'”’)\]]*\s+|[。！？]+[\"'”’)\]」』）]*\s*", paragraph):
            parts.append(paragraph[start:match.end()].strip())
            start = match.end()
        parts.append(paragraph[start:].strip())
        current = ""
        for part in filter(None, parts):
            current = f"{current} {part}".strip() if current else part
            last_word = current.split()[-1].rstrip(".").lower().strip("\"'(“")
            # "Mr. Smith" or an initial like "J." should not end a sentence.
            if last_word in ABBREVIATIONS or (len(last_word) == 1 and last_word.isalpha()):
                continue
            sentences.append(current)
            current = ""
        if current:
            sentences.append(current)
    return sentences


WEAK_ENDINGS = {"a", "an", "the", "of", "to", "in", "on", "at", "for", "with", "and", "or", "but",
                "your", "my", "his", "her", "their", "our", "its", "from", "by", "into", "as", "is",
                "was", "that", "this", "very", "so", "be"}


def _bare(word):
    return re.sub(r"[^\w']", "", word).lower()


def _split_clause(indexes, words, target):
    """Split one clause into balanced chunks, nudging cuts so a chunk doesn't end on 'the', 'of'..."""
    n = len(indexes)
    if n <= target + 1:
        return [indexes]
    k = -(-n // target)
    cuts, prev = [], 0
    for j in range(1, k):
        cut = round(j * n / k)
        for candidate in (cut, cut + 1, cut - 1):
            if prev + 2 <= candidate <= n - 2 and _bare(words[indexes[candidate - 1]]) not in WEAK_ENDINGS:
                cut = candidate
                break
        cuts.append(cut)
        prev = cut
    bounds = [0, *cuts, n]
    return [indexes[a:b] for a, b in zip(bounds, bounds[1:]) if b > a]


def split_phrases(words, target):
    """Group a sentence's words into caption phrases of about `target` words.

    Returns lists of word indexes. Clauses (split at , ; : dashes) are kept together when short,
    tiny clauses join the next one, and long clauses are split into even chunks."""
    target = max(1, int(target))
    clauses, current = [], []
    for i, word in enumerate(words):
        current.append(i)
        if SOFT_BREAK.search(word):
            clauses.append(current)
            current = []
    if current:
        clauses.append(current)
    merged = []
    for clause in clauses:
        # A punctuation-only "word" (a lone dash) belongs with what came before.
        if merged and not any(HAS_WORD.search(words[i]) for i in clause):
            merged[-1].extend(clause)
        elif merged and len(merged[-1]) <= 2 and len(merged[-1]) + len(clause) <= 2 * target:
            merged[-1].extend(clause)   # "Naturally," + "you did what..." read better together
        else:
            merged.append(clause)
    phrases = []
    for clause in merged:
        phrases += _split_clause(clause, words, target)
    return phrases


def build_captions(words, word_times, target, offset=0.0):
    """Turn per-word timings into caption events: [{'start','end','text'}] (seconds)."""
    events = []
    cjk = [w for w in words if CJK.search(w)]
    if cjk and len(cjk) * 2 >= len(words):      # mostly spaceless text: aim for ~CJK_CAPTION characters
        target = max(1, round(CJK_CAPTION / (sum(map(len, cjk)) / len(cjk))))
    for group in split_phrases(words, target):
        start = word_times[group[0]][0]
        end = max(word_times[i][1] for i in group)
        events.append({"start": offset + start, "end": offset + end,
                       "text": join_words(words[i] for i in group)})
    return events


def _ass_time(t):
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_color(hex_color, alpha=0):
    value = (hex_color or "#FFFFFF").lstrip("#")
    if len(value) != 6:
        value = "FFFFFF"
    r, g, b = value[0:2], value[2:4], value[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def _ass_escape(text):
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def caption_margins(style, play_w=1080, play_h=1920):
    """Caption box from the X/Y/width settings (same rule as the browser preview).

    position_x: box center, % from the left. max_width: box width, % of the frame.
    position: bottom of the text, % from the bottom. Near an edge the box narrows (text wraps
    more) so the caption really is centered at X; it never leaves the frame."""
    x = min(max(float(style.get("position_x", 50)), 0.0), 100.0)
    half_pct = min(float(style.get("max_width", 84)) / 2, x, 100 - x)
    half_pct = max(half_pct, 12.0)                     # keep at least ~24% width for a word or two
    center_pct = min(max(x, half_pct), 100 - half_pct)
    half, center = play_w * half_pct / 100, play_w * center_pct / 100
    margin_v = int(play_h * float(style.get("position", 26)) / 100)
    return int(round(center - half)), int(round(play_w - center - half)), margin_v


# Watermark anchor -> ASS n alignment (which point of the text sits at X/Y).
WATERMARK_ANCHORS = {"tl": 7, "tc": 8, "tr": 9, "ml": 4, "c": 5, "mr": 6, "bl": 1, "bc": 2, "br": 3}


def watermark_line(wm, play_w, play_h, duration):
    """One Dialogue line showing the watermark for the whole video. X/Y are % of the frame
    (from the left / from the top) of the anchor point, so "br" at 96/96 hugs the bottom-right."""
    text = (wm.get("text") or "").strip()
    if not wm.get("enabled") or not text:
        return None
    an = WATERMARK_ANCHORS.get(wm.get("anchor"), 3)
    x = play_w * min(max(float(wm.get("position_x", 96)), 0), 100) / 100
    y = play_h * min(max(float(wm.get("position_y", 96)), 0), 100) / 100
    alpha = int(round(255 * (1 - min(max(float(wm.get("opacity", 70)), 0), 100) / 100)))
    tags = (rf"{{\an{an}\pos({x:.0f},{y:.0f})\q2\alpha&H{alpha:02X}&"
            rf"\4a&H{min(255, alpha + 120):02X}&}}")
    return f"Dialogue: 1,{_ass_time(0)},{_ass_time(duration + 1)},Watermark,,0,0,0,,{tags}{_ass_escape(text)}"


STATION_CHAR = 0.55       # Consolas-like: a character is ~0.55 em wide (+ the Station style's 3 px spacing)
ON_AIR_RED = "#FF4A3D"


def _text_width(text, size):
    """Rough width (px) of `text` in the Station style (monospaced), to place lines and boxes around it."""
    return len(text) * (size * STATION_CHAR + 3)


def _rect(x, y, w, h, color, alpha, layer, when):
    """A filled rectangle drawn with ASS vector drawing (\\p1), top-left at (x, y)."""
    return (rf"Dialogue: {layer},{when},Station,,0,0,0,,{{\an7\pos({x:.0f},{y:.0f})\p1\bord0\shad0"
            rf"\1c&H{_ass_color(color)[4:]}&\1a&H{alpha:02X}&}}m 0 0 l {w:.0f} 0 {w:.0f} {h:.0f} 0 {h:.0f}{{\p0}}")


def _side_lines(cx, cy, half_gap, length, color, alpha, when, thick=2):
    """Two thin horizontal lines left and right of a centered thing (its half width = half_gap)."""
    return [_rect(cx - half_gap - length, cy - thick / 2, length, thick, color, alpha, 2, when),
            _rect(cx + half_gap, cy - thick / 2, length, thick, color, alpha, 2, when)]


def station_lines(st, play_w, play_h, duration):
    """The station tag for the whole video, centered at X/Y % of the frame: a radio-display look, with a
    soft glow of its own color behind it (its own layer, so the letters stay sharp), thin lines on both
    sides, and an "ON AIR" light under it."""
    text = (st.get("text") or "").strip()
    if not st.get("enabled") or not text:
        return []
    x = play_w * min(max(float(st.get("position_x", 50)), 0), 100) / 100
    y = play_h * min(max(float(st.get("position_y", 5)), 0), 100) / 100
    size = int(st.get("size", 38))
    color = st.get("color", "#FFB347")
    when = f"{_ass_time(0)},{_ass_time(duration + 1)}"
    pos = rf"\an5\pos({x:.0f},{y:.0f})\q2"
    out = []
    if st.get("glow", True):
        glow = _ass_color(color)[4:]       # BBGGRR, for \3c
        out.append(rf"Dialogue: 2,{when},Station,,0,0,0,,{{{pos}\3c&H{glow}&\bord6\blur10\3a&H90&\1a&HFF&\shad0}}"
                   + _ass_escape(text))
    out.append(rf"Dialogue: 3,{when},Station,,0,0,0,,{{{pos}}}{_ass_escape(text)}")
    if st.get("lines", True):
        out += _side_lines(x, y, _text_width(text, size) / 2 + size * 0.7, size * 2.6, color, 0x50, when)
    if st.get("on_air", True):
        out += on_air_lines(x, y + size * 1.75, max(16, round(size * 0.85)), when)
    return out


def on_air_lines(x, y, size, when):
    """A red "ON AIR" light: the words in a rounded outlined box, radio waves on both sides, a soft glow."""
    red = _ass_color(ON_AIR_RED)[4:]
    label = "ON AIR"
    tw = _text_width(label, size)
    bw, bh, r = tw + size * 1.2, size * 1.55, size * 0.35
    left, top = x - bw / 2, y - bh / 2
    box = (f"m {r:.0f} 0 l {bw - r:.0f} 0 b {bw:.0f} 0 {bw:.0f} 0 {bw:.0f} {r:.0f} l {bw:.0f} {bh - r:.0f} "
           f"b {bw:.0f} {bh:.0f} {bw:.0f} {bh:.0f} {bw - r:.0f} {bh:.0f} l {r:.0f} {bh:.0f} "
           f"b 0 {bh:.0f} 0 {bh:.0f} 0 {bh - r:.0f} l 0 {r:.0f} b 0 0 0 0 {r:.0f} 0")
    frame = rf"\an7\pos({left:.0f},{top:.0f})\p1\shad0\1a&HFF&\3c&H{red}&"
    words = rf"\an5\pos({x:.0f},{y:.0f})\fs{size}\1c&H{red}&\3c&H{red}&\shad0"
    waves_l = rf"\an6\pos({left - size * 0.45:.0f},{y:.0f})\fs{size}\1c&H{red}&\3c&H{red}&\shad0"
    waves_r = rf"\an4\pos({left + bw + size * 0.45:.0f},{y:.0f})\fs{size}\1c&H{red}&\3c&H{red}&\shad0"
    return [
        rf"Dialogue: 2,{when},Station,,0,0,0,,{{{frame}\bord5\blur8\3a&HA0&}}{box}{{\p0}}",      # glow
        rf"Dialogue: 2,{when},Station,,0,0,0,,{{{words}\bord5\blur8\3a&HA0&\1a&HFF&}}{label}",
        rf"Dialogue: 3,{when},Station,,0,0,0,,{{{frame}\bord2}}{box}{{\p0}}",
        rf"Dialogue: 3,{when},Station,,0,0,0,,{{{words}\bord0}}{label}",
        rf"Dialogue: 3,{when},Station,,0,0,0,,{{{waves_l}\bord0}}((·",
        rf"Dialogue: 3,{when},Station,,0,0,0,,{{{waves_r}\bord0}}·))",
    ]


def flourish_lines(style, margins, play_w, play_h, duration):
    """A thin tapered swoosh under the captions, in the caption color, for the whole video."""
    margin_l, margin_r, margin_v = margins
    cx = (margin_l + play_w - margin_r) / 2
    k = int(style.get("size", 84)) / 80
    w, y = 300 * k, play_h - margin_v + 16 * k
    path = (f"m 0 {9 * k:.1f} b {w * 0.3:.1f} {-2 * k:.1f} {w * 0.7:.1f} {-2 * k:.1f} {w:.1f} {5 * k:.1f} "
            f"b {w * 0.7:.1f} {2 * k:.1f} {w * 0.3:.1f} {3.5 * k:.1f} 0 {9 * k:.1f}")
    color = _ass_color(style.get("color", "#FFFFFF"))[4:]
    when = f"{_ass_time(0)},{_ass_time(duration + 1)}"
    return [rf"Dialogue: 1,{when},Station,,0,0,0,,{{\an7\pos({cx - w / 2:.0f},{y:.0f})\p1\bord0\shad0"
            rf"\1c&H{color}&\1a&H30&\blur0.6}}{path}{{\p0}}"]


def sign_off_lines(so, style, margins, play_w, play_h, station=None):
    """The sign-off line ("Tune in tomorrow ...") just under the last caption (and its flourish), fading
    in once the voice is done. Drawn like the station tag (its font, color and glow), so the ending
    carries the brand."""
    if not so:
        return []
    margin_l, margin_r, margin_v = margins
    x = (margin_l + play_w - margin_r) / 2
    y = play_h - margin_v + int(style.get("size", 84)) * (0.75 if style.get("flourish", False) else 0.45)
    size = max(20, int((station or {}).get("size", 38)))
    when = f"{_ass_time(so['start'])},{_ass_time(so['end'] - 0.005)}"
    pos = rf"\an8\pos({x:.0f},{y:.0f})\fs{size}\q2\fad(700,0)"
    glow = _ass_color((station or {}).get("color", "#FFB347"))[4:]
    text = _ass_escape(so["text"])
    return [rf"Dialogue: 2,{when},Station,,0,0,0,,{{{pos}\3c&H{glow}&\bord6\blur10\3a&H90&\1a&HFF&\shad0}}" + text,
            rf"Dialogue: 3,{when},Station,,0,0,0,,{{{pos}}}" + text]


def visualizer_lines(vis, play_w, play_h, duration):
    """Thin lines on both sides of the voice waveform (the waveform itself is drawn by FFmpeg)."""
    if not vis:
        return []
    x = play_w * float(vis["position_x"]) / 100
    y = play_h * float(vis["position_y"]) / 100
    half = play_w * float(vis["width"]) / 200
    when = f"{_ass_time(0)},{_ass_time(duration + 1)}"
    return _side_lines(x, y, half + 14, play_w * 0.12, vis.get("color", "#FFE9C7"), 0x60, when)


def write_ass(path, events, style, width=1080, height=1920, watermark=None, duration=0.0, station=None,
              sign_off=None, visualizer=None):
    """Write subtitles in a 1080x1920 reference frame, so sizes are 1080p pixels at any output or
    preview resolution."""
    play_w, play_h = 1080, 1920
    margin_l, margin_r, margin_v = caption_margins(style, play_w, play_h)
    bold = -1 if style.get("bold", True) else 0
    shadow = float(style.get("shadow", 2))
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {play_w}",
        f"PlayResY: {play_h}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,{font},{size},{primary},{primary},{outline},{back},{bold},{italic},0,0,100,100,0,0,1,"
        "{ow},{shadow},2,{ml},{mr},{mv},1".format(
            font=style.get("font", "Arial Black").replace(",", " "),
            size=int(style.get("size", 84)),
            primary=_ass_color(style.get("color")),
            outline=_ass_color(style.get("outline_color", "#000000")),
            back=_ass_color("#000000", 0x60),
            bold=bold, italic=-1 if style.get("italic") else 0, ow=float(style.get("outline", 6)), shadow=0,      # the shadow is its own soft layer (below)
            ml=margin_l, mr=margin_r, mv=margin_v),
        "Style: Watermark,{font},{size},{color},{color},{outline},&H00000000,{bold},{italic},0,0,100,100,0,0,1,"
        "{ow},{sh},3,0,0,0,1".format(
            font=(watermark or {}).get("font", "Arial").replace(",", " "),
            size=int((watermark or {}).get("size", 40)),
            color=_ass_color((watermark or {}).get("color", "#FFFFFF")),
            outline=_ass_color((watermark or {}).get("outline_color", "#000000")),
            bold=-1 if (watermark or {}).get("bold", True) else 0, italic=-1 if (watermark or {}).get("italic") else 0,
            ow=float((watermark or {}).get("outline", 2)), sh=float((watermark or {}).get("shadow", 1))),
        "Style: Station,{font},{size},{color},{color},&H00081218,&H00000000,-1,0,0,0,100,100,3,0,1,"
        "1,0,5,0,0,0,1".format(
            font=(station or {}).get("font", "Consolas").replace(",", " "),
            size=int((station or {}).get("size", 38)),
            color=_ass_color((station or {}).get("color", "#FFB347"))),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    # A soft drop shadow, like the preview: drawn on its own layer under the text (invisible text,
    # blurred shadow straight below), so the letters and outline stay sharp. libass's plain shadow is
    # a hard-edged copy offset down-right, which looks like doubled letters.
    soft = (rf"{{\1a&HFF&\3a&HFF&\shad0\xshad0\yshad{shadow:g}\blur{shadow * 0.7:.1f}\4a&H50&}}"
            if shadow > 0 else "")
    for ev in events:
        text = _ass_escape(ev["text"].upper() if style.get("uppercase") else ev["text"])
        # End rounded DOWN to ASS's 1/100 s: rounding up would leave a caption on the next scene's first frame.
        when = f"{_ass_time(ev['start'])},{_ass_time(ev['end'] - 0.005)}"
        if soft:
            lines.append(f"Dialogue: 0,{when},Default,,0,0,0,,{soft}{text}")
        lines.append(f"Dialogue: 1,{when},Default,,0,0,0,,{text}")
    wm_line = watermark_line(watermark or {}, play_w, play_h, duration)
    if wm_line:
        lines.append(wm_line)
    lines += station_lines(station or {}, play_w, play_h, duration)
    if events and style.get("flourish", False):
        lines += flourish_lines(style, (margin_l, margin_r, margin_v), play_w, play_h, duration)
    lines += visualizer_lines(visualizer, play_w, play_h, duration)
    lines += sign_off_lines(sign_off, style, (margin_l, margin_r, margin_v), play_w, play_h, station)
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines) + "\n")
