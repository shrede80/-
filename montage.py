#!/usr/bin/env python3
"""
أداة المونتاج التلقائي
======================

تأخذ فيديو كاميرا واحدة أو كاميرتين وتقوم بـ:
  1. مزامنة الكاميرتين تلقائيًا عن طريق الصوت.
  2. قص السكتات (الأماكن التي لا يوجد فيها كلام).
  3. حذف كلمات الحشو العربية مثل: اممم، آه، إيه.
  4. التبديل بين الكاميرتين بشكل تلقائي.
  5. إخراج مقاطع مرقمة بالترتيب، وفيديو كامل جاهز، وملف ترجمة SRT
     تستطيع استيراده في CapCut.

الاستخدام:
    python montage.py كاميرا1.mp4 كاميرا2.mp4
    python montage.py كاميرا1.mp4                 (كاميرا واحدة فقط)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16000

# حتى تظهر الحروف العربية بشكل صحيح في شاشة الأوامر على ويندوز
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# كلمات الحشو الافتراضية (بعد إزالة التشكيل والتطويل وتوحيد الهمزات)
DEFAULT_FILLERS = {
    "اا", "ااا", "اه", "ااه", "اااه", "اهه", "اههه",
    "امم", "اممم", "امممم", "مم", "ممم", "مممم",
    "همم", "هممم", "اييه", "ايييه", "اوه",
    "uh", "um", "umm", "uhm", "hmm", "mm", "ah", "eh", "er",
}
# ملاحظة: لم نضع "أم" و"هم" و"إيه" لأنها كلمات حقيقية لها معنى.
# كلمات قد تكون حشوًا وقد تكون مقصودة، تُحذف فقط عند الطلب
OPTIONAL_FILLERS = {"يعني", "طيب", "يعنى", "خلاص", "اوكي"}


# ---------------------------------------------------------------------------
# أدوات مساعدة
# ---------------------------------------------------------------------------

def say(msg):
    print(msg, flush=True)


def fail(msg):
    say(f"\n❌ {msg}")
    sys.exit(1)


def run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        fail("حدث خطأ أثناء تشغيل ffmpeg:\n" + proc.stderr.decode(errors="ignore")[-1500:])
    return proc.stdout


def check_ffmpeg():
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            fail(f"برنامج {tool} غير مثبت. راجع ملف README لطريقة التثبيت.")


def probe(path):
    out = run(["ffprobe", "-v", "error", "-print_format", "json",
               "-show_streams", "-show_format", path])
    info = json.loads(out)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        fail(f"الملف لا يحتوي على فيديو: {path}")
    num, den = (video.get("avg_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    if not 1 < fps < 241:
        fps = 30.0
    width, height = int(video["width"]), int(video["height"])
    rotation = 0
    for sd in video.get("side_data_list", []):
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    rotation = int(video.get("tags", {}).get("rotate", rotation))
    if abs(rotation) % 180 == 90:
        width, height = height, width
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    return {
        "duration": float(info["format"]["duration"]),
        "fps": fps,
        "width": width,
        "height": height,
        "has_audio": has_audio,
    }


def load_audio(path, sr=SAMPLE_RATE):
    raw = run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1",
               "-ar", str(sr), "-f", "f32le", "-"])
    return np.frombuffer(raw, dtype=np.float32).copy()


def fmt_time(sec):
    sec = max(0.0, sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


# ---------------------------------------------------------------------------
# 1) مزامنة الكاميرتين
# ---------------------------------------------------------------------------

def find_delay(a1, a2, sr=SAMPLE_RATE, max_minutes=8.0):
    """
    يرجع كم ثانية بدأت الكاميرا الثانية بعد الأولى.
    (موجب = الثانية بدأت التصوير متأخرة)
    """
    from scipy.signal import fftconvolve

    work_sr = 4000
    step = sr // work_sr
    n = int(max_minutes * 60 * sr)
    x = a1[:n:step].astype(np.float64)
    y = a2[:n:step].astype(np.float64)
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    corr = fftconvolve(x, y[::-1], mode="full")
    lag = int(np.argmax(corr)) - (len(y) - 1)
    peak = corr.max() / (np.abs(corr).mean() + 1e-9)
    return lag / work_sr, peak


# ---------------------------------------------------------------------------
# 2) كشف الكلام والسكتات
# ---------------------------------------------------------------------------

def detect_speech(audio, sr=SAMPLE_RATE, threshold_db=None, min_silence=0.6,
                  min_speech=0.15, pad=0.15):
    frame = int(sr * 0.02)
    n = len(audio) // frame
    if n == 0:
        return [], -60.0
    frames = audio[: n * frame].reshape(n, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms)

    if threshold_db is None:
        noise = np.percentile(db, 10)
        loud = np.percentile(db, 95)
        threshold_db = noise + (loud - noise) * 0.35

    voiced = db > threshold_db
    hop = frame / sr

    # تحويل الإطارات إلى فترات زمنية
    regions = []
    start = None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            regions.append([start * hop, i * hop])
            start = None
    if start is not None:
        regions.append([start * hop, n * hop])

    # دمج الفترات التي بينها سكتة قصيرة
    merged = []
    for r in regions:
        if merged and r[0] - merged[-1][1] < min_silence:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    total = n * hop
    out = []
    for s, e in merged:
        if e - s < min_speech:
            continue
        s, e = max(0.0, s - pad), min(total, e + pad)
        if out and s <= out[-1][1]:
            out[-1][1] = e
        else:
            out.append([s, e])
    return out, threshold_db


# ---------------------------------------------------------------------------
# 3) التفريغ النصي وكشف الحشو
# ---------------------------------------------------------------------------

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_word(w):
    w = _DIACRITICS.sub("", w.strip().lower())
    w = _PUNCT.sub("", w)
    w = re.sub("[إأآٱ]", "ا", w)
    w = w.replace("ى", "ي").replace("ة", "ه")
    return w


def is_filler(word, fillers):
    w = normalize_word(word)
    if not w:
        return False
    if w in fillers:
        return True
    # أصوات ممطوطة مثل: اممممم، ااااه، مممم، هممم
    return bool(re.fullmatch(r"ا+م{2,}|ا{2,}ه*|ا+ه{2,}|م{2,}|ه+م{2,}", w))


def transcribe(audio_path, model_size, language="ar"):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        fail("مكتبة faster-whisper غير مثبتة. شغّل: pip install -r requirements.txt")
    say(f"   تحميل نموذج التفريغ ({model_size})... أول مرة قد تأخذ عدة دقائق.")
    model = WhisperModel(model_size, device="auto", compute_type="int8")
    segments, _ = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        vad_filter=False,
        # نطلب من النموذج كتابة كلمات الحشو بدل تجاهلها
        initial_prompt="اممم، يعني، آه، إيه... طيب، اممم.",
    )
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({"text": w.word.strip(), "start": float(w.start), "end": float(w.end)})
        say(f"   ... {fmt_time(seg.end)}")
    return words


def subtract_ranges(keep, remove, min_len=0.2):
    """يحذف الفترات remove من الفترات keep."""
    result = []
    remove = sorted(remove)
    for s, e in keep:
        pieces = [[s, e]]
        for rs, re_ in remove:
            nxt = []
            for ps, pe in pieces:
                if re_ <= ps or rs >= pe:
                    nxt.append([ps, pe])
                    continue
                if rs > ps:
                    nxt.append([ps, rs])
                if re_ < pe:
                    nxt.append([re_, pe])
            pieces = nxt
        result.extend(p for p in pieces if p[1] - p[0] >= min_len)
    return result


# ---------------------------------------------------------------------------
# 4) اختيار الكاميرا لكل مقطع
# ---------------------------------------------------------------------------

@dataclass
class Clip:
    start: float          # بداية المقطع في توقيت الكاميرا الأولى
    end: float
    cam: int              # 1 أو 2
    new_start: float = 0  # بدايته في الفيديو النهائي
    reason: str = field(default="")

    @property
    def dur(self):
        return self.end - self.start


def assign_cameras(segments, cams, delay, switch_every, min_shot=2.0):
    """
    يوزع المقاطع على الكاميرات: يبقى على كاميرا حتى تمر مدة switch_every
    ثم ينتقل للأخرى عند أقرب قصة. لا يختار كاميرا ليس فيها تصوير في ذلك الوقت.
    """
    clips = []
    current, on_cam = 1, 0.0
    cam2_dur = cams[1]["duration"] if len(cams) > 1 else 0

    def cam2_ok(s, e):
        return len(cams) > 1 and s - delay >= 0 and e - delay <= cam2_dur

    # المقاطع الطويلة جدًا نقسمها حتى يحصل تبديل حتى في منتصف الكلام
    if len(cams) > 1:
        pieces = []
        for s, e in segments:
            while e - s > switch_every * 1.6:
                pieces.append([s, s + switch_every])
                s += switch_every
            pieces.append([s, e])
        segments = pieces

    for s, e in segments:
        if len(cams) > 1 and on_cam >= switch_every and (e - s) >= min_shot * 0.5:
            other = 2 if current == 1 else 1
            if other == 1 or cam2_ok(s, e):
                current, on_cam = other, 0.0
        cam = current
        if cam == 2 and not cam2_ok(s, e):
            cam = 1
        clips.append(Clip(s, e, cam))
        on_cam += e - s

    t = 0.0
    for c in clips:
        c.new_start = t
        t += c.dur
    return clips


# ---------------------------------------------------------------------------
# 5) الإخراج
# ---------------------------------------------------------------------------

def render_clip(clip, idx, cams, paths, delay, audio_path, target, out_path, crf, audio_offset):
    src = paths[clip.cam - 1]
    vstart = clip.start - (delay if clip.cam == 2 else 0.0)
    astart = clip.start - audio_offset
    w, h, fps = target["width"], target["height"], target["fps"]
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps:.3f}")
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-ss", f"{max(0.0, vstart):.3f}", "-t", f"{clip.dur:.3f}", "-i", src,
        "-ss", f"{max(0.0, astart):.3f}", "-t", f"{clip.dur:.3f}", "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-shortest", "-movflags", "+faststart",
        out_path,
    ]
    run(cmd)


def build_srt(words, clips, max_words=7, max_gap=0.6):
    """يبني ترجمة على توقيت الفيديو النهائي (بعد القص)."""
    def map_time(t):
        for c in clips:
            if c.start <= t <= c.end:
                return c.new_start + (t - c.start)
        return None

    kept = []
    for w in words:
        if w.get("filler"):
            continue
        mid = (w["start"] + w["end"]) / 2
        nm = map_time(mid)
        if nm is None:
            continue
        ns = map_time(w["start"])
        ne = map_time(w["end"])
        ns = nm - 0.05 if ns is None else ns
        ne = nm + 0.05 if ne is None else ne
        kept.append((ns, max(ne, ns + 0.05), w["text"]))

    lines, cur = [], []
    for item in kept:
        if cur and (len(cur) >= max_words or item[0] - cur[-1][1] > max_gap):
            lines.append(cur)
            cur = []
        cur.append(item)
    if cur:
        lines.append(cur)

    def ts(t):
        ms = int(round(t * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    out = []
    for i, line in enumerate(lines, 1):
        text = " ".join(t for _, _, t in line).strip()
        out.append(f"{i}\n{ts(line[0][0])} --> {ts(line[-1][1])}\n{text}\n")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# البرنامج الرئيسي
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="مونتاج تلقائي: قص السكتات، حذف الحشو، التبديل بين كاميرتين.")
    p.add_argument("videos", nargs="+", help="فيديو الكاميرا الأولى، ثم الثانية (اختياري)")
    p.add_argument("-o", "--out", help="مجلد الإخراج (افتراضيًا بجانب الفيديو)")
    p.add_argument("--audio", help="ملف صوت منفصل (مايك خارجي) يُستخدم بدل صوت الكاميرا")
    p.add_argument("--audio-from", type=int, choices=[1, 2], default=1,
                   help="صوت أي كاميرا يُستخدم في الفيديو النهائي (افتراضي 1)")
    p.add_argument("--switch-every", type=float, default=7.0,
                   help="بعد كم ثانية تقريبًا يتم التبديل بين الكاميرات (افتراضي 7)")
    p.add_argument("--min-silence", type=float, default=0.6,
                   help="أقل مدة سكتة (بالثواني) يتم قصها (افتراضي 0.6)")
    p.add_argument("--pad", type=float, default=0.15,
                   help="هامش يُترك قبل وبعد الكلام حتى لا يكون القص حادًا (افتراضي 0.15)")
    p.add_argument("--silence-db", type=float, default=None,
                   help="مستوى الصوت الذي يعتبر سكوتًا (مثلاً -40). افتراضيًا يحسب تلقائيًا")
    p.add_argument("--no-transcribe", action="store_true",
                   help="لا تفرغ الكلام (أسرع، لكن بدون حذف حشو وبدون ترجمة)")
    p.add_argument("--remove-yaani", action="store_true",
                   help="احذف أيضًا: يعني، طيب، خلاص")
    p.add_argument("--extra-fillers", default="",
                   help="كلمات إضافية للحذف مفصولة بفواصل")
    p.add_argument("--model", default="small",
                   help="حجم نموذج التفريغ: tiny / base / small / medium / large-v3 (افتراضي small)")
    p.add_argument("--crf", type=int, default=18, help="جودة الفيديو (أقل = أفضل، افتراضي 18)")
    p.add_argument("--no-render", action="store_true",
                   help="لا تصدّر الفيديو، فقط التقرير وملف الترجمة")
    args = p.parse_args(argv)

    check_ffmpeg()
    paths = [os.path.abspath(v) for v in args.videos]
    if len(paths) > 2:
        fail("الأداة تدعم كاميرتين كحد أقصى.")
    for v in paths:
        if not os.path.isfile(v):
            fail(f"الملف غير موجود: {v}")

    base = os.path.splitext(os.path.basename(paths[0]))[0]
    out_dir = os.path.abspath(args.out or os.path.join(os.path.dirname(paths[0]), f"{base}_مونتاج"))
    clips_dir = os.path.join(out_dir, "مقاطع")
    os.makedirs(clips_dir, exist_ok=True)

    say("🎬 أداة المونتاج التلقائي\n")
    cams = [probe(v) for v in paths]
    for i, (v, c) in enumerate(zip(paths, cams), 1):
        say(f"الكاميرا {i}: {os.path.basename(v)}  ({fmt_time(c['duration'])}, "
            f"{c['width']}x{c['height']}, {c['fps']:.0f}fps)")

    # --- الصوت ---
    say("\n1) قراءة الصوت...")
    cam_audio = []
    for v, c in zip(paths, cams):
        if not c["has_audio"]:
            fail(f"الملف لا يحتوي على صوت، ولا يمكن مزامنته: {v}")
        cam_audio.append(load_audio(v))

    delay = 0.0
    if len(paths) == 2:
        say("2) مزامنة الكاميرتين عن طريق الصوت...")
        delay, conf = find_delay(cam_audio[0], cam_audio[1])
        who = "الثانية" if delay >= 0 else "الأولى"
        say(f"   الكاميرا {who} بدأت التصوير بعد الأخرى بـ {abs(delay):.2f} ثانية "
            f"(ثقة المزامنة: {'عالية' if conf > 8 else 'منخفضة، راجع النتيجة'})")
    else:
        say("2) كاميرا واحدة، لا حاجة للمزامنة.")

    # صوت الفيديو النهائي، وكل التوقيتات بتوقيت الكاميرا الأولى
    if args.audio:
        audio_path = os.path.abspath(args.audio)
        ext = load_audio(audio_path)
        audio_offset, _ = find_delay(cam_audio[0], ext)
        main_audio = ext
    elif args.audio_from == 2 and len(paths) == 2:
        audio_path, audio_offset, main_audio = paths[1], delay, cam_audio[1]
    else:
        audio_path, audio_offset, main_audio = paths[0], 0.0, cam_audio[0]

    # نحول الصوت الرئيسي إلى توقيت الكاميرا الأولى
    shift = int(round(audio_offset * SAMPLE_RATE))
    if shift > 0:
        main_audio = np.concatenate([np.zeros(shift, np.float32), main_audio])
    elif shift < 0:
        main_audio = main_audio[-shift:]
    main_audio = main_audio[: len(cam_audio[0])]

    say("3) البحث عن السكتات...")
    speech, thr = detect_speech(main_audio, threshold_db=args.silence_db,
                                min_silence=args.min_silence, pad=args.pad)
    if not speech:
        fail("لم يتم العثور على كلام. جرّب خيار --silence-db -50")
    say(f"   وُجد {len(speech)} جزء فيه كلام (حد السكوت {thr:.0f} dB)")

    # --- التفريغ والحشو ---
    words, filler_ranges = [], []
    if not args.no_transcribe:
        say("4) تفريغ الكلام وكشف كلمات الحشو...")
        fillers = set(DEFAULT_FILLERS)
        if args.remove_yaani:
            fillers |= OPTIONAL_FILLERS
        fillers |= {normalize_word(x) for x in args.extra_fillers.split(",") if x.strip()}

        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "audio.wav")
            from scipy.io import wavfile
            wavfile.write(wav, SAMPLE_RATE, main_audio)
            words = transcribe(wav, args.model)
        for w in words:
            if is_filler(w["text"], fillers):
                w["filler"] = True
                filler_ranges.append([w["start"] - 0.03, w["end"] + 0.03])
        say(f"   وُجدت {len(filler_ranges)} كلمة حشو سيتم حذفها")
    else:
        say("4) تم تخطي التفريغ.")

    keep = subtract_ranges(speech, filler_ranges)
    clips = assign_cameras(keep, cams, delay, args.switch_every)

    original = cams[0]["duration"]
    final = sum(c.dur for c in clips)
    say(f"\n   المدة الأصلية: {fmt_time(original)}  ←  بعد القص: {fmt_time(final)}  "
        f"(تم حذف {fmt_time(original - final)})")

    # --- الملفات ---
    if words:
        with open(os.path.join(out_dir, "ترجمة.srt"), "w", encoding="utf-8") as f:
            f.write(build_srt(words, clips))
        with open(os.path.join(out_dir, "النص_الكامل.txt"), "w", encoding="utf-8") as f:
            f.write(" ".join(w["text"] for w in words))

    report = [
        "تقرير المونتاج",
        "=" * 40,
        f"المدة الأصلية: {fmt_time(original)}",
        f"المدة بعد القص: {fmt_time(final)}",
        f"عدد المقاطع: {len(clips)}",
        f"فرق التوقيت بين الكاميرتين: {delay:.2f} ثانية" if len(paths) == 2 else "كاميرا واحدة",
        f"كلمات حشو محذوفة: {len(filler_ranges)}",
        "",
        "رقم   الكاميرا   من (الأصل)     إلى (الأصل)    في الفيديو النهائي",
    ]
    for i, c in enumerate(clips, 1):
        report.append(f"{i:03d}   {c.cam}          {fmt_time(c.start)}   {fmt_time(c.end)}   {fmt_time(c.new_start)}")
    with open(os.path.join(out_dir, "تقرير.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    with open(os.path.join(out_dir, "cuts.json"), "w", encoding="utf-8") as f:
        json.dump({"delay": delay, "videos": paths,
                   "clips": [{"start": c.start, "end": c.end, "cam": c.cam,
                              "new_start": c.new_start} for c in clips]},
                  f, ensure_ascii=False, indent=2)

    if args.no_render:
        say(f"\n✅ انتهى. الملفات في: {out_dir}")
        return

    say(f"\n5) تصدير {len(clips)} مقطع...")
    target = cams[0]
    clip_files = []
    for i, c in enumerate(clips, 1):
        name = os.path.join(clips_dir, f"{i:03d}_كاميرا{c.cam}.mp4")
        render_clip(c, i, cams, paths, delay, audio_path, target, name, args.crf, audio_offset)
        clip_files.append(name)
        if i % 10 == 0 or i == len(clips):
            say(f"   {i}/{len(clips)}")

    say("6) تجميع الفيديو النهائي...")
    list_file = os.path.join(out_dir, ".list.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for name in clip_files:
            f.write("file '" + name.replace("'", "'\\''") + "'\n")
    final_path = os.path.join(out_dir, "الفيديو_النهائي.mp4")
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
         "-c", "copy", "-movflags", "+faststart", final_path])
    os.remove(list_file)

    say(f"\n✅ انتهى! الملفات في:\n   {out_dir}\n")
    say("   • الفيديو_النهائي.mp4  ← فيديو كامل مقصوص وجاهز")
    say("   • مقاطع/              ← كل مقطع لوحده بالترتيب، اسحبها إلى CapCut")
    if words:
        say("   • ترجمة.srt           ← ترجمة عربية تستوردها في CapCut")
    say("   • تقرير.txt           ← تفاصيل كل قصة")


if __name__ == "__main__":
    main()
