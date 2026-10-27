# -*- coding: utf-8 -*-
"""
Class attendance by face recognition  -  Kivy version for Pydroid 3

Install once (Pydroid -> Pip):  kivy  opencv-python  numpy
Persian text needs NO extra package: the Vazirmatn font and the Arabic
shaping / right-to-left code are built into this file.

Flow: choose/create a course -> label students (names next to faces)
      -> take attendance (date + photos) -> view the records table.
"""
# --- Android APK: ask for storage/photo permissions at startup (reopen the app once after granting)
try:
    from kivy.utils import platform as _plat
    if _plat == "android":
        from android.permissions import request_permissions, Permission
        request_permissions([Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE,
                             "android.permission.READ_MEDIA_IMAGES"])
except Exception:
    pass

import os, sys, json, time, csv, zlib, base64, threading, datetime, textwrap
import urllib.request
import numpy as np
import cv2

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.core.text import LabelBase
from kivy.graphics import Color, RoundedRectangle
from kivy.graphics.texture import Texture
from kivy.metrics import dp, sp
from kivy.properties import BooleanProperty, NumericProperty
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.modalview import ModalView
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.utils import platform

# ================================================================ settings
BASE = "/storage/emulated/0/Attendance"
try:
    os.makedirs(BASE, exist_ok=True)
    open(os.path.join(BASE, ".t"), "w").close()
except Exception:
    BASE = os.path.join(os.getcwd(), "Attendance")
    os.makedirs(BASE, exist_ok=True)

CLASSES_DIR = os.path.join(BASE, "classes")
os.makedirs(CLASSES_DIR, exist_ok=True)
FONT_DIR = os.path.join(BASE, "fonts")
CFG_FILE = os.path.join(BASE, "config.json")
DET_FILE = os.path.join(BASE, "yunet.onnx")
REC_FILE = os.path.join(BASE, "sface.onnx")
DET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
           "face_detection_yunet/face_detection_yunet_2023mar.onnx")
REC_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
           "face_recognition_sface/face_recognition_sface_2021dec.onnx")
CAMERA_DIRS = ["/storage/emulated/0/DCIM/Camera", "/sdcard/DCIM/Camera",
               "/storage/emulated/0/DCIM", "/storage/emulated/0/Pictures"]
IMG_EXT = (".jpg", ".jpeg", ".png")
MAXSIDE = 2000          # photos are shrunk to this size before detection
THRESH = 0.363          # cosine threshold for SFace (higher = stricter)
MAX_EMB = 20            # max stored face samples per person
GALLERY_MAX = 45

CFG = {"theme": "dark", "lang": "fa", "cal": "jalali", "folder": ""}
FONT = "Roboto"
PEOPLE_FILE = SHEET_FILE = ""
DET = REC = None


def load_cfg():
    try:
        with open(CFG_FILE, encoding="utf-8") as f:
            CFG.update(json.load(f))
    except Exception:
        pass


def save_cfg():
    try:
        with open(CFG_FILE, "w", encoding="utf-8") as f:
            json.dump(CFG, f, ensure_ascii=False)
    except Exception:
        pass


# ================================================================ theme
THEMES = {
    "dark": dict(bg="#0e1217", bar="#151b23", card="#1b232d", card2="#26313e",
                 field="#10161d", line="#34445a", text="#e8eef5", muted="#8fa1b3",
                 primary="#4c9aff", on_primary="#06121f", ok="#34d399",
                 on_ok="#04130d", okbg="#15402f", warn="#fbbf24",
                 danger="#f87171", on_danger="#2a0a0a", shade="#000000"),
    "light": dict(bg="#f1f4f8", bar="#ffffff", card="#ffffff", card2="#e6ecf3",
                  field="#f1f4f8", line="#c5d0dc", text="#16202b", muted="#617285",
                  primary="#1f6feb", on_primary="#ffffff", ok="#059669",
                  on_ok="#ffffff", okbg="#c9f1de", warn="#b45309",
                  danger="#dc2626", on_danger="#ffffff", shade="#000000"),
}


def hexc(h, a=1.0):
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255., int(h[2:4], 16) / 255., int(h[4:6], 16) / 255., a)


def col(k, a=None):
    """Theme colour key (or an rgba tuple) -> rgba."""
    if isinstance(k, (tuple, list)):
        return tuple(k)
    c = hexc(THEMES[CFG["theme"]][k])
    return c if a is None else (c[0], c[1], c[2], a)


def mix(c1, c2, t):
    return tuple(c1[i] * (1 - t) + c2[i] * t for i in range(4))


# ================================================================ Persian text
# Kivy (SDL2_ttf) draws glyphs left-to-right with no shaping, so Arabic-script
# text is converted here: letters -> joined presentation forms, then reordered
# into visual (left-to-right) order.  Pure Python, no extra packages.
_D, _R = "D", "R"      # dual-joining / right-joining letters
_LET = [
    ("ء", 0xFE80, None), ("آ", 0xFE81, _R), ("أ", 0xFE83, _R),
    ("ؤ", 0xFE85, _R), ("إ", 0xFE87, _R), ("ئ", 0xFE89, _D),
    ("ا", 0xFE8D, _R), ("ب", 0xFE8F, _D), ("ة", 0xFE93, _R),
    ("ت", 0xFE95, _D), ("ث", 0xFE99, _D), ("ج", 0xFE9D, _D),
    ("ح", 0xFEA1, _D), ("خ", 0xFEA5, _D), ("د", 0xFEA9, _R),
    ("ذ", 0xFEAB, _R), ("ر", 0xFEAD, _R), ("ز", 0xFEAF, _R),
    ("س", 0xFEB1, _D), ("ش", 0xFEB5, _D), ("ص", 0xFEB9, _D),
    ("ض", 0xFEBD, _D), ("ط", 0xFEC1, _D), ("ظ", 0xFEC5, _D),
    ("ع", 0xFEC9, _D), ("غ", 0xFECD, _D), ("ف", 0xFED1, _D),
    ("ق", 0xFED5, _D), ("ك", 0xFED9, _D), ("ل", 0xFEDD, _D),
    ("م", 0xFEE1, _D), ("ن", 0xFEE5, _D), ("ه", 0xFEE9, _D),
    ("و", 0xFEED, _R), ("ى", 0xFEEF, _R), ("ي", 0xFEF1, _D),
    ("پ", 0xFB56, _D), ("چ", 0xFB7A, _D), ("ژ", 0xFB8A, _R),
    ("ک", 0xFB8E, _D), ("گ", 0xFB92, _D), ("ی", 0xFBFC, _D),
    ("ۀ", 0xFBA4, _R),
]
_FORM = {}      # char -> (isolated, final, initial, medial)
_KIND = {}
for _c, _cp, _k in _LET:
    _KIND[_c] = _k
    if _k == _D:
        _FORM[_c] = tuple(chr(_cp + i) for i in range(4))
    elif _k == _R:
        _FORM[_c] = (chr(_cp), chr(_cp + 1), chr(_cp), chr(_cp + 1))
    else:
        _FORM[_c] = (chr(_cp),) * 4
_LAMALEF = {"آ": (0xFEF5, 0xFEF6), "أ": (0xFEF7, 0xFEF8),
            "إ": (0xFEF9, 0xFEFA), "ا": (0xFEFB, 0xFEFC)}
_MARKS = set(chr(c) for c in range(0x064B, 0x0660)) | {"ٰ"}
_ZW = {"‌", "‍", "‎", "‏"}
_MIRROR = {"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{",
           "<": ">", ">": "<", "«": "»", "»": "«"}


def _joins_next(c):
    return _KIND.get(c) == _D


def _joins_prev(c):
    return _KIND.get(c) in (_D, _R)


def reshape(s):
    out, n, i = [], len(s), 0
    while i < n:
        ch = s[i]
        if ch in _ZW:
            i += 1
            continue
        if ch not in _FORM:
            out.append(ch)
            i += 1
            continue
        j = i - 1
        while j >= 0 and s[j] in _MARKS:
            j -= 1
        prev = s[j] if j >= 0 else ""
        if prev and not _joins_next(prev):
            prev = ""
        k = i + 1
        while k < n and s[k] in _MARKS:
            k += 1
        nxt = s[k] if k < n else ""
        if ch == "ل" and nxt in _LAMALEF:
            iso, fin = _LAMALEF[nxt]
            out.append(chr(fin if prev else iso))
            out.extend(s[i + 1:k])
            i = k + 1
            continue
        pj = bool(prev)
        nj = _joins_next(ch) and _joins_prev(nxt)
        f = _FORM[ch]
        if pj and nj:
            out.append(f[3])
        elif pj:
            out.append(f[1])
        elif nj:
            out.append(f[2])
        else:
            out.append(f[0])
        i += 1
    return "".join(out)


def _cls(c):
    o = ord(c)
    if 0x0660 <= o <= 0x0669 or 0x06F0 <= o <= 0x06F9:
        return "L"                        # digits keep their order
    if o in (0x060C, 0x061B, 0x061F, 0x066A, 0x066B, 0x066C, 0x066D):
        return "N"
    if (0x0590 <= o <= 0x08FF) or (0xFB1D <= o <= 0xFDFF) or (0xFE70 <= o <= 0xFEFF):
        return "R"
    if c.isalnum():
        return "L"
    return "N"


def has_rtl(t):
    return any(_cls(c) == "R" for c in t)


def bidi(s):
    cls = [_cls(c) for c in s]
    n = len(s)
    res = cls[:]
    i = 0
    while i < n:
        if cls[i] == "N":
            j = i
            while j < n and cls[j] == "N":
                j += 1
            left = cls[i - 1] if i > 0 else "R"
            right = cls[j] if j < n else "R"
            d = left if left == right else "R"
            for k in range(i, j):
                res[k] = d
            i = j
        else:
            i += 1
    runs, start = [], 0
    for i in range(1, n + 1):
        if i == n or res[i] != res[start]:
            runs.append((res[start], s[start:i]))
            start = i
    out = []
    for d, t in reversed(runs):
        out.append("".join(_MIRROR.get(c, c) for c in reversed(t)) if d == "R" else t)
    return "".join(out)


def visual(line):
    if not has_rtl(line):
        return line
    return bidi(reshape(line))


def fa(text, wrap=0):
    """Logical text -> text ready for a Kivy Label (handles wrapping/newlines)."""
    lines = []
    for ln in str(text).split("\n"):
        if wrap and len(ln) > wrap:
            lines += textwrap.wrap(ln, wrap) or [""]
        else:
            lines.append(ln)
    return "\n".join(visual(l) for l in lines)


# ================================================================ Jalali date
def g2j(gy, gm, gd):
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    if gy > 1600:
        jy, gy = 979, gy - 1600
    else:
        jy, gy = 0, gy - 621
    gy2 = gy + 1 if gm > 2 else gy
    days = (365 * gy + (gy2 + 3) // 4 - (gy2 + 99) // 100 + (gy2 + 399) // 400
            - 80 + gd + g_d_m[gm - 1])
    jy += 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if days < 186:
        jm, jd = 1 + days // 31, 1 + days % 31
    else:
        jm, jd = 7 + (days - 186) // 30, 1 + (days - 186) % 30
    return jy, jm, jd


def today_str():
    d = datetime.date.today()
    if CFG["cal"] == "jalali":
        return "%04d/%02d/%02d" % g2j(d.year, d.month, d.day)
    return d.strftime("%Y-%m-%d")


# ================================================================ strings
STR = {
    "fa": {
        "app_title": "حضور و غیاب کلاس", "empty_courses":
        "هنوز درسی نساخته‌اید.\nبا دکمهٔ پایین یک درس جدید بسازید.",
        "new_course": "+ درس جدید", "new_course_t": "درس جدید",
        "course_name": "نام درس را بنویسید:", "name_hint": "نام…",
        "n_students": "{n} دانشجو", "n_sessions": "{n} جلسه",
        "delete": "حذف", "cancel": "انصراف", "ok": "تأیید", "save": "ذخیره",
        "close": "بستن", "back": "بازگشت", "settings": "تنظیمات",
        "act_label": "برچسب‌زنی دانشجویان", "act_label_s": "نام‌گذاری چهره‌ها در عکس‌ها",
        "act_att": "حضور و غیاب", "act_att_s": "شناسایی حاضرین از روی عکس",
        "act_rec": "سوابق حضور", "act_rec_s": "جدول جلسه‌های قبلی",
        "session_date": "تاریخ جلسه", "session_date_p": "تاریخ این جلسه را وارد کنید:",
        "session_name": "نام جلسه",
        "exists_p": "برای این تاریخ قبلاً جلسه ثبت شده:\n{names}\nنام جلسهٔ جدید را بنویسید:",
        "overwrite_q": "جلسهٔ «{s}» وجود دارد. جایگزین شود؟",
        "photos": "عکس‌ها",
        "photos_hint": "با دوربین عکس بگیرید (حدود ۱۰ نفر در هر عکس) و بعد به برنامه برگردید.",
        "open_cam": "باز کردن دوربین",
        "cam_fail": "دوربین خودکار باز نشد؛ برنامهٔ دوربین را خودتان باز کنید.",
        "use_new": "استفاده از {n} عکس جدید", "use_new0": "هنوز عکس جدیدی گرفته نشده",
        "gallery": "انتخاب از گالری", "gallery_use": "استفاده از {n} عکس انتخاب‌شده",
        "no_images": "عکسی پیدا نشد.", "folder_bad": "پوشهٔ عکس‌ها پیدا نشد.",
        "loading": "در حال بارگذاری…",
        "loading_models": "آماده‌سازی مدل‌ها…\n(بار اول حدود ۴۰ مگابایت دانلود می‌شود)",
        "model_err": "خطا در دریافت مدل‌ها (اینترنت را بررسی کنید):\n{e}",
        "processing": "پردازش عکس {i} از {n}",
        "no_faces": "در عکس‌ها چهره‌ای پیدا نشد.", "faces_n": "{n} چهره پیدا شد",
        "att_review": "جلسهٔ {d}", "write_names": "نام هر چهره را بنویسید",
        "unknown": "ناشناس", "match": "شباهت {s}٪", "pick": "لیست",
        "pick_title": "انتخاب نام", "saved_n": "ذخیره شد ({n} نفر)",
        "result_title": "نتیجهٔ حضور و غیاب", "present_n": "حاضر: {n}",
        "absent_n": "غایب: {n}", "absent_names": "غایبین:", "none": "—",
        "rec_title": "سوابق حضور", "no_sessions": "هنوز جلسه‌ای ثبت نشده.",
        "del_session": "حذف جلسه", "del_session_t": "کدام جلسه حذف شود؟",
        "del_session_q": "جلسهٔ «{s}» حذف شود؟",
        "del_course_q": "درس «{c}» با همهٔ چهره‌ها و سوابقش حذف شود؟",
        "name_col": "نام", "present_row": "حاضر", "total_col": "جمع",
        "theme": "ظاهر", "dark": "تیره", "light": "روشن", "language": "زبان",
        "calendar": "تقویم", "jalali": "شمسی", "gregorian": "میلادی",
        "folder": "پوشهٔ عکس‌های دوربین",
    },
    "en": {
        "app_title": "Class Attendance", "empty_courses":
        "No courses yet.\nCreate one with the button below.",
        "new_course": "+ New course", "new_course_t": "New course",
        "course_name": "Course name:", "name_hint": "Name...",
        "n_students": "{n} students", "n_sessions": "{n} sessions",
        "delete": "Delete", "cancel": "Cancel", "ok": "OK", "save": "Save",
        "close": "Close", "back": "Back", "settings": "Settings",
        "act_label": "Label students", "act_label_s": "Name the faces in photos",
        "act_att": "Take attendance", "act_att_s": "Recognize who is present",
        "act_rec": "Attendance records", "act_rec_s": "Table of previous sessions",
        "session_date": "Session date", "session_date_p": "Enter the date of this session:",
        "session_name": "Session name",
        "exists_p": "Already recorded for this date:\n{names}\nName of the new session:",
        "overwrite_q": "Session '{s}' exists. Replace it?",
        "photos": "Photos",
        "photos_hint": "Take photos with the camera (about 10 students per photo), then come back.",
        "open_cam": "Open camera",
        "cam_fail": "Could not open the camera automatically; open it manually.",
        "use_new": "Use {n} new photos", "use_new0": "No new photos yet",
        "gallery": "Choose from gallery", "gallery_use": "Use {n} selected photos",
        "no_images": "No photos found.", "folder_bad": "Photos folder not found.",
        "loading": "Loading...",
        "loading_models": "Preparing models...\n(first run downloads ~40 MB)",
        "model_err": "Could not download the models (check internet):\n{e}",
        "processing": "Processing photo {i} of {n}",
        "no_faces": "No faces were found.", "faces_n": "{n} faces found",
        "att_review": "Session {d}", "write_names": "Write a name for each face",
        "unknown": "unknown", "match": "match {s}%", "pick": "List",
        "pick_title": "Choose a name", "saved_n": "Saved ({n} people)",
        "result_title": "Attendance result", "present_n": "Present: {n}",
        "absent_n": "Absent: {n}", "absent_names": "Absent:", "none": "-",
        "rec_title": "Attendance records", "no_sessions": "No sessions recorded yet.",
        "del_session": "Delete session", "del_session_t": "Which session to delete?",
        "del_session_q": "Delete session '{s}'?",
        "del_course_q": "Delete course '{c}' with all its faces and records?",
        "name_col": "Name", "present_row": "Present", "total_col": "Total",
        "theme": "Theme", "dark": "Dark", "light": "Light", "language": "Language",
        "calendar": "Calendar", "jalali": "Jalali", "gregorian": "Gregorian",
        "folder": "Camera photos folder",
    },
}


def tr(key, **kw):
    s = STR[CFG["lang"]].get(key) or STR["en"].get(key, key)
    return s.format(**kw) if kw else s


def RTL():
    return CFG["lang"] == "fa"


# ================================================================ data layer
def cdir(name):
    return os.path.join(CLASSES_DIR, name)


def list_courses():
    return sorted(d for d in os.listdir(CLASSES_DIR)
                  if os.path.isdir(cdir(d)))


def set_course(name):
    global PEOPLE_FILE, SHEET_FILE
    os.makedirs(cdir(name), exist_ok=True)
    PEOPLE_FILE = os.path.join(cdir(name), "people.json")
    SHEET_FILE = os.path.join(cdir(name), "attendance.csv")


def migrate_old():
    old_p, old_s = os.path.join(BASE, "people.json"), os.path.join(BASE, "attendance.csv")
    if os.path.exists(old_p) or os.path.exists(old_s):
        d = cdir("Default")
        os.makedirs(d, exist_ok=True)
        for f in (old_p, old_s):
            if os.path.exists(f):
                os.replace(f, os.path.join(d, os.path.basename(f)))


def course_stats(name):
    n = s = 0
    try:
        with open(os.path.join(cdir(name), "people.json"), encoding="utf-8") as f:
            n = len(json.load(f))
    except Exception:
        pass
    try:
        with open(os.path.join(cdir(name), "attendance.csv"), encoding="utf-8-sig") as f:
            first = next(csv.reader(f), [])
            s = max(0, len(first) - 1)
    except Exception:
        pass
    return n, s


def load_people():
    try:
        with open(PEOPLE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_people(p):
    with open(PEOPLE_FILE, "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False)


def norm(v):
    v = np.asarray(v, dtype=np.float32).flatten()
    return v / (np.linalg.norm(v) + 1e-9)


def match(emb, people):
    best, bs = "", -1.0
    for name, embs in people.items():
        for e in embs:
            sc = float(np.dot(emb, norm(e)))
            if sc > bs:
                best, bs = name, sc
    return (best, bs) if bs >= THRESH else ("", bs)


def read_sheet():
    if not os.path.exists(SHEET_FILE):
        return ["name"], {}
    with open(SHEET_FILE, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return ["name"], {}
    header, data = rows[0], {}
    for r in rows[1:]:
        if r:
            r = r + [""] * (len(header) - len(r))
            data[r[0]] = r[1:len(header)]
    return header, data


def write_sheet(session, present, all_names):
    header, data = read_sheet()
    if session in header:
        c = header.index(session) - 1
    else:
        header.append(session)
        c = len(header) - 2
        for k in data:
            data[k].append("")
    for n in all_names:
        if n not in data:
            data[n] = [""] * (len(header) - 1)
    for n in data:
        data[n] = (data[n] + [""] * (len(header) - 1))[:len(header) - 1]
        data[n][c] = "1" if n in present else ""
    with open(SHEET_FILE, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for n in sorted(data):
            w.writerow([n] + data[n])


def remove_session(name):
    header, data = read_sheet()
    if name not in header:
        return
    i = header.index(name)
    header.pop(i)
    with open(SHEET_FILE, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for n in sorted(data):
            row = list(data[n])
            if i - 1 < len(row):
                row.pop(i - 1)
            w.writerow([n] + row)


def list_images(folder, depth=2):
    out = []
    folder = folder.rstrip(os.sep)
    for root, dirs, files in os.walk(folder):
        if root[len(folder):].count(os.sep) >= depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.lower().endswith(IMG_EXT) and not f.startswith("."):
                out.append(os.path.join(root, f))
    return out


def new_photos(folder, since):
    res = []
    for p in list_images(folder):
        try:
            if os.path.getmtime(p) >= since:
                res.append(p)
        except OSError:
            pass
    return sorted(res, key=os.path.getmtime)


# ================================================================ vision
def download(url, path, min_size, hook=None):
    if os.path.exists(path) and os.path.getsize(path) > min_size:
        return
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp, hook)
    if os.path.getsize(tmp) <= min_size:
        raise IOError("bad download: " + os.path.basename(path))
    os.replace(tmp, path)


def load_models(hook=None):
    download(DET_URL, DET_FILE, 100000, hook)
    download(REC_URL, REC_FILE, 10000000, hook)
    det = cv2.FaceDetectorYN.create(DET_FILE, "", (320, 320), 0.7, 0.3, 5000)
    rec = cv2.FaceRecognizerSF.create(REC_FILE, "")
    return det, rec


def face_thumb(img, box, size=112):
    x, y, w, h = [int(v) for v in box]
    m = int(0.25 * max(w, h))
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    ch, cw = crop.shape[:2]
    s = max(ch, cw)                                   # pad to a square
    sq = np.zeros((s, s, 3), np.uint8)
    sq[(s - ch) // 2:(s - ch) // 2 + ch, (s - cw) // 2:(s - cw) // 2 + cw] = crop
    return cv2.resize(sq, (size, size))


def detect_faces(path):
    img = cv2.imread(path)
    if img is None:
        return []
    h, w = img.shape[:2]
    s = min(1.0, MAXSIDE / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
        h, w = img.shape[:2]
    DET.setInputSize((w, h))
    _, faces = DET.detect(img)
    out = []
    if faces is None:
        return out
    for f in faces:
        try:
            emb = norm(REC.feature(REC.alignCrop(img, f)))
        except Exception:
            continue
        out.append({"emb": emb, "thumb": face_thumb(img, f[:4]),
                    "src": os.path.basename(path), "text": "", "auto": ""})
    return out


def cv_texture(img):
    # Android (OpenGL ES) has no BGR textures -> convert to RGBA, flip for GL
    img = cv2.flip(cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2RGBA), 0)
    h, w = img.shape[:2]
    tex = Texture.create(size=(w, h), colorfmt="rgba")
    tex.blit_buffer(np.ascontiguousarray(img).tobytes(), colorfmt="rgba", bufferfmt="ubyte")
    return tex


def gallery_thumb(path, size=180):
    im = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_8)
    if im is None:
        return None
    h, w = im.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(im[y0:y0 + s, x0:x0 + s], (size, size))


# ================================================================ widgets
class Lbl(Label):
    """Label that takes logical (Persian/English) text."""

    def __init__(self, text="", size=15, color="text", bold=False, align=None,
                 wrap=0, **kw):
        raw = str(text)
        shown = fa(raw, wrap)
        if "height" not in kw and "size_hint_y" not in kw:
            kw["size_hint_y"] = None
            kw["height"] = dp(size * 1.6 * (shown.count("\n") + 1) + 4)
        if align is None:
            align = "right" if (RTL() or has_rtl(raw)) else "left"
        super().__init__(text=shown, font_size=sp(size), color=col(color),
                         bold=bold, font_name=FONT, halign=align, valign="middle", **kw)
        self.bind(size=self._ts)

    def _ts(self, *a):
        self.text_size = (self.width, self.height)

    def set(self, raw, wrap=0):
        self.text = fa(raw, wrap)


class Box(BoxLayout):
    """BoxLayout with a rounded background (and optional border)."""

    def __init__(self, bg=None, radius=12, line=None, lw=2, **kw):
        super().__init__(**kw)
        self._rad, self._lw = dp(radius), dp(lw)
        with self.canvas.before:
            self._c1 = Color(*(col(line) if line else (0, 0, 0, 0)))
            self._r1 = RoundedRectangle(radius=[self._rad])
            self._bgk = bg
            self._c2 = Color(*(col(bg) if bg else (0, 0, 0, 0)))
            self._r2 = RoundedRectangle(radius=[max(0, self._rad - (self._lw if line else 0))])
        self._line = line
        self.bind(pos=self._upd, size=self._upd)

    def _upd(self, *a):
        self._r1.pos, self._r1.size = self.pos, self.size
        i = self._lw if self._line else 0
        self._r2.pos = (self.x + i, self.y + i)
        self._r2.size = (max(0, self.width - 2 * i), max(0, self.height - 2 * i))

    def set_bg(self, k):
        self._bgk = k
        self._c2.rgba = col(k)


class PBox(ButtonBehavior, Box):
    def on_state(self, inst, val):
        if not self._bgk:
            return
        base = col(self._bgk)
        self._c2.rgba = mix(base, col("text"), 0.14) if val == "down" else base


KINDS = {"primary": ("primary", "on_primary"), "soft": ("card2", "text"),
         "danger": ("danger", "on_danger"), "dsoft": ("card2", "danger"),
         "ok": ("ok", "on_ok")}


def btn(text, cb=None, kind="primary", h=52, size=16, **kw):
    bg, fg = KINDS[kind]
    if "size_hint" not in kw:
        kw.setdefault("size_hint_y", None)
    b = PBox(bg=bg, radius=14, height=dp(h), padding=[dp(10), 0], **kw)
    b.lbl = Lbl(text, size=size, color=fg, bold=True, align="center", size_hint_y=1)
    b.add_widget(b.lbl)
    if cb:
        b.bind(on_release=lambda *_: cb())
    return b


def row(*ws, spacing=8, **kw):
    """Horizontal row; mirrored automatically when the UI is right-to-left."""
    r = BoxLayout(orientation="horizontal", spacing=dp(spacing), **kw)
    for w in (reversed(ws) if RTL() else ws):
        r.add_widget(w)
    return r


def vlist(spacing=10):
    b = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(spacing))
    b.bind(minimum_height=b.setter("height"))
    return b


def scroll(child, **kw):
    sv = ScrollView(bar_width=dp(4), scroll_type=["bars", "content"], **kw)
    sv.add_widget(child)
    return sv


class FaInput(FloatLayout):
    """Text field that shows Persian correctly while you type.

    A transparent TextInput receives the keyboard; a Label next to it draws
    the shaped, right-to-left version of what was typed."""

    def __init__(self, text="", hint="", size=17, on_text=None, **kw):
        kw.setdefault("size_hint_y", None)
        kw.setdefault("height", dp(50))
        super().__init__(**kw)
        self.hint, self.on_text = hint, on_text
        with self.canvas.before:
            self._c1 = Color(*col("line"))
            self._r1 = RoundedRectangle(radius=[dp(12)])
            Color(*col("field"))
            self._r2 = RoundedRectangle(radius=[dp(11)])
        self.bind(pos=self._u, size=self._u)
        self.disp = Label(font_name=FONT, font_size=sp(size), valign="middle",
                          padding=(dp(12), 0), pos_hint={"x": 0, "y": 0})
        self.add_widget(self.disp)
        self.ti = TextInput(
            text=text, multiline=False, write_tab=False, use_bubble=False,
            use_handles=False, foreground_color=(0, 0, 0, 0),
            cursor_color=(0, 0, 0, 0), selection_color=(0, 0, 0, 0),
            background_color=(0, 0, 0, 0), background_normal="",
            background_active="", background_disabled_normal="",
            font_name=FONT, font_size=sp(size), padding=[dp(12), dp(12)],
            pos_hint={"x": 0, "y": 0})
        self.add_widget(self.ti)
        self.ti.bind(text=self._refresh, focus=self._refresh)
        self.ti.bind(on_text_validate=lambda *_: setattr(self.ti, "focus", False))
        self.disp.bind(size=self._refresh)
        self._refresh()

    def _u(self, *a):
        self._r1.pos, self._r1.size = self.pos, self.size
        self._r2.pos = (self.x + dp(1.5), self.y + dp(1.5))
        self._r2.size = (self.width - dp(3), self.height - dp(3))

    @property
    def text(self):
        return self.ti.text

    @text.setter
    def text(self, v):
        self.ti.text = v

    def _refresh(self, *a):
        t, foc = self.ti.text, self.ti.focus
        if t:
            shown, color, rtl = visual(t), col("text"), has_rtl(t)
        else:
            shown = "" if foc else visual(self.hint)
            color, rtl = col("muted"), RTL()
        if foc:
            shown = ("|" + shown) if rtl else (shown + "|")
        self.disp.text, self.disp.color = shown, color
        self.disp.halign = "right" if rtl else "left"
        self.disp.text_size = (self.disp.width, self.disp.height)
        self._c1.rgba = col("primary") if foc else col("line")
        if self.on_text:
            self.on_text(t)

    def request_focus(self, delay=0.25):
        Clock.schedule_once(lambda dt: setattr(self.ti, "focus", True), delay)


class Tile(ButtonBehavior, FloatLayout):
    """Gallery thumbnail; shows an order badge when selected."""
    sel = BooleanProperty(False)

    def __init__(self, tex, **kw):
        super().__init__(**kw)
        with self.canvas.before:
            self._c = Color(*col("line"))
            self._r = RoundedRectangle(radius=[dp(10)])
        self.img = Image(texture=tex, size_hint=(None, None))
        self.add_widget(self.img)
        self.badge = Box(bg="primary", radius=11, size_hint=(None, None),
                         size=(dp(26), dp(26)), pos_hint={"x": .05, "top": .95}, opacity=0)
        self.bl = Lbl("", size=13, color="on_primary", bold=True, align="center", size_hint_y=1)
        self.badge.add_widget(self.bl)
        self.add_widget(self.badge)
        self.bind(pos=self._u, size=self._u, sel=self._s)

    def _u(self, *a):
        self._r.pos, self._r.size = self.pos, self.size
        pad = dp(4) if self.sel else dp(2)
        s = max(0, min(self.width, self.height) - 2 * pad)
        self.img.size = (s, s)
        self.img.pos = (self.x + (self.width - s) / 2, self.y + (self.height - s) / 2)

    def _s(self, *a):
        self._c.rgba = col("primary") if self.sel else col("line")
        self.badge.opacity = 1 if self.sel else 0
        self._u()


class Bar(Widget):
    value = NumericProperty(0)

    def __init__(self, **kw):
        kw.setdefault("size_hint_y", None)
        kw.setdefault("height", dp(10))
        super().__init__(**kw)
        with self.canvas:
            Color(*col("card2"))
            self.t = RoundedRectangle(radius=[dp(5)])
            Color(*col("primary"))
            self.f = RoundedRectangle(radius=[dp(5)])
        self.bind(pos=self._u, size=self._u, value=self._u)

    def _u(self, *a):
        self.t.pos, self.t.size = self.pos, self.size
        self.f.pos = self.pos
        self.f.size = (self.width * max(0.0, min(1.0, self.value)), self.height)


# ================================================================ application
class AttApp(App):
    title = "Attendance"

    # ---------------------------------------------------------- skeleton
    def build(self):
        self.events, self.token, self.back_fn = [], 0, None
        self.sync_entries = None
        self.cls, self.mode, self.date, self.faces = "", "", "", []
        Window.clearcolor = col("bg")
        Window.softinput_mode = "below_target"
        Window.bind(on_keyboard=self.on_key)
        self.root_box = BoxLayout(orientation="vertical")
        self.bar = Box(bg="bar", radius=0, size_hint_y=None, height=dp(58),
                       padding=[dp(10), dp(8)], spacing=dp(8))
        self.body = BoxLayout(orientation="vertical", padding=dp(12), spacing=dp(10))
        self.root_box.add_widget(self.bar)
        self.root_box.add_widget(self.body)
        migrate_old()
        Clock.schedule_once(lambda dt: self.go(self.s_courses), 0)
        return self.root_box

    def on_key(self, win, key, *a):
        if key == 27 and self.back_fn:
            self.back_fn()
            return True
        return False

    def go(self, fn, *args):
        for ev in self.events:
            ev.cancel()
        self.events = []
        self.sync_entries = None
        self.token += 1
        self.cur = (fn, args)
        Window.clearcolor = col("bg")
        self.bar.set_bg("bar")
        self.body.clear_widgets()
        self.bar.clear_widgets()
        fn(*args)

    def refresh(self):
        if self.sync_entries:
            self.sync_entries()
        fn, args = self.cur
        self.go(fn, *args)

    def set_bar(self, title, back=None):
        self.back_fn = back
        items = []
        if back:
            items.append(self.bar_btn(tr("back"), back))
        else:
            items.append(Widget(size_hint_x=None, width=dp(86)))
        items.append(Lbl(title, size=19, bold=True, align="center", size_hint_y=1))
        items.append(self.bar_btn(tr("settings"), self.open_settings))
        for w in (reversed(items) if RTL() else items):
            self.bar.add_widget(w)

    def bar_btn(self, text, cb):
        b = PBox(bg="card2", radius=12, size_hint=(None, 1), width=dp(86), padding=[dp(4), 0])
        b.add_widget(Lbl(text, size=14, align="center", size_hint_y=1))
        b.bind(on_release=lambda *_: cb())
        return b

    # ---------------------------------------------------------- modals
    def modal(self, widgets, dismiss=True, width=.92):
        mv = ModalView(size_hint=(width, None), auto_dismiss=dismiss, background="",
                       background_color=(0, 0, 0, 0), overlay_color=(0, 0, 0, .6))
        card = Box(orientation="vertical", bg="card", radius=20, padding=dp(16),
                   spacing=dp(10), size_hint_y=None)
        card.bind(minimum_height=card.setter("height"))
        card.bind(height=lambda i, h: setattr(mv, "height", h))
        for w in widgets:
            card.add_widget(w)
        mv.add_widget(card)
        mv.open()
        return mv

    def toast(self, msg):
        mv = ModalView(size_hint=(None, None), size=(Window.width * .86, dp(52)),
                       pos_hint={"center_x": .5, "y": .08}, background="",
                       background_color=(0, 0, 0, 0), overlay_color=(0, 0, 0, 0))
        b = Box(bg="card2", radius=14, line="line", lw=1.5, padding=[dp(12), 0])
        b.add_widget(Lbl(msg, size=15, align="center", size_hint_y=1))
        mv.add_widget(b)
        mv.open()
        Clock.schedule_once(lambda dt: mv.dismiss(), 2.2)

    def ask_text(self, title, prompt, initial, cb, hint=""):
        fi = FaInput(text=initial, hint=hint)
        holder = {}

        def ok():
            v = fi.text.strip()
            holder["mv"].dismiss()
            if v:
                cb(v)

        mv = self.modal([
            Lbl(title, size=19, bold=True), Lbl(prompt, size=14, color="muted", wrap=34),
            fi, row(btn(tr("ok"), ok, h=48), btn(tr("cancel"), lambda: holder["mv"].dismiss(),
                                                 "soft", h=48),
                    size_hint_y=None, height=dp(48))])
        holder["mv"] = mv
        fi.ti.bind(on_text_validate=lambda *_: ok())
        fi.request_focus()

    def confirm(self, msg, yes, danger=True):
        holder = {}

        def go_yes():
            holder["mv"].dismiss()
            yes()

        holder["mv"] = self.modal([
            Lbl(msg, size=16, wrap=32),
            row(btn(tr("delete") if danger else tr("ok"), go_yes,
                    "danger" if danger else "primary", h=48),
                btn(tr("cancel"), lambda: holder["mv"].dismiss(), "soft", h=48),
                size_hint_y=None, height=dp(48))])

    # ---------------------------------------------------------- settings
    def open_settings(self):
        holder = {}

        def seg(options, key):
            items = []
            for val, label in options:
                b = btn(label, None, "primary" if CFG[key] == val else "soft", h=44, size=15)
                b.bind(on_release=lambda *_, v=val: change(key, v))
                items.append(b)
            return row(*items, size_hint_y=None, height=dp(44))

        def change(key, val):
            if CFG[key] == val:
                return
            CFG[key] = val
            save_cfg()
            fi_save()
            holder["mv"].dismiss()
            self.refresh()
            self.open_settings()

        folder0 = CFG["folder"] or self.default_folder()
        fi = FaInput(text=folder0, size=14)

        def fi_save():
            v = fi.text.strip()
            if v != folder0:
                CFG["folder"] = v
                save_cfg()

        def close():
            fi_save()
            holder["mv"].dismiss()

        holder["mv"] = self.modal([
            Lbl(tr("settings"), size=20, bold=True),
            Lbl(tr("theme"), size=14, color="muted"),
            seg([("dark", tr("dark")), ("light", tr("light"))], "theme"),
            Lbl(tr("language"), size=14, color="muted"),
            seg([("fa", "فارسی"), ("en", "English")], "lang"),
            Lbl(tr("calendar"), size=14, color="muted"),
            seg([("jalali", tr("jalali")), ("gregorian", tr("gregorian"))], "cal"),
            Lbl(tr("folder"), size=14, color="muted"), fi,
            btn(tr("close"), close, "primary", h=48)])

    def default_folder(self):
        return next((d for d in CAMERA_DIRS if os.path.isdir(d)), BASE)

    def folder(self):
        return CFG["folder"] or self.default_folder()

    # ---------------------------------------------------------- courses
    def s_courses(self):
        self.set_bar(tr("app_title"))
        lst = vlist()
        courses = list_courses()
        for c in courses:
            lst.add_widget(self.course_card(c))
        if not courses:
            lst.add_widget(Widget(size_hint_y=None, height=dp(70)))
            lst.add_widget(Lbl(tr("empty_courses"), size=16, color="muted",
                               align="center", wrap=32))
        self.body.add_widget(scroll(lst))
        self.body.add_widget(btn(tr("new_course"), self.new_course, "primary", 56, 18))

    def course_card(self, c):
        n, s = course_stats(c)
        card = PBox(bg="card", radius=16, size_hint_y=None, height=dp(84),
                    padding=[dp(14), dp(8)], spacing=dp(10))
        info = BoxLayout(orientation="vertical", padding=[0, dp(4)])
        info.add_widget(Lbl(c, size=19, bold=True, size_hint_y=1))
        info.add_widget(Lbl(tr("n_students", n=n) + "  ·  " + tr("n_sessions", n=s),
                            size=13, color="muted", size_hint_y=1))
        d = btn(tr("delete"), lambda c=c: self.del_course(c), "dsoft", 40, 14,
                size_hint=(None, None), width=dp(74), pos_hint={"center_y": .5})
        card.add_widget(row(info, d))
        card.bind(on_release=lambda *_: self.open_course(c))
        return card

    def new_course(self):
        def made(name):
            name = "".join(ch for ch in name if ch not in '/\\:*?"<>|').strip() or "course"
            self.open_course(name)
        self.ask_text(tr("new_course_t"), tr("course_name"), "", made, tr("name_hint"))

    def del_course(self, c):
        import shutil

        def yes():
            shutil.rmtree(cdir(c), ignore_errors=True)
            self.refresh()
        self.confirm(tr("del_course_q", c=c), yes)

    def open_course(self, c):
        self.cls = c
        set_course(c)
        self.go(self.s_menu)

    # ---------------------------------------------------------- course menu
    def s_menu(self):
        self.set_bar(self.cls, lambda: self.go(self.s_courses))
        n, s = course_stats(self.cls)
        self.body.add_widget(Lbl(tr("n_students", n=n) + "  ·  " + tr("n_sessions", n=s),
                                 size=14, color="muted"))
        for key, sub, acc, fn in (
                ("act_label", "act_label_s", "primary", lambda: self.begin("label")),
                ("act_att", "act_att_s", "ok", lambda: self.begin("attend")),
                ("act_rec", "act_rec_s", "warn", lambda: self.go(self.s_records))):
            card = PBox(bg="card", radius=18, size_hint_y=None, height=dp(100),
                        padding=[dp(14), dp(10)], spacing=dp(14))
            stripe = Box(bg=acc, radius=3, size_hint=(None, 1), width=dp(6))
            txt = BoxLayout(orientation="vertical")
            txt.add_widget(Lbl(tr(key), size=20, bold=True, size_hint_y=1))
            txt.add_widget(Lbl(tr(sub), size=14, color="muted", size_hint_y=1))
            card.add_widget(row(stripe, txt))
            card.bind(on_release=lambda *_, f=fn: f())
            self.body.add_widget(card)
        self.body.add_widget(Widget())

    # ---------------------------------------------------------- start a job
    def begin(self, mode):
        self.mode, self.date = mode, ""
        if not os.path.isdir(self.folder()):
            self.toast(tr("folder_bad"))
            self.open_settings()
            return
        nxt = lambda: self.ensure_models(lambda: self.go(self.s_photos))
        if mode == "attend":
            self.ask_session(lambda name: (setattr(self, "date", name), nxt()))
        else:
            nxt()

    def ask_session(self, cb):
        header, _ = read_sheet()
        sessions = header[1:]

        def got(d):
            if d not in sessions:
                return cb(d)
            prev = [h for h in sessions if h == d or h.startswith(d + " #")]
            k, default = 2, ""
            while True:
                default = "%s #%d" % (d, k)
                if default not in sessions:
                    break
                k += 1
            self.ask_text(tr("session_name"), tr("exists_p", names=", ".join(prev)),
                          default, got2)

        def got2(d):
            if d in sessions:
                self.confirm(tr("overwrite_q", s=d), lambda: cb(d), danger=False)
            else:
                cb(d)

        self.ask_text(tr("session_date"), tr("session_date_p"), today_str(), got)

    def ensure_models(self, then):
        global DET, REC
        if DET is not None:
            return then()
        self.go(self.s_loading)
        tok = self.token

        def hook(count, block, total):
            if total > 0:
                Clock.schedule_once(lambda dt: setattr(self.load_bar, "value",
                                                       min(1.0, count * block / total)))

        def work():
            global DET, REC
            try:
                det, rec = load_models(hook)
            except Exception as e:
                Clock.schedule_once(lambda dt: failed(e))
                return
            DET, REC = det, rec
            Clock.schedule_once(lambda dt: then() if tok == self.token else None)

        def failed(e):
            if tok != self.token:
                return
            self.go(self.s_menu)
            holder = {}
            holder["mv"] = self.modal([
                Lbl(tr("model_err", e=str(e)[:150]), size=14, wrap=34),
                btn(tr("close"), lambda: holder["mv"].dismiss(), "soft", 46)])

        threading.Thread(target=work, daemon=True).start()

    def s_loading(self):
        self.set_bar(tr("loading"), lambda: self.go(self.s_menu))
        self.body.add_widget(Widget())
        self.body.add_widget(Lbl(tr("loading_models"), size=16, align="center", wrap=32))
        self.load_bar = Bar()
        self.body.add_widget(self.load_bar)
        self.body.add_widget(Widget())

    # ---------------------------------------------------------- photos
    def s_photos(self):
        self.t0 = time.time()
        self.new_list = []
        title = tr("act_att") if self.mode == "attend" else tr("act_label")
        self.set_bar(title, lambda: self.go(self.s_menu))
        if self.mode == "attend":
            chip = Box(bg="card", radius=12, size_hint_y=None, height=dp(44), padding=[dp(14), 0])
            chip.add_widget(Lbl(tr("att_review", d=self.date), size=16, bold=True,
                                color="primary", size_hint_y=1))
            self.body.add_widget(chip)
        self.body.add_widget(Lbl(tr("photos_hint"), size=15, color="muted", wrap=36))
        self.body.add_widget(btn(tr("open_cam"), self.open_camera, "soft", 56, 17))
        self.use_btn = btn(tr("use_new0"), self.use_new, "primary", 60, 17)
        self.use_btn.opacity = .45
        self.body.add_widget(self.use_btn)
        self.body.add_widget(btn(tr("gallery"), lambda: self.go(self.s_gallery), "soft", 56, 17))
        self.body.add_widget(Widget())
        self.events.append(Clock.schedule_interval(self.poll_new, 2))

    def poll_new(self, dt):
        try:
            self.new_list = new_photos(self.folder(), self.t0 - 2)
        except Exception:
            self.new_list = []
        n = len(self.new_list)
        self.use_btn.lbl.set(tr("use_new", n=n) if n else tr("use_new0"))
        self.use_btn.opacity = 1 if n else .45

    def use_new(self):
        self.poll_new(0)
        if self.new_list:
            self.process(self.new_list)

    def open_camera(self):
        act = "android.media.action.STILL_IMAGE_CAMERA"
        try:
            try:
                from android import mActivity as activity
            except Exception:
                from jnius import autoclass
                activity = autoclass("org.kivy.android.PythonActivity").mActivity
            from jnius import autoclass
            Intent = autoclass("android.content.Intent")
            activity.startActivity(Intent(act))
            return
        except Exception:
            pass
        try:
            if os.system("am start -a %s >/dev/null 2>&1" % act) == 0:
                return
        except Exception:
            pass
        self.toast(tr("cam_fail"))

    # ---------------------------------------------------------- gallery
    def s_gallery(self):
        self.set_bar(tr("gallery"), lambda: self.go(self.s_photos))
        self.body.add_widget(Lbl(tr("loading"), size=16, color="muted", align="center"))
        tok = self.token

        def work():
            files = list_images(self.folder())
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            items = []
            for p in files[:GALLERY_MAX]:
                t = gallery_thumb(p)
                if t is not None:
                    items.append((p, t))
            Clock.schedule_once(lambda dt: self.build_gallery(items) if tok == self.token else None)

        threading.Thread(target=work, daemon=True).start()

    def build_gallery(self, items):
        self.body.clear_widgets()
        if not items:
            self.body.add_widget(Lbl(tr("no_images"), size=16, color="muted", align="center"))
            return
        self.sel = []
        gap = dp(6)
        side = (Window.width - dp(24) - 2 * gap) / 3.0
        grid = GridLayout(cols=3, size_hint_y=None, spacing=gap)
        grid.bind(minimum_height=grid.setter("height"))
        self.go_btn = btn(tr("gallery_use", n=0), self.use_selected, "primary", 56, 17)
        self.go_btn.opacity = .45

        def toggle(tile, path):
            if path in self.sel:
                self.sel.remove(path)
                tile.sel = False
            else:
                self.sel.append(path)
                tile.sel = True
            for t in grid.children:
                if t.sel:
                    t.bl.set(str(self.sel.index(t.path) + 1))
            self.go_btn.lbl.set(tr("gallery_use", n=len(self.sel)))
            self.go_btn.opacity = 1 if self.sel else .45

        for p, thumb in items:
            tile = Tile(cv_texture(thumb), size_hint_y=None, height=side)
            tile.path = p
            if os.path.getmtime(p) >= self.t0 - 2:          # pre-select new photos
                self.sel.append(p)
                tile.sel = True
            tile.bind(on_release=lambda t, p=p: toggle(t, p))
            grid.add_widget(tile)
        for t in grid.children:
            if t.sel:
                t.bl.set(str(self.sel.index(t.path) + 1))
        self.go_btn.lbl.set(tr("gallery_use", n=len(self.sel)))
        self.go_btn.opacity = 1 if self.sel else .45
        self.body.add_widget(scroll(grid))
        self.body.add_widget(self.go_btn)

    def use_selected(self):
        if self.sel:
            self.process(list(self.sel))

    # ---------------------------------------------------------- processing
    def process(self, photos):
        self.go(self.s_processing, len(photos))
        tok = self.token

        def work():
            faces = []
            for i, p in enumerate(photos):
                Clock.schedule_once(lambda dt, i=i: self.progress(tok, i + 1, len(photos)))
                try:
                    faces += detect_faces(p)
                except Exception:
                    pass
            Clock.schedule_once(lambda dt: self.after_detect(tok, faces))

        threading.Thread(target=work, daemon=True).start()

    def s_processing(self, n):
        self.set_bar(tr("loading"), lambda: self.go(self.s_menu))
        self.body.add_widget(Widget())
        self.prog_lbl = Lbl(tr("processing", i=1, n=n), size=17, align="center")
        self.body.add_widget(self.prog_lbl)
        self.prog_bar = Bar()
        self.body.add_widget(self.prog_bar)
        self.body.add_widget(Widget())

    def progress(self, tok, i, n):
        if tok == self.token:
            self.prog_lbl.set(tr("processing", i=i, n=n))
            self.prog_bar.value = (i - 1) / float(n)

    def after_detect(self, tok, faces):
        if tok != self.token:
            return
        if not faces:
            self.go(self.s_menu)
            self.toast(tr("no_faces"))
            return
        self.faces = faces
        self.people = load_people()
        for f in self.faces:
            name, sc = match(f["emb"], self.people) if self.people else ("", 0.0)
            f["auto"], f["text"], f["score"] = name, name, sc
        self.go(self.s_review)

    # ---------------------------------------------------------- review
    def s_review(self):
        self.set_bar(tr("att_review", d=self.date) if self.mode == "attend"
                     else tr("act_label"), lambda: self.go(self.s_menu))
        head = Lbl(tr("faces_n", n=len(self.faces)) + "  ·  " + tr("write_names"),
                   size=14, color="muted", wrap=40)
        self.body.add_widget(head)
        lst = vlist(8)
        entries = []
        for f in self.faces:
            card = Box(bg="card", radius=14, size_hint_y=None, height=dp(104),
                       padding=dp(8), spacing=dp(10))
            img = Image(texture=cv_texture(f["thumb"]), size_hint=(None, 1), width=dp(86))
            fi = FaInput(text=f["text"], hint=tr("name_hint"), size=17)
            known = bool(f["auto"])
            info = Lbl((tr("match", s=int(f["score"] * 100)) if known else tr("unknown"))
                       + "   " + f["src"][-14:], size=12,
                       color="ok" if known else "warn")
            col2 = BoxLayout(orientation="vertical", spacing=dp(4))
            col2.add_widget(info)
            col2.add_widget(fi)
            parts = [img, col2]
            if self.people:
                parts.append(btn(tr("pick"), lambda fi=fi: self.pick_name(fi), "soft", 44, 14,
                                 size_hint=(None, None), width=dp(62), pos_hint={"center_y": .5}))
            card.add_widget(row(*parts))
            lst.add_widget(card)
            entries.append((f, fi))
        self.body.add_widget(scroll(lst))
        self.body.add_widget(btn(tr("save"), self.save, "ok", 56, 18))

        def sync():
            for f, fi in entries:
                f["text"] = fi.text
        self.sync_entries = sync

    def pick_name(self, fi):
        names = sorted(self.people)
        holder = {}
        lst = vlist(6)
        for n in names:
            def choose(n=n):
                fi.text = n
                holder["mv"].dismiss()
            lst.add_widget(btn(n, choose, "soft", 46, 16))
        sv = scroll(lst, size_hint_y=None, height=min(dp(380), len(names) * dp(52)))
        holder["mv"] = self.modal([Lbl(tr("pick_title"), size=19, bold=True), sv,
                                   btn(tr("cancel"), lambda: holder["mv"].dismiss(), "soft", 46)])

    def save(self):
        self.sync_entries()
        people = self.people
        names = [f["text"].strip() for f in self.faces]
        for f, n in zip(self.faces, names):
            if not n:
                continue
            lst = people.setdefault(n, [])
            if self.mode == "label" or f["auto"] != n:       # learn only new/corrected
                lst.append([float(v) for v in f["emb"]])
                del lst[:-MAX_EMB]
        save_people(people)
        if self.mode == "attend":
            present = set(n for n in names if n)
            write_sheet(self.date, present, set(people))
            absent = sorted(set(people) - present)
            self.go(self.s_menu)
            holder = {}
            widgets = [Lbl(tr("result_title"), size=20, bold=True),
                       Lbl(tr("present_n", n=len(present)), size=17, color="ok", bold=True),
                       Lbl(tr("absent_n", n=len(absent)), size=17, color="danger", bold=True)]
            if absent:
                widgets.append(Lbl(tr("absent_names") + " " + "، ".join(absent) if RTL()
                                   else tr("absent_names") + " " + ", ".join(absent),
                                   size=14, color="muted", wrap=34))
            widgets.append(btn(tr("ok"), lambda: holder["mv"].dismiss(), "primary", 48))
            holder["mv"] = self.modal(widgets)
        else:
            self.go(self.s_menu)
            self.toast(tr("saved_n", n=len(people)))

    # ---------------------------------------------------------- records
    def s_records(self):
        self.set_bar(tr("rec_title"), lambda: self.go(self.s_menu))
        header, data = read_sheet()
        sessions = header[1:]
        if not sessions:
            self.body.add_widget(Widget())
            self.body.add_widget(Lbl(tr("no_sessions"), size=16, color="muted", align="center"))
            self.body.add_widget(Widget())
            return
        names = sorted(data)
        self.body.add_widget(btn(tr("del_session"), self.del_session, "dsoft", 44, 15))
        rh, nw, cw = dp(42), dp(132), dp(104)
        nrows = len(names) + 2

        def cell(text, w, bg=None, fg="text", bold=False, size=13):
            c = Box(bg=bg, radius=8, size_hint=(None, None), size=(w, rh), padding=[dp(6), 0])
            c.add_widget(Lbl(text, size=size, color=fg, bold=bold, size_hint_y=1,
                             align=None if w == nw else "center", shorten=True,
                             shorten_from="right"))
            return c

        left = GridLayout(cols=1, size_hint=(None, None), width=nw, height=rh * nrows,
                          spacing=dp(2))
        left.height = (rh + dp(2)) * nrows
        left.add_widget(cell(tr("name_col"), nw, "card2", bold=True))
        for n in names:
            left.add_widget(cell(n, nw, "card"))
        left.add_widget(cell(tr("present_row"), nw, "card2", bold=True))

        right = GridLayout(cols=len(sessions) + 1, size_hint=(None, None), spacing=dp(2))
        right.width = (cw + dp(2)) * (len(sessions) + 1)
        right.height = (rh + dp(2)) * nrows

        def add_row(cells):
            for c in (reversed(cells) if RTL() else cells):
                right.add_widget(c)

        add_row([cell(s, cw, "card2", bold=True, size=12) for s in sessions]
                + [cell(tr("total_col"), cw, "card2", bold=True)])
        counts = [0] * len(sessions)
        for n in names:
            vals = (data[n] + [""] * len(sessions))[:len(sessions)]
            cells = []
            for j, v in enumerate(vals):
                if v == "1":
                    counts[j] += 1
                    cells.append(cell("1", cw, "okbg", "ok", True, 15))
                else:
                    cells.append(cell("-", cw, "card", "muted"))
            cells.append(cell("%d / %d" % (vals.count("1"), len(sessions)), cw, "card"))
            add_row(cells)
        add_row([cell(str(c), cw, "card2", bold=True) for c in counts]
                + [cell("", cw, "card2")])

        lsv = ScrollView(size_hint=(None, 1), width=nw, do_scroll_x=False,
                         do_scroll_y=False, bar_width=0)
        lsv.add_widget(left)
        rsv = ScrollView(do_scroll_x=True, do_scroll_y=True, bar_width=dp(4),
                         scroll_type=["bars", "content"])
        rsv.add_widget(right)
        rsv.bind(scroll_y=lambda i, v: setattr(lsv, "scroll_y", v))
        # start on the newest sessions (last columns; first ones when mirrored)
        Clock.schedule_once(lambda dt: setattr(rsv, "scroll_x", 0 if RTL() else 1), 0.05)
        self.body.add_widget(row(lsv, rsv, spacing=2))

    def del_session(self):
        header, _ = read_sheet()
        holder = {}
        lst = vlist(6)
        for s in header[1:]:
            def pick(s=s):
                holder["mv"].dismiss()
                self.confirm(tr("del_session_q", s=s), lambda: (remove_session(s), self.refresh()))
            lst.add_widget(btn(s, pick, "dsoft", 46, 16))
        sv = scroll(lst, size_hint_y=None, height=min(dp(380), (len(header) - 1) * dp(52)))
        holder["mv"] = self.modal([Lbl(tr("del_session_t"), size=19, bold=True), sv,
                                   btn(tr("cancel"), lambda: holder["mv"].dismiss(), "soft", 46)])


# ================================================================ fonts
def ensure_fonts():
    os.makedirs(FONT_DIR, exist_ok=True)
    out = {}
    for key, name in (("regular", "Vazirmatn-Regular.ttf"), ("bold", "Vazirmatn-Bold.ttf")):
        p = os.path.join(FONT_DIR, name)
        if not os.path.exists(p) or os.path.getsize(p) < 20000:
            data = zlib.decompress(base64.b85decode("".join(FONT_DATA[key])))
            with open(p, "wb") as f:
                f.write(data)
        out[key] = p
    return out


def main():
    global FONT
    load_cfg()
    try:
        fonts = ensure_fonts()
        LabelBase.register(name="UI", fn_regular=fonts["regular"], fn_bold=fonts["bold"])
        FONT = "UI"
    except Exception as e:
        print("font error:", e)
    AttApp().run()



# ======================================================================
# Embedded Vazirmatn font (SIL OFL 1.1), subset, zlib+base85
# ======================================================================
FONT_DATA = {
    "regular": (
        'c-pLed0dl6^Z2vR^CZCwfe`RSAVN?P<Oo3o<UR!CMg-&#<i0`V6c9l?KtK=#!TY}O2DB<xt=6iw)_T_-YHPJsTYIQv^V{7dfY$f@',
        'zJL7uG$EPnGc&t$?9A*uPbolAlqyw5F;uj>i>sS&tKU}?_1$`k0=DjczJWvhKV>ND``Z*{rQ#kK?4%i+cYp%g-%*s>e&0Y-^Tf%L',
        'UW&Sti?Bq;rp06m&iv(!zJE<o?6HKHoJ<5)L--oeUzLQ^g7_7F<7c4H&nfEDjfo3l;<U9)QYh+jF@jkoBEZPgJTbc74}G7Qn3kI#',
        ')b;F7^!pKt8nHe#BR1xrd!kJgb#5<3alcH9$<Jh#Q-7fEtI&Ae^q8~-#uJ=dDe7DF9X!d*$jOyXqT&#J%%3hrQ{yR$qr>T>c+^me',
        '(qr@(A+P{Ik6}YsN;j-HPfyuJPwlG)zx@qLU<If1tsgy>-hr;uMDxjFYyNc|b2Lr*PsS(U;WltbIt<i(2DIT*w&q<9H~{+4b-A+V',
        '>>JKGN=TWa>kJJ=T&_^x&`@kSQBO~7X(-g^3WSzcRu<-3S^{-VEp35;xs{bfU8u_yXlk+M!&Di!r){flE&?9FAE+GY>^OI_;_tof',
        'FRS@{X?n_mg0a8AzkXXk+Rw-_F8M^!v+Ip(4pyxyzmSo7cdJxOQev!DXB99N#FsSYG*h(f6-Dtv(d<W38Wd$9kk3O;pvT8AG#g%M',
        '->Q+)FRuK^WHDV*A5D6;v`br>4lYh;X#p{C1E;eEPHvI@2J-0#6q?--w3@%8Rryc&IKN8`hr#$QpV7_CB<Z8!;7ceTA$<#+HP@~M',
        'Ch#k^W-UC@1j~R4ZAQsta_R6Sv>}DHl1DXw(?EjYWc9f^2u>^Dag+{%lhx&gAULCd-v=Qd!E=PV^btH<>PY&>upFVD3}E3(`>~v)',
        '-V9*jJnHA>B-Js9n{oYkP6Dlua`P0Jesoq(sjojetEa$w5S}$kJP3;)aO%fXsKf(IsXwVn{df*3@u2Z(Af_LWupf^Pbq>tv$1_y9',
        'I>0AMzaNWJKNcVAdk}^`??R)ikuIrH+Q_mfJw0_jb1i`;mxnZ2lV$K%E2QCkq~zo?y#vzV!FTU+;)*KZn|JUL&@X9TdAM>7>;ZPm',
        'TR5HFmlw28(LYu6#chrTG%rriPwlHKtdYL1p%Ar|=<Oxaw2xBbGD3BW52$%UC$RSX)5y@|w8y_Z_$8%eaYt%C{2g#|3fY?0t$vI3',
        '#<b1bzLR+k_k;w51xmw1Q(@(Nv?BRvzPFLZ>LCwep-LlbrNv@PjfvPqwS|VrWN87iO)W&wiXC-kWZ5qV_Q+~$0q@ARN2MdrXtnI>',
        'z38{2yr{toTw8IsbO)IBMg13G+Roy8bu9<sqo3ETezt0M142&4$X>E$Y&(_*uu%8~vD_KR32`wYM2A__!+-w6+J}WjKepZdh2NvL',
        'cy;^M=8a8dpkVjIWq)mt-|^S7Z##3BT<=NfyuTR9EFU`KIUt#hAsTcD+>rf+;4#n>Q$+Bw3iv`9_EK1upCMTeqXdezWi=3Onk5UX',
        'fw-b+3z!Gip9XrHH^bXq_bxqs{+!df175i?=LWpGgQM|oCD45__Z`r{egR2~)kV@8z=y<y<fIf7k`(v}ZUu?k-`3Q~HXYb<_S7l1',
        'ruy%l4m<u?#=P#^`c0S9=_`1RFlTOrGwff~^%O7GtuK~YBQ>G(q$}FnIh`HQxLw+Z#$!D>Kq$XAo;lEWpV5)F(VEidEiG)#I(VoV',
        'E=Qa$lKsNb<$OwwqV$nf=uec(TAQWR1$sI(;#H!JxYp-#dDvYW($BttKi6#pf_*>_81`v{$lk`Pi!;*#|N6A-aNH0^<hxhv2(Y_4',
        '?@M5N5CFIvo~>?rU6)~>8oH8WwfJ_m;o{G7u%{#QB37lV@LVg1EffBUPta4qt%*+{eDY!tTwei)#FjtAbA)RA2p;6=oIZx-2v8Y>',
        'MWr9hN#qgwu@Fxu$8s`z0E?&}%SmAUQC^->Z$CPxr_|1m&gm&g2Qf${1_;k%vP+F8+9MQO4AMv>O@&Gdh3d3F%-g)VUoQ*b6ReoG',
        '?$PU7=+b`eWXxJ1t58okK_-)ZBG*quc><x_B{PxbDYVTLQAYb3Ky-w1-T6Y=(aTgxXE1sl(h{s|BPoTZwINM=f2_CkpKeA?x{F@b',
        '0*7NgYr$g+kxzMme9A=hMIHMTtY>PJ9&srqwA`g=Sdgt>Ps~0Tn|E!?LHO%>Fb#~^@qMXA?+BmT%+F6fI+0N3!G{Ovo6=`Sv7TFj',
        ')!i)M2GVjqFZW&I3xD`Crw=|BnrQ(y9KKO;$iG-2|Dva)iv5ck^)L4#5hI#bqmz!VYW-~)820Mh@3oGrm#^5i;bePp_9*xh?FZd-',
        'XIn3Q(-{PKz~I`Zh~k@{gq;0ato2(RSqa%Q<}IRQh-^+m93zfwgknA+!WE#kG1EYE)ds{_BR;@t@q{2rTzd7gpx<<YW(GzYcF~}w',
        ';*2kRx%uR``>iu|TKKvpG4q2~0Fk!HXR)_boqh%Org+1yksDVYIX_A+D~^iXGXQm@6IcT11(NA+&zzC!_i{R=wRAQL;5JF4&@j2g',
        '*tv4w@ZJ?JbVPTzbZa+TBZV}LLbu<cDjJ5aIF8=((ZvkD0*%b?#UY^_8y2iT8m9cszZ-{r2!J&EAaa5M#1gFK)=F6sOQ=O~Q(_6z',
        'i9I#&ksZi>K~VQWqmRfO@Mzg*2tKiYG=`7Ib|brq;8Da{<>ODvGRgQu2yRQPb-DuXL&iTu9F`5j|5Ub}j6aFs^9FJAv_c2XODucN',
        'e1WE@j^|~euBq6VEaW@U@H2VWs;a)s>ut)*UI%uuymxQvE+j;pT)m=<>}kfbmy8cugpXcLujyHEAT#eoYEN-OSSAQ$Z2B%_YzX#h',
        '%}b0(Rd{<22hkpm>>lvYs#>T^aGcFfkV+5r_I7}$KTFTiiS+hv$Vc)u-%smGOZsl0D-S8+dUB<ZTxmhBG?%{W?Co6(`d&!CrbFn}',
        '(kyggDP1mgkg6$kVh-Z>LyAluoZr~py#%a!1odc&R#^>~A^rGfHE5H*l@=&`#W+NfBj0H()Y)-;HNAK<di$%=KdeNU`+sLvN~vzV',
        '`ScB=<Wgr|bHArVR1n&IYG^JKOex^(llR~ulwxhT0VLj{_=rKQy?h`9e0AZjf4~E9|DQ8_sMi4X`DZwt@8+>vkmg&{G4B%CHfg)`',
        'qJkq%K8c||#IW)OSO%(Ib@O5NGMK~fLOf<MtC1x%&>DRa%x39Jn6BVKWr)I$>2t|)Y0Kjk@~(~mB&XqN3CKtKZuz+>y~kDDA5^Zm',
        'FwOmC(`S!+*DkL(KM9;!%h7GiefI)|eNpM@C8?!9z}J0~ncJN$9eHEM$v-JIVu->cs|&HASpap;kh4-@Z!i5FwnjGg9a{5ut!(^Y',
        '8-H&+I~SXGHywd3lu7MtMDuH8`ZZ1k&4~ZujO24+==wV*y~l@mzglzapEJ8Ty1PpIkkh%6Snv!!>$7DpbZ^2~)*;I?W*=hD!beK4',
        'Wnm#eYA;^5D=}&G=+h_IhkX}n<{$g~9v&MB!|5#QHj*!8fEJEN95KGcT;Uu=avv&pqWU->q(_XDjMbQI;2NktCUUw>Z<|%P;gs<%',
        'Q|<G7^wqs8+%j)-hlLFfHXW^DGqrR%Uv2IPHKCKK`mhR*Bpc*)Qn5zyd?^vw4FaiY3XP&=U$L)f9ybW(=O{{K5LNliF&BspiQJ8&',
        'MKRGBF-60_!q+c;)!xxzxsS^iSWK3LI&j2+!}NK}7sYwS`RR#;5|L#<ELjTxoX}b*{i4$@_Tf8y_FF~N!x=&{kJ*Y`;w{NMh}1~t',
        ';j4f*Qu>PVD#Ua7^^XrDQQLFQP$V^Vynh2676_H0n*|CK7_l1OQ`H4(XiwGP46TQ65AKD3uWbS<dk+GY`qN*YIDG5Q;geSwZp2-9',
        'e%CHA<8H)VFk{y)cpiQp30R;BpToXL_!d3`0z@ww?WEyohf~sfKUmk|DKdHacFKk?e$=}5rzPc2*0w$_J-)ZC@!;Xcw%v4sQ>V1L',
        'TH5K<39GB&?=I)R{`$Pjg_}1?B!uR}n~vr=R}tO)0Hz~2p-Vh;m;zp+h@9RZe}sDe(eX#9Wdq~6vi|WW!MK6(WXA|o#C^v<ivKAH',
        '_$dCT)UT8dlJp|k7S4Fib80jtq>TF|BH{@p{mGNI!~*4N98DAj;1o)oBeoO)t$y+BUK+EyJ}eGP=+>HqRUOVAea03hv^Z|(;aiNr',
        'T6!<6uwPt(>wgXh{DVYd#am(6U$2XMU6|?&BD2U7i~DLFkklB&?FA&Y<A^JLYD&M{2v<tkDY+s{aRo}I<T6~h&}(g(i#L+v)_m8d',
        ';+d{}pV)y*vvV2`(yJ_?tyiK_sBW33J-@2Na-E&n<8k%u#zcDe3?y8a=2&UC<xQ%8MGcfIiZuBrVAsE*LzOFv;7bs<@{FuknUMh!',
        'UYft3SR2x3;m%%q^tMNN@XbGPCkWl$=(viWAwAFO%)V59;|YCBdZDA(YpK67BO8nS;!xu06!>uFtVBmg!YAF}#;qq$ZowyD%<`fl',
        'a1733Yp!kFd4{%!;W=T!6$nqdY$%6|EKh(WP2|ku@C>;B&H6x9K46x>D;|DI6FdZX^{20$Jaqf+AtKYS;FXXo`L8wp4>Ls6!(>)$',
        'D6&H(buLevQ2)=i7?QQ%b_skp!%1O~PfEM?wl#GgY9g}R{@QI*`e!3PVw99Z2A?Y`94I=FFB*%f0s&C28`B5R!QH*!hd;rPHA_H<',
        'v==@EqPF&Cdbacsr}Mj;)tAksYlm|*q<1!CEyf!KwF2QaRMay1@oFjXDk~XG6zqdfd%*6mpO&NpU8wr<&(hLj&<kg>HC=laZkPne',
        '4`p7%1vw#+xybAKqYNVl$pu><(hZT}MCF1AgqDgeMw^7+Xt|JGS;#9G%t4#AD{jlEqK8deALpOx*wWg#>Bv7%kn%71bVJmc1zYDe',
        'ZvmMu(X)+aUf7j&II-kXcHzcl3&M-c-M-wN*BNb{xh*Yrv26+Bzy(Eq-y+#eR&rqYpEyuPgoFX1$-=bWC#mO_tokgur>wH1uxjJk',
        '>wkQfcfYpBugb4v37Bl=X*0%jTYY3*K+L9?xj8ZJE^{ZD?Wt?n?YFteCu&MKqHT^=w3$?OmAROhZHC6D;8c{Zz5R`&(Sdi~Nobdo',
        'k{WLRC`Jmix*&?$*Yyo&UK8o;K)B`+sgEYRx!ms1vY{*FQYY58qZ?>-+{w(m(*b|!2G`&(Kr6M<&aRTv3Gbz!DJndZ4lgjTr4M|I',
        'UECJ?$t{pMP9jaD$|yP&D7RVxUn%pI-N&%~9Yg<+{joKlU)#9r^mAsi!ukNU3Q2i4;tkvIKya;2LJow1`Q=Q*tD43aD^GW?ZRz54',
        '_KCLMO-sAGmATU=+OuWL9)w{b@;Ac}7yk)^e^1Hv#TD1`kGB++G@&r*_0q#(VTYG8!~0&B7ZjA^hzWTzoOfv=-4%Usy#$BB0X(1I',
        'YihsF?d@D&`K{@h{ox7mKXfyn_RZ+oRF}!>zV*ZYkj#YCL~|57TMwy;68^{zldFJ7ksXOpB{k873ivBh6a5g+5$fDW@NlUkNjl2W',
        'q@tUn*zaBsj3)bFt^&Sp5Y=)0R8N8_AEo*fO#UeMPpNAk<^Cy%0U>BD$0J>nqP65AO+x`W3dVl}?O@Y&>Ek`pCn%`(W98{4Ahp>b',
        'BsUc#H!{!yXw$QS!v!eW4<#3Y{TLu*l5zgo*Bpt)+o4JSimud<kJpy)@kJsYzLKv^8v(sdVDzL3-}l0e2jJJ*+VIi=xZ%u`@t}+O',
        'r%&bJ1?EQ-SsP?zyt{|gXk0JCXoh03SBL|Qk&ke^j}ct@XWMJl;pe;HH~U~80Pyub_)XVm!+;jEsIST{kcmT)_gW68FBx&zIve4$',
        'K{L84&)<j8C>;@!|9~Ng1d-oEOK|H)xDKxS5!`~7C}aDIHj$c2_2}u+&*+Erb#y(7pILKxHb40KVPHI%@Bl6Z>mI;w;O+ZhEjDlH',
        '0}e}1NsodP(3SS0ozV3o(e)=0eHF#p<5=}MC<I0K;Lu(;6wx%G@AYL$zoT`TIy7Dt=X-Jq^pF1*NPy%^7|;bnQF3^ic9af*$zU`6',
        '49T3}=UE|{FH>mja$tqv1P{@XY6aYq=m;6l?du<ZlKOODJkiZ+c?~WdXX$egv{0TGq3lu&^u&4sz!tuH!07b7VE*d+uD!jEUDC1Y',
        'T><ud9?186FVBhwMl;{{CD1L>EM}23simcuDQL;>tH2&GP4=Amg`JJ!kf|S>Sy@=Te`eKxWF-;H9XvB>$OzVQeO&ssaG>(=curPQ',
        'd3SPeMduHEJ{%smMrYlIy@{H3dJ^A|`mp^+L!xsMTxPF{I&uuESQQ$p)yCO9j9uW9f@V#4;8JMTs}-xX0B6S^z>!^N;Q1A?cj<9l',
        'd`>{|ky;j=eF$H(cYz_i+kXw`LO-sP)CLNV^_4y2max8*DP>JL<La+srfA>lpQy-Encr%gYY6&l!m2c*uP6zFf!VV!sxCTM+}yGE',
        '_@n#ZrYB}D?Mm$}&FwU^KbUyyN2w9(zB+SJgSn$tOlxyNof<vbJhShpce}~`imC;I=wRpUgN+^ggHnpJd^92(f&v45LjipIU0(c-',
        '`JVdK>N8B4%LgkbcrW(p`@+}kx4hZzu^(Bdr_@a{w|L~C{y_17KAN4)AWv;+B_Zz?@?DL`186(5z=0G%qLf%Eay|yTa|z(7(ySob',
        '>+`t!8`<q8UUM&QOKA&A4^0Cux)ZnC&it}(i=#MkrfWn{XxO4v^^FCveX@rWKgZc6xhH?g;n1*-kln|FR|Y0GIQqk4yZXT5n;~!5',
        'B*`MDSySeC&xl#r_hoWx2G})096by1oQS+<1F2U|r6fcinB9NzD`!`TX2cMd-{+Xi1wS9U8XMx_5p$q6dX;yeZ7>+_y3@(-vm5g~',
        'Pd#xpa}4rua&QbuO>ano_k(BXsqHR~ubgQ)t8DJVde22}-X+F%@W-6?^z;5-&$qH=TFpYtP3(Lu-F&149`n3_h5cl-o6EkG-Jvb0',
        'SBPsAO&DnNL?ZM8J65e)*^6GUR@T(4L@y)>*%jFd+6-JEBXNBIjpHcB%a$!$2F%ghHT>ZMddpiT|3|K)(RCw~*8#;f_*$@Ac@4f8',
        'j8I&JFGd4p*JL*lXMbaBkf71g_4W0CqZj(+6!JUAcxUi0j2H-<=zq96>A*65z_KEgngmR#zYtwOInZ|)fqahw3_13ulTMx93#OET',
        '$@CrYBOH$sjY-%FE5I7|1tY{~OF$wIEkyD<Gf!bfB;;e~v;;Op&8eJi5n}EgvMHb;Ds;K0B+$$!iWveDcE`K9m|CVr(n;H+XE~ai',
        '<zdX@==n?t=OC$E@eL###)2`>=J0t8Q+^44WeQA&v<wB4P2iRD#ob5gb3ij}Z3YO1n={sh!&mcA^!yX{Yl(0M_iIiwWlMP>q^gWi',
        'Y48#&4Z<G7l~-OQ8kC+sOY^uItl|U+Cpdg74V2*MvKoL(VjU}Gf%_smEqYyQWWi{wRkPNt(oiewNk}cWwooy2()G`<=cvO=aO+p_',
        '@`0hlSA&T+Lyy`Sx?P_qg>8q24=Mc?n16zHt;oIaUXAb%wa7IBCZlbIX`8^`5#6ox!u*0lem~!d0>7D4q~FBE3DdeVK_1xgC9Hk@',
        '2Cn^d^2F?^#y^0~V8T@p4(Qq+VH8|l(;AV?a=>3B)V>?=e$SO26Zj;vXf1FAqld!|WhO<W#l!W_fjo*-f7BI6oqDk+tCHKrLwEOx',
        '*Tq+VGQY>FIBCIR;l#4IxZ>%wp`hcR)n0BTTuOT0$$iC}9zp7n)EIX-$zy)<VH|tQDwyf=N;(It&~oX<PCB}TJM|5&-gd&@f$N|;',
        'yK57t^Nkbt6qdaMY@j!A?eLirhi8J*OhciVoFQ5f(Fhc$z{vm78${J@j<H!;v5sxkqD5hTbLRMkzsnJ=IG%Jht-QW0?OM|Dm4?{{',
        'r;Dz|Z80_38h5qm6C7J13ORBLe32EwEUyq4iX=MZu-8IBGD<FY2EsZgDr!#HBGK|Thq%nlIEOaakYo5s(bc5QQ>JW5yjFDDAlq=|',
        '@uX{MW%cE0SCfvfKor))i}WyV8Oc1*65~Tjdf4gHupVS^etwhR!t25_9Re?c0{JLD8X>k&7Y+fLu&%e4TlSAf3%44{tr4x14_PUm',
        'e64Wu0DQX9KNPVHrT=>4#NgABmGmfD$XupIAY5|yDUfGgE7KyQ($k_M)9L$@LNYT$CIx3?pz-sNmi&tLd_7#1w31jTkBby3oZ`TU',
        'q}5uqeQXo2le6ed&X$`=N#ATPJ(b4k<Soz0E*D#tFI^NKy=_6$U-gsg|7uD&m>IV!_fX{Is6*t`a~^cYRFE!qD$buBWPc&*xhOu{',
        'jUXrnvapaJ=?g_9B@(q%4!4PFZQ{Ac_ymcEde4c>Nih+d8}QIoNwQ(gTaF<)carnW69~mbMUZNUEd(F(?M3*GY>nzE+Zh<Rv#h7B',
        'hc}q<;KoFapMM$E`d#_t^6y$legEy~n~ekfBZ|A(#|#fy4UWS30wJ;&F`pj!PwU#ImOr@^d@*#_u3<n8`x@kJK^qf_JSfWX`Ic7w',
        ')m`;LpX+B!B(n#<=&vNczUZ&F^jF0POm#eb0L0XNdGv;qo#-eKM+7FBTHbF8zwF}aw9>x?@A?t4XPlo%HCu!u8os556)wqWX$UlF',
        'F0PRp(h^M^!r0SdOMLFAN&^eRiddl}!ok&(i1c#+tXTuVr|IdRLfM)%P<AdoCt~aEqM}<{A|kfjDk{3YHG-Bk9A3BK#EA{-jx<C?',
        '#3m%fMnuwgHUPEDSy`9i%MBag%PZO0SAg1v$|rlHqxU|kta`L3Dr(Oo_;-8zt-U#WZ>?{S4XMw{&928e(Q0}nGaT(A49U{P%y4N1',
        'BW!6wYnmi`%n4^-A&+H2)>I;rC{7>HHYX1c&~~TrWFln363zZfI?;kvGsJ5u8fG1djXOH4X=$^ijd)t^(rWLMi3umXsw!5RN%m%+',
        'S-9{_Ru+EGrlXTi`mC*PvYzQMy=8f;cTZyM5wE)CEA8D}?W(J5ypAqde66VH+Tz7m3k$DenMWZTdZqNKd}SvGK4(-!UlK*%h(uk^',
        'PgKg8^7&2Tp4~&lt?{L2QhIC`$Hf#HOe~68P-;u73%0#k;bc#nzol{=-B<r}Sdel!In)*W?mQ>;2&PxToS{oN=aJQnCF`LKlH{~B',
        'l?So<+zPi`SJ!M8muy$pT(@E$D=SY=D=Qzi*FDc2|F_W9b)mat)+~v&w>QSSfPF>3;9U5(?ivjGyuey)Wo>OGwgv~#PB?wK*h<0)',
        'pK5PEb&8$cB*kAO!&Q_AOEH%aF2W@kgaP0rWbpT!uG6Nu;uk#5<VyR$F@e+E+|XYa*xuX>q2deDlyuoNR)f19XDPr^QE-x*{f8AN',
        'g<nwCoI)3`KXtcv+uEHcy2|nwFDl&D`_=V>P4UOFdV<RxtJZ);bC+1mxNvCuHO+hXwSK))wXZBOroh_%%%T04_@_?|i~6J`tHivN',
        ')V%o2x11cxWN^*ve_}yi_PV&UY4552s@G@Ro05BSKPY>Z?H_sPp5U`31BI`pTh+e*eppCrIw^kP-37e`vo`&^XK*)b^X>uyAX#F+',
        '{s;2wqvicAgIv1c{RS75%$U&h?B<6>%kC7kou4P2$jq*ZonLK{X?eCgy@gMYXD!k-tA1^neDF=h$`>txv0I}3oag$LGoNo<tLjl5',
        'O8h&fMzWD!Mb;KX{Tq(5w#a@c$;)~bd08#>F)UB1b0nw9LNUuo<lV+1?}j`8(EK2$=}!{0v>4^$qY3f{eP%}%U9YJ4s<0w6CM)A?',
        'amm??+?cGYd(~-aRaNO}%k)}HN}HO>%NxOp%I@gs?#jXw(@giKlpczVJXD&x$8_3B+BB_ZSx#1Uwe(71Yir@+RjV+^zH|+}fzw6v',
        'sL#=&w;)I*;#G)v<x&?kG4C<5m&|^Y+l)cGmH9__|G!yh_Gg{WTXZHfJ0>XR!S1X>39&)*b8an+n3tIq8Id!7X>w{=d0O%kV3*qD',
        '?cJC*e~rknDx__PS6xELGDFcCaBOyLRDgd>40H)iNeK;Kuz+O1f6^NmU1aB^Kh;1}k^c7c^xs*cH|$!xc#qdi$;`Hb{C4zb)<Fk}',
        '#KBP_nJJiUKhw+0!QSU~W=>9~)BLex=GvrWWu@4JjTsvQbWN<SrkPk<!`qfKW>}&ZS}}!mX|!TQ3z{g3{R0f641o?yQ6iuNx=CK5',
        'gLYM{^{c=M&Jj9*XuY0Tk8|YhZH^H*(X9B5Jv#6kd-UBcb`trGrdp^s%w0xZ5ka9}QKY5E-0eHXIN=vv*ev~dEgH8N@ngcBiXtK;',
        'FTO;+$!G`!1Ck@|-B7XDFv`6~Y$;EAWYb@Cp7h+Ax3u6!^{QJ1xtGg>8pNYxobGy_aoae~Wocwci8DQJX=1}hPR&NEm1VKnC2;?>',
        '`CH$us(4V{`nc5IB5}K8Q`{0W*S>iHZL!J6a^iHxTln|(M;9C{!I8K|%?O!6e)QP+G4(T*5d~*KpFjLk8BEaBO4tnISs&t$u@)-I',
        '9p+jL-+~X~m9G4f>JC-rk+HqKv9X=q+hwfa-7`vFRirO-1*EG_i2Jh0wTd%rfrUT_Hg)gW^T~EQ=b@8r=EgpG!tAZhKG37qJS1s;',
        'b}g1F!odEGFc?xdh;84XjqzDDsaIMc_#{$u`ZsPpIhW{bKP_CqH^BbxJ({5IMwjObPu;17?u(KhJm`j*Y(w{8HX^&v3CFD0&`U{;',
        'tnCXzzYtdm266D;=p%Xv&`<wE+Pw#zen{;u@&4?|6GVA1S@c1q_11_Jt9dz_&{eD_%DDyP9B6>%dsGO8L1^VT2T-B4LwO5xg<u)`',
        '0iHptMOJDl!u55R#|qC0CIudNd1z0(HiMT$z8r|RR;>Rh(?HmP2sd;ySm8C4`Y*E38VE2zY@;XrOnS)JsL1cq(*<9w&pTf_qEo#t',
        'ucEGb)x0SBs3CoG56smzf`30;d2ey(;p*bjLoLfsm?MpdK)!P)XCLyNxTU-WQC<KyvRrl*$xM}$C06~R&7l4W5=8;s)_6u!S}HLv',
        '3$4D8n*V%%@}&qH=lyZXt-&74B12-uAXc3jwFfR7V+iPi2TctxY9upOSWb)CnG(Kcj?e+mwjBBA^K8urv(@_eY^OK6&^x7n8tSEc',
        '9J?LXy*l~$qM?UIl+VvySXCGlLhBgNvo^zrYI@g?rgkRHYDk?MTAdQVl+;KfkjLH02yh<(pU~l14Os5`WPU3#0}tUAZN8UDX<gmg',
        'SxbZ1HjcNA=girj5^E5r3+gNOsAw%OOxY}s_pq22U}ryj99qvDw0fFo^|;hMRcu*eWO4pTjze+v1jyA^3@p_3^wfE*|K)CJBE9Oo',
        'v>aT}2EVkx{U9KXdDUk_U&bBeR&)z}lE~&qA{&~rhsW5B<Q!QQXCv=p+jzcPI*z_|L25@29oO6fj|0ziR<~~`-Dib4F++O#Sb432',
        '6OrOt=5c4Y^f*mBot2&im196h3w#T>IY_SR;GHSH8lyOgR!(^}il^jPTd4Esld0`+1rHQ%gi9Z)0plgRcgNEls0}8JMYGUHJTFCb',
        'xzzWMuncUe3?_vm&oF=pFe%dMiS%de_wQ<**la{81_m>~lU&$rP#iDmKQ2=!pjiCzs6Hm~^wNZN;vqY8uPw7~t~ZNuY4WUCFK&)q',
        'ayqel#^SixqKU%dm<6RX<PpAyOa8kDt3OW?2D+sOIIO9SvUgU*`eAP1SLeAn)@P^*a2BhMcrvAC;vN|VSK4~oLM_FXgmR35Hll&x',
        'LcW_Tc_gjZ%4(^y#Ry$FFRgd6zXwR8*Um^iQ|jj~m@rGc_w1d;FZQgxmYm`--E(rlyl%VnS)OUOC{C)fU$#1{-9M)M%Si@iF7#Jg',
        'iJK!d8-h}`=5L7+s9PACPONS5?nw4oJS)1wz{JaE{sgminOc@nwzg50T7gAHeV)$IwxN?;ytsl$#i?d8SA)bwxa&%85UR95Q-@I+',
        'hO)2Jgb`cf<dxW=OV`~4TW*TXMQ({nZg+NC<;RYhEUa`}7>~bCwwjz4kdVuUzT3%$-g~%ppI?gqYzxU8zp!kdJ=+C(8f{uuiJsHv',
        '_=n~B+eY}$ogNoYdOoMKcfnoxSpwh-0T|u80Ya4A-v#)7iDlGz<^k)iII$B00a!+_l_oPGeY?R@zytJYSk(Y4kru_veqbgr+o%!P',
        'D;x0S;?O2S;=K0_oQf(gDJhPMii!rOql(c#{LwCa&7+i*M-36BM*;$lmW4MwPDy!OFaJh5pnhPc^<yWJ#A}4(wfDb)+fk8`QBftu',
        '#U;=|{;{aI*sknoK){jGhz3k!O}OG)J*LrsW{C^e-;qofNknndqipTN`-q#(=GcuVdNPvcR*Sm1l(gH;_sC7@4EnRu)M%f+cU(mb',
        '#+VMoV2r7B9=$~ogP}MdB%$b-L?MGSOuX~eDua-VJ)F6UO!76^m5?p3MVCuB5akAE<Q=qQp-d*bhWqdOp}14lK!)Kx2<}4egE*t;',
        'hz%xiluv*KgRneZ2`O533VMMs!qwOhxWcACu=9vZmANm!r>1tavo_#Q)39?K#H?%6hM~MBmg5>g$ybvdf!5?-J*F0L`NE!_Sm5^q',
        '3<I~oUj?v&OZ7$PfjDR<U9QONLKM1)*FB_%OTERqr|9a7f3K@An3A9Mons2wS)^VMI0B@jOrf-n&g{F&m@`#SCOrp=>OmnISqk!*',
        'Q;Z(&PzCaCh*RCtLPihH0ldxfUP^8OIS2e`x958*VQEO)$L8zwbaEsclM=T~vD&wA|0g{q$(dzCyB9Nuf?WH4Vr}vqJp;pMNB3Nj',
        'PNy$b$5cj1&FPD!DVXck9Bsygv#r1CV<L^=Y^$wBzqG+<<~JmKfugrk*=Haxs)*EuZC$o;c4<Z9yLYHeshHd7-^JQU8<y9a2e>mj',
        '?=Ivxc+7%Rus%|iOg>Xa&QgE`0BwNG6}kUr_;(T*28JcUzrnC%_!_=WrWr6Y9liu=>F^bNnGV$9i*%(`p!N7yuc)HgbvCw!16-S}',
        '3$Sf4hp;KIHp@H%l(vC1KTZaje~^yp;5gR+7vdy?^9fRJR%L`jMLHo-1UbNr-+VU1l(pF$Ro4|7|9N9cXNKi?|JJ{c^0jRi&W&oD',
        'IJ<0Cc}<(&iv3ZpA$1QIt$b7+xAUtE=d7n+?}%aUgtW(I2YFT=k2;GqXBJ&gH{om*u!>!M3Eagxu$!oHxO<0#$PP|%v;^{YKBE4E',
        '_`+;`x8q0cg<W;mvx={mHh#Cb4gR^)snNmEJT=a(JgjidxJ?ato6h|FB(eO*h8<7xi{737{W(0nb=-tn{sbl>V$=L&m!@+Ux)28;',
        'vNuc{%8$oVMr2)3#K--R300b`%OLU^8Xc$R@}?}0DBBqj-BVeAB5X#@pFMmwD<`OYTy0c(;hM1Zai$JwhZ8fqGD>ePby@QCNCKl4',
        '(;Q^$RJ*1L(VZoGLpPxeSPMs+5{z1T1l(UU7xD+wyslbZHFUxp%hILp6=AE|zW#osz|F<nUCnQXgGc`Sgz*Mz7W$O>>~68R{ynW~',
        '8!%=3j6lCIY_E}MjmM!i<^uXY{!X-soXbG@xl9Z`msy0Y`UZ;D#^MN3kB?#;K60KY#ltTY#*>$jugCDw4?x@RB^c%cr9j8O&L61w',
        'ulB$E#Pz#(eyjc9KYl4d#TEVoT33LWm0%-`#;=ud!wMJ=T46jrk=$z&f%aZ8(rNX<HA}7jEcd;*fkZbE`S+XGJ;=|$-_r6RzuRSb',
        'LVUH0Yc+nS)wSFIsjvU1U3>kTy1F-qwNs9!r5#PtPClBEaTL#((86^cprWV_MN<EUbEzYV{ZL#TQj9){@3{B~mZ#KJ;`jC-UB1KV',
        'B7P6qBS&mN-|2=cX?0H5TMzbn^$yYlzY&ePL}J1IQJ1=59X*t^hVNLp<8x&{cL!THcvMFZJPMw%b?-8n1M@q!k*)*u_5wwuicvhV',
        '0eG?gHki){`kv8qr<dm<ZeJrQkH_?YY7pk$582lx&OS;+J*J*hD#&}Q$!;PW<N<$1hs+OoaZUpsfquD*enBI_C^Qnq;Pf|SrbToc',
        'c%e9v9ZZ+gsi2GGx`sZ&sBz9y<8hx1c)zhh(bu5t-xz4$pi_o%b>ltq%XjTgD|YqWv};v)$8c4+#dz-6)bJ4ctk6K;vM3?FKwKrZ',
        '9I;^O%H{BaS*UT*q_H)wrt|Q=UPK>eM2IShO5Q(rlE_O&iukjryM4p)V;vj1x6F;0J9ln)cqq&FZ@X}z&A;`+g=hZZg@xh%bBl`P',
        'XBq{pCEHDnrY7PV6@PH!u3&IWFQy>x<K=Qt`~&2Ai}f{Qo@dd5c?%ZZZ%GK9RkUD!YWFlVFvoTlR@!)DW4KS^)UUsj8n4WHwi>JO',
        '#O5zNU+mEi-yorn>1G^F4sNlQ`i({!J)oV>l{yQ2V#%p0jAkd09b)jQ>cB?v?^D$Y^G{aAwMu$+50R{oUw(3ak6rnK$ij&R1<`S3',
        'cJfnIM_cLNoV$yhJrt*^!4A^54uNS$as3%BkfY+R7nDvZ-K8Gm*fVFt{j99}8-O~kQt<V%s;>*N=2Supx`~@Yb}&s14O}0fQ82Bc',
        'fn!mc2G)LU*@hWM%H7<`k7O8TxB1b&Ct)+ZJEze08!#HU?b!p|!02ym3+KSQkW|`42%|)S(UPX|7NLPpn`nGYrJ(_q`mP_0alJ3X',
        '2$Fyy#t46cJ$v@R9{3Z+2n<0IQhh2(_KZ`3bO`s_3sLs!{Ga^NfHoQ6EbjjpljKnq5n1WsUO8{xGLMzM0e-W6{r!BI29o=|9Qd)y',
        '6aDBv8x8Pfy?g?z0|Kf8ef;sC4j<R_G&djj^mLkJitk>}4#4mL`Jaum-O|(DX1k}Q;dQA;Z*5>0jy5D8JMpU=biUU}r-A~;lXQm+',
        '8rx`NZ9QX#wY3d>$JEi$)MVyN62*9+m~F589yT1a;X8};1awvlJPdqVfDbqcSApCX5DItVojryE%%AKZih7@->lI%^qO3n*(+;>8',
        'EZG4Z`#xdjw7~f&@doQ*oYb+o8J}pO*z2WIp9g2cLiPV}qLt9QY<`QC%I+Sk#^}o4gdV%HxQIeS!@{ry<#zIX*V#Vpts%|Y*pn>x',
        'Xj;Hbns4uyehm8$FWGBm5sHq+<0>2wqTooG>+#SRMQY3AatD_?=?OhR&wHkc&$RT!&=sJYPV31zecdzLL1K8e|BR|Nx+vb!G%&^`',
        'aZyxa*5;5eI>VZm+FF#s0ur;^+hgi3a5019-C>0K;MyoYTE_V(nX3G3oSZRRS<%5t8fOt3p7xWyr{$)GE(hHptvmPATb`M=Ruh@W',
        'ib5)Ej$Rb!JT)lJH7zGDF>71s#UtTul{S{enEG;-PftWHot58F%HSJH(Hqdx$pCFS8C<Fwwt-*C8}dI%tQmcRt4SFGBNbc~Hv-ON',
        '_poKP=w%3tA(%WI!zY88WY36U%|JdHKSVA^dE-P(cwmx=Icrv+I72SRBYjydkbp4Ykrvpc;$+pg%gi(>FoH`}_^)*Gbh%8<ou;9Y',
        '9`cc3;9S}xJTTeRj7ya%kmA1OiSRM7rH*}Yf^CaOPCJ%>26{NH@=wM%6Ip!<W!pF>;iKvE#ZXWVxKui<15u^&7@TfFGEzf>dCKHN',
        ';a%w#Fv%>cP^bF{JE?T8rX}mJS0&h8O0hP@PP2#Ll?_L}<%!~a+>x5~znzeCgvtheMVpDXKqzkyVT1w;l5B_nZwKYETWf%zabKCM',
        'mCBI%p^S5Cb^FX1zyIm1{7bl>@hdGWavYcDZe`%Ctu`iF%hJ;ol*(v3n_0F<e+t;6w215f?^D~sC&2~9;o-&j73*kW;fP<H@{q#9',
        'kkH)RP-Ayj6H|A0ECqa$eG;wml>gGp|6VgcIZ0#~VM@x;>#;`*mUPYUUYMQ}XOt3u@sr1&tiBN6!}{e;&&mfAj9jM+3~MSwR?mr8',
        'H_v%d+zcZd6K#`*irT~Ro5Q^48iZn72#4xy0ontsDdZqInk>0xL2;UocTf@fsyL;>(i_Ya%Am-Z6(F~>iRIkOYhzlhH~i~CbWii#',
        '?u0hqx{jIamNi_@j1F;j14AD^2HfuUruH?{TUIrFl^zr7;sMQ!119<eIkqjI8f4(>ZPQx5|BP|m_5wP&$g(iZKFHn3D1Jlnyc#33',
        '6|vhdetSK4OUWGPU}w>!`1K`GD~+Zt2lgXIs|psl%nb3IG<?iReuAs>99Lwcli&kTz%IsTTC#7M92CbL8;)0fHoh-J5!VTT5a@|n',
        'L+Q3p>9D@%oX*6zU7S()WN`-FLT|@4Ss+o=%V)u3%<qbrR(>7xyL8hDdX97~Ta)y*fin}2aze4fnfFJXkv=^~YfIhGsCW1A?GJ#O',
        'fpBhD#I*8J^me#;JBWlc7(tlSm2NCYJmkS+tPY84#l~1(__T98=Mc&tvBuz=ACa3L+>bz-a*(Nk0U2W3WoD|W9lV`1%yv^mLsw|R',
        'rN+F@TpRm!Kn>pTEB7p`nVuIG(=mZ_sJ;45vCGU=_w!eLR~fVULj0UoVDz%KFSh(}#L4<Otx>sw*!P9dHS}K2akMgaic{Eih%Yn5',
        '7Z*&|#C&GUUNR@xMtPR`F+Pc1+tXl>$?)i3HPzbAM)OSS!pgV#MIW!MIUVEB@EZP6ACq2~lj~nOo^5Q8+LmbUmUb{9^GI^>wW6S^',
        'XNTjZ`i-s26BpF_+mZbCPkJvifzp&eS41Gx9bEXwhw{vOH2GS?;EPeZ^K8o&w=b*N_Q*YO=1ebl7jqvCW<tdSJ>zv5IqP@TOTK&x',
        'so>xMTYE2K+(VAfEVbnGeS<YhLso&Z0Mb{dEd&+^H_*`?CETfU3+D!NHE<vKdls%es1jjDPPycpp0+%Jke@9njqQJP25|xW(P5jC',
        '7c~d!e*U?qiMHhkpyh6!u;%2+c`2b0W@bJAr>D>{kDBIR;p4He(mu3SS`z7?!Z!0pH0-69)P}^Dhr0PLZ%h8DD*+VRsMtjE{3QzL',
        'X=`zLdMNI(K+au+X2K}4e8n{Qdh|PZK?O{hK;M7&w~ug=y2|!VLvk;bp1X6VZeHy2Az&t*@xZOjw7q>uV5;%A?SNVFY2vN@B`NjG',
        'pp=r|Bf-32+eyVt8&MiWP=pDJ3<uEzRV-KoGXaBq)|$SGp@v4PG}yY4<y}8;_DuJpfbb;x=fVQv+-<fkEyGe)*a!;>uRZx@Us)o)',
        '8)1m-IX#EtNR5)8qzo34TuLa!qXY1ATae{;$#%YQp_{d-l|~1e3}MjnS(T>J8)^X$zQLB!gfbVSfAxc^|CfGHBSq%YBWa2K&xs3f',
        'qx7<a^S?6}yu-Xja>4#WjP+h^@NEUigA+eK_~8Lq?ZC37^|dgsp*!<()%ict@?+Ab7JRvM@mGtxo65@8?5v^NT53}OwJba1SnBo5',
        '{#_OmGOC=aPsK$YEzem|Rb7qmUm6LwFyA8Ig}agDH$wE2Qr36X^n@S-<Vkaw99WupJgot4NxqmY_C65aa~)RM(y^eSWg$!*x$sok',
        '{y!F5SR!YCZ6oj_(G&788<7U|&+GmBC6o^=`q#vwi%JSoGh!F_Bz5N>`)*!r{=CfIf~?+!-Aht4OX~`0UQ>c^V04(zyt;s$n>}ib',
        'giUi(*ZFy_N%xJ54NHhXlxHFj)5ho$&-OlLCE-E0-YLlr4f6DKaGmbezCFKUv#Vp{feNqXA?<!nZl1xjKxmHLG-F%yDe;T4(<R}f',
        'C$CN~FiZ5GYG-dYQ;hxX47i0IN_GKvWBff6?bW|=PLPU4Gp=n=KQiW|XTy^+quVf#bRFG>M*M%{k@@4l@hHDDgqg|tnbN`*sY!OC',
        'f$|6GVL4k;5^mvSIeqfv=_SSC)BR?PC);p-9+qA?-eA0gDK~w&=`=@j-#+pr7D!{qDDI(AW+nr&#RXN1K6mgSG)A<vu%WIy_3O5)',
        'Pd8=4!j_gwnm>Pe+Vqr`Z52gd6nt}_9KI`rb4k}cc^1t^a*p#cyz)~VR<~~%_{GafEi)%?U3cEqrBz*#Df8j#mX@WU%Q1YG;jFwB',
        '1zA0rVJ(p{heF|DqKn8IT}HY%T7lW>Biaa<%Y7YSZ+1Xt?9zgsgmbI9Z>&j%T`esYpgpP5)-R=Ce*CVu?xyrlGvHHs6hZGnS~lt<',
        'I29;)^d9MMpzC7kFvH8eV?#|<OJeAOva(b#(mqsgn!We*l-ANTza@SRSZDIk?D=T+T;Tf$=GlpEqxT>U86`gl_@D7JV?Ra!P|J~@',
        'dBEBXx?53xv!e18fiY&y*Qakd*F8PLJgZu-C&kwJ<ZZe~>JpPODa4)8e0KqJNwAQbW00b98AAOV&ZUk-_d`kLGDuOmtfM}L<tgrT',
        'LOimvL}nf7aHH^M!K~{`X~EHp@&t`b{YJ(uM{kE{ACk$VfVMw)K#?#w<mT@#vA6eiElZ$BJ1h>jx3_Z-omG9q4_#+1Gob%Q(KJ3=',
        '(FP(qWtH^GD$aScH}z)7^B1$r8E5WBN`UW1<fG*lNwl@JHSl6{^~fy}Jc%w50&k_x)DRtk?*9Ej=>XP^ahRtA4BmiIlBv0?*e-Ek',
        'RHV*4M;3@ueQ&F&Mmc|%KiT>&=DV0Jqu+9W#?N&^e52$~LBm4PWri|Lk+M@#A`!6oHa=|)Z%j;6FIsD2VPck-fs3PK@Hh!Gob!u}',
        'y1^N4;HGCP4vdVP5aqy*cQ%V!?&WU<e62_Ni~=l}5$lKdv|#2XvybP3{1A2p1FdcH{wy9(BvIa#qD``UJ){!yM--6+)zVNz>l;kj',
        'b>qv=hiYhBdur<1YK{z25hl+R2AR#to~F|HauqOA9XA3bjT&3G$!OWa(b~)(Ol@=3jj1eesP-6xF`AQAr@QKCPd1_>hlef;x(Wb2',
        ')Oz9c!Yc*=dJXf;60H!IrJP|5%d0?A9fjAQXr+~W_4ULCx{84C#Ss3Gkc6B$Ni=MNKLeEBNQ`aDTsZL~byUF*u%#^7@0)OtH_199',
        '3D**^rm!4E&+P}gGPrm$SZ^LGYg5ei!3xw@Hqbp5O96(%nTz6lqyCO#Ws?y;tQQFJ3H{*YGd5*ktk*hcAJa6;;9{HLV4EGbbHi-k',
        'C`qw&Vr#IK&1wT-h>xj{UtG-sktilVVAlLTWhXqjH{l?Cmwrs1y@8a(o|Xu>EEdr~+oysFVB+fK9N}wH4DXy7W8viJX%)0O#5LE}',
        'LJ~a5eF4~O6gx?49A9PE*u=ygWA(>7E^-WB9Wv3lYOGc`Ma!ZnKu2@k(Cly*5WbyEqRk`q<qtK6PagQBI&z$cMYy%t*T~SaTy@ki',
        'Eo;;1;Wm!plg3JPvv+a6<vE%Jh|Nv4XK+;}nobX}b(Uz1gcp?0LeWRz$37kxWr0^^J=8^-OOfaQP_fdVnJFAg45zB$Uo;og6Zkmk',
        'F0g|WL&hI@6O24&5+LItkK@D^lJO^@EgIiIQOf7Apz#CGVd2EQ-N=cSU*pPMPhUfpv4L@bndq8xa@rKj%;5-;hpHnUYCYA<niDw1',
        'UC-P^@N;rFT}3}q!r9V(cspH9cO!VX)Q6QZ3lKbmz&Fs>Wga*hm&3`Kgn(sfftnr(TS?&sKR-f3wrAx+>(K1?@F8@2LsMD5CQMC&',
        'f2=$MZ|vIv{9(X`MWgJZO~$Sp>2WyE{^-~D=-9p5+hx>7xM}LN9cC73j}L7F6K+}SJ4~5rqwh5;H{UzXSIHSih-VIdV2#ntGstSi',
        'QZ$proW*q|Vsj!1t0B;2_4GwzE#L;q&V1_&z#AxqKUJkx9QiErtAdUS_@LwDnNdf_%2y6u&nYE%)!1r;Hw)vHy_P<qLuoy7ACAaM',
        'L*AHyYfYvzUSBlO+wQUOPwA5lVld=r@QE1fE2nnAA3wuSfg;x-3&!~D7*&KXN6x2IIz*j5PdT8y0{dpglW+Q)NtA6p*ug3ysP{g~',
        'qfZ{W^NsVlbF?m44Bvct8%lGG4L6K8@;SC)$8k@GaeHjXWzI|X=5ThXalC^A?Z*$>qoHNHJTE%Wz<Rmfs2xz+&;fYu$q$nqBF=`V',
        '9!gF-xM0!Gvs^!!6Plee#$%@7x_?%2pjV5jiF318>Zygm&(q&SuHA@#vi|LyP0TzruZ5)l6T@2=L&})=>i;q&4T{7>Tvom}(Jash',
        '7P=Gh6HE?%WEui{gTo+nZ+2%KfDN!lyzH1Ao-y0<68^jij0dB$rP7bj#A!B|2J~Wgr`)@tuZ<_n=0LaX;?P{hELu1tuylB9SS>V*',
        'Me<po2zWWNIdu{x%6++pl3x_f=mTxSD)8b@=I_4sDDU8W0PA6`^dz(66?`gZ=$P)z3GCFqUcf}}O22pG80dlS027w$nVbT)+(tsV',
        'K%o~tY#s9p&OiarP+zFeDwhteBB{SvYV^Cy{)bM<>!$3KVfuRN@8G4{qKi4{_d`s+dQtH0f%v9wCvoQSq>f`w>r>~|%rS~zJ6kvf',
        'Y}`l3c9)f}buba`h_Wd@x1s@zm^Z81cfr2o;>L)aSnmafs;UfGBg7GBH|CUfV-F)vi>LvArHcI39$BhXOYX1aYsztD4ib1*UpLKV',
        'oXGRgilHYxmVHbcGOi?-*dwE)zascq1Yd;V^d1E~j=+CG@C^jMR{>u|;Lj2K9)a&uz%MG{j3<Hbm%}M50>^aUv1<|CS&H#G(hfOY',
        '#tFme&VITba=J`1fge=Bmk>C{zn#DjDd4A+a1{As_+bTIV`&HS7|)R+c~Zx)-xi^8P?Me~-E!=hLiXHMoTqq2*^TV)XyslhR@99n',
        'V!IK1J>9~fodjG|Y9luX>B>0DQPFEcVqe@wfwfR>Us_r$|GnNi3vGPAoTd(~-QRG_%;`AL8>q3%pcvlX2CL!H38UN81YilEr;eL5',
        ')h5C^AVA`gku!xiIXlEUz<=~ukA==6U&CWYvln_ZxqYjPdy4J9c;~v*&c--y>cSEeu$MbTTbIw*(3x&#A3A+po{q(Mb#-;U3A)pq',
        'JjbIw5$Q8jW2+GFC~q@R!>(>n!yIN--GZ%xze8!yFX_bMxAh)Ax+@o+gSRVxQyTN;Bj@h2GP!rDV58_OD1yMx$LUY26xpD>fQ4)N',
        'X!-^u-oj}*A_L;AkG3aok7-&zqaY(<g7tJh!(;byxFgIG>3!)T_8~S}Y4sDGoxzk(AEtg%wCs?fnW5+YR+p4Sr-)4fnV)Y-QA=3q',
        'Q{p~N;;E1mkMSoK;Yx?mTzZH?PH5L<>L^tbt&w|9mBAXR%_%1hf<?(oFJ*tZd;b^3;HvWi%ZSAJY3Z@_BjNJ2Pmad49NDup_Y(Zs',
        'BqnBR=t^&!5I-_=6iv|gQ2wKfSCS*Lnut4Gbk(t>(FV&gmniO#)YM|8NK0ot0UDzX;|$@NsdE<lMb$)@z}=y9h9*WK!Nl#6b`@<f',
        'zS1^a16a*!D)ElocrkX&&cgv2Q$6j8%#dZ%QsRR?>cNEroi!gm_l|wvd)I!q#L#?m<D%4y%Pn(kru)t<U2SNybH#!dcn|X4_t)-O',
        'Gj*+3mo9y<beyQg=unyIl%|}F^F?^Q`;rJh$S%MaeRYb+h^A~<3OwT6mEQrv7aIrwdf9;8Fa+#ouSD&Sf|ckETh<t68SR3f#R%3@',
        '3}`|zSEI=?gn`2>p~E;!P)1`mkkg<9Nra|`(+8`;chK-ibSmo4WHq(uVP>CK>gkw|HXUMRoWfPHZk*zl7wPUEIaXCiMWEw2OwCkZ',
        'WERX$a1CZWf+kt22J)*%*$7oc#+-l<=a}$W*3LBJ0+`^@(?=PaSm|0AA)aPPE5Kvfy!U4{Lq03$*FP&HCr5T6c!s!V<l`LCtQVr6',
        '@Dsq9d$c{b7|&NG1=EmMYN1SI@d!JgX$D_&nvgx>+9p=YfdE`N%fV9wUkG$G<7Fe_S||2Ttynp)seqwtRegq-84j9{U|=%%=`fH7',
        '9?3M7JYwnaF#CYgX1<{vIo>3jB$Z}mOvM!rJir4fp2AUxKq$@Ao1{N#I5(W{;WyerFmkS`r;Dop(D5QiEx;KwX^xGhfu74a4F~7B',
        'Y1;ZTrX+fqS&f)%U|=<G-gvYoqnWR1XPz!)2-edOtw<w!xgr03i`hr}@Qlz}t(AQRwj<qGKl~OM{}t`bMOKd1PKCOL;dCE@k7Bmc',
        'PCP9HZ>4lWIfkQ%ou(!;_dqqr=D(lYO-}IeNWd>YiII_&m64HzUEz_KNPy(KwKa}5|6m>HH5@bY<SOjMMC8z1{!9aH`7PZ)@|_&i',
        'XN?OA)7SP7(b2H7QuiF|HCtCZcn*8Q)JDT;g3om0sX|kAr?F1crWzx@4VZS&$ytVGeO2~}4EwPu%pUNHyPISpP&Nl^iR>3(gnrRx',
        '9Kj==7y0EC{1yGuO0nP=`bAB83$)3436`C2DiFj07EtSCk7yqFf}-9E(Mrr=expZmp38NNkI(P0|KMr$=dKb%(ZCL1X&5D#!2eLi',
        'hN=@4<AnSt6*Dj;Dh4PA2GZu7;W#n}@@-Cj^M!;qA6k@~TR1F!sDr2H%%Pm&nH?ROo{>==xL~}9bDH7vCdi+LA<>p-GqieZKyiux',
        'I?iceZ3LkayeJ>n!tlBOka0W_PsF!i`2Hou0j)`#Gk6rh@}drbF|<G3MRGJ0-hw4xUcC;+e0lYnOa^X%F-$i4jR*DuJrd*1M{h?t',
        ')A1>~o=~iYn{71I!O^|0^E;BgegwWW`}y=|(ueRcy7~=8G2Un_59|lc%Bz_x9BXp^G;l8szN1ZnOYwLPVo%HkhIMFLE=sc%@DToX',
        'ZIPC7L56RT?ocjg%0xH)($5zT(^w1tTKt`yuC@I<`@&$?R9j;|A(HK{6vc)jT0GE+yNKn~*&h&O8r~ZeW-7wnu7ikcn_JDK2Qfj6',
        'n#-JbE)?n)o9c@VhYe#fGZQ9{ALHWb;S9h=mSgX1Ce|9l-OIHoroWds%(Ui=$l2RASIL)emb*hV2whReTey0GbFz)`9Q~0_j`?4f',
        '%Ht{aN5m5s90HifB(y?N93A{T0q|jE0V|z$e+vi>sZj(4niESGUS2ZZe4L85sa8ox$40@ngHBr(dsWWVYOcL5qv%uWOBbfD^QYMZ',
        '3?1YfWVhXV(L%{A2Tk>3I!zl<1`Uw2V|5Ts9ymgsCc6{n;szTi@5t1{YH!b=of_W>!RV_pO6@mcXq}(0jCY8ykBP32chI@|)68(Y',
        'iE4RGO?het_F*r39rhn<kBMnNw%?)mC9~bHWM!c+x3)G{Sh%u;Jb~>uEF(mR2aZvdL`KnwvO8xq&RwvK#LDym%~}J6M<j*da>1Uc',
        '4L*K*!ejOXk5yG4J@y2D<WLo7U)F5A))DP-<F)zC@QZba;PD+TK%=n?u+<}mqj{PlYFY?44|E})h^hG_$O5$bxH3Cb(Kx1UrEEoG',
        'XpT|yUeETaNr@q&M};INPi^<uw{{Xx13!YIg~9&c!@uEA)zx4$81lV;@IqJ#$HSNS#1~=RM`YULL8!=XBPwc`3a}VM%RAms+NkJH',
        'qd9vgt=;F*J~cTpWYp-8#H6Y1o_m`)YVak(SBdF>Az*ZMHT((wjp=|Q@FO`_zKy7;Au1<5$X(WG<!m@&<ab46?MNJLuBD-^!)Oal',
        '7^LnZw37Y7Citf3v~fsj;EIhNEX{DZ%wC#hXrL26)jVbtYcyw;gO^?;olv>gwXWEE4Sf2U>ydLOy?jpgtTh;-x^Lu=$d2gXtjV4;',
        '(7IZp`KlwzJaCG-MQX^kvRBM}jybtEbgaDMJ9t+ae8)X~ze;+jYN(&YdiHE<YhQ5ES7Pn!Yi&K7V{Z4h&yMxN-z0uMeMO27<o*Ka',
        'LO-F;^;eS<N~MpM(C22nJ7J4^7+@EeM_tDMmmGW_ye5f_tWIpUnLE{Ynw{>RrrB%q4V;{6Nyje8V)IcJjr)iAd_4}|0R2ag&s_+z',
        ';mvIL75pj(-emUT|MAQ{xbe;%5PgqSYA8~>!v7bL_wyJFdDo7j%ArcL4o~4SSlOSf^#0<~v?M@l<bDl)#I}2b6B3n)!9Lb)-o~?c',
        '*jH~28Yid1t3Vk8J`E*x2l7N`5tkMIizZI!InQ>eb7NvgjmoMw4TRDsJKiU};|1VxXkX1=qT-GPM@9z@rX9C*QY0nM`@mx~FExs?',
        '5Q^Wg*R0;MDb9VA>V9=ir?tm&=Q=ie8T+-m<)2JZ<ck;RSn`}$#Ho@|u26{tnC?&;ep;ZNQ8~$~GdeoDG>x$G=`lY~-&x19^QML1',
        'JOSsB%y6{xPWxE&E|%vEipBsniW#iV?}zTP8<tV_aF@^8l%$5RN${MxX{3qDlpr@dZ~ghu?fuyOv%@1bz^LfnvZTZJbKz}xmF`v_',
        ';l3tzcG^@IQ+ZX0(}c1YJ(9Qmn=2(x8hkWs6f9cw=Q8i}^vk2&u+q0VW&Jm7ZXP^3<lyjx7zfYsQ6JWR_HF?d^Up1TPvP&!{(vv(',
        '?%qz6+_>4{Is{s0#+B7YS;QZd;yXK%aYEl;6MyEwhEVG`y(D5qX|}^$HUy-=CN%O33ebSNjW)y;UK9}oz4a%ZZb9G=u04s>lm{W&',
        '^1Vh&OM=!{qRkO~qd#qtx03`+u+pC53GA$C4zpAhR;aSyk8zn1lCB%8rebM6VTK9~ro}eTS5{;>DwK$BrWvVtpt1{p9#p{<$t$=N',
        'k6ue>P~Xab=T$H_(RBnuM%2)c(Dmohcvkit{z=bgw~_1uKU1#%d1QP`$p_lnAF%ZYuqavJf^QVhMr+eND^n=5t&5&0)!REQ3<tI5',
        'hT5a7*lmg@r@d>OA{x;2oa5rrKE~c@Y2G;4HI&d?W{jLGDKn9sO}H|mEZ^s9fUN<m_&0C()IiDFdNpwB6fzIC-(bcVOfv81kc8c{',
        'l??tN5)$K?A~z+`_|ZY;N)C-A5>nAxq^PoY(3{}Gep$$C(C2ZK42H;-$!pMZIHzeaoPMR$G!DyYCvY=a6oUWW53ha?{>LEv&v1<u',
        'xUmKw;Qw9$ANv9RE&@+te*XafVE7*t>7wkf;!i=EVDZ1!@|i(Tu>0TZ`dbD+>h6Cj^1~^E9*3u_%yTzji=f8AFXWYZqVdm3#UE)r',
        'wVC>1Fq{o$ZVkZc|Hs~Y2SjnakHa%NyL%@fy{U+FklsO>iim(B7DNR^v3C)b*kbQuH}=FDTZ~bom>6S>QDclPMiY&RvBxMHlbFN^',
        'xBEUbyLa5tG@tM9&zE6lXP=#&dFFYZo%+mk|04a0{2rkR%||ZRr70`&44%$b_-Xo{a!X@42{49pIt5<YT1gL6+H-%;s7hbiT1l^m',
        '8`57<(xZ*Ib)^iW(~ZexMsHnF-ZFive;05*%%7qhL~}S<|CX`}&-bEliKWsocFiOMmDPCl`M<S2Cs>gTsg+Q!JSY8>=fsOW=k(9T',
        'tF=BSYW|u0QgJ$ytW*3aRcpflzau<g@Hc5&OB(4Vk$fduGI$uDrhNWt8@yB4$KWH$BY-zCtbCU8&Gb9JM(bVd6q$Qf7j6$-I5p{P',
        'GCmW^!v@4jdjh*n?%a1#USLG=(fQILX}YwK_2_dje-w8wS(x}mNNx!2*^=73_rr6ntg&;u6_eWc-*t8LTZenQ8~5>zdp?4GZF#?r',
        '!-G?T75t#JO-5mKAICc)JX~2x(WA&#A6bj#<6C337V8%*7425iy+?zBvCR`w;`&Z$lRB+mTxw!$50Ap$KC|A*%sjEXd5+0QgnQ}J',
        'wMBH-fDXgrbIJ&px}amyl;N#g51ZC9Yi>J&%Nn{hacS1jWUcl_i=3d~KAjxhmbCNFp*Dr|J)tt%rN38bIyzt!j>0Ab-oc|6wq`Vi',
        'VF4v+)Yv0)q^0<u;Jjcx1<M-uEC3oh^*KIWC80+dF3^*$ZID_r?Q>yNrr6M_Mkm_f8xhjG8l|SBCbj_56!^^En3=g{LG#j5|Gd^!',
        'gp=N`q7DAum>y-AucFi>Q+nvv<;6Wz3W>}=P}b$uk?JX7BHb&mIZ~xlI8HSfg&hQ#--#lhtut9&|Ge_V&R#2aX1I9-g>n~l3v-Tc',
        '9f)OP?zA10JY~`wag9^Gny1-eul;i3+_mrbJ+e(5!M?$-X7*HI8tDbSGZl4JuiahI1&H?YqMPZw#yy;8BTb&4BQ5d0D&1b~{0+_?',
        'h9?fcu%lneq0cj<#-m3@gty1`IQPID=?AHN1n2t@r+qN)lJuL{#Qhx?OS5K4kwIY&e7BZ?y_zR2o*(&sK=J&JAyX#~n$^xlrH8g9',
        'U=;ITFwz)}y(w;C-%7vMg4>9Qu$s8pn)&2LhK7z^96m()>J>;fNpNgxSr+dX6R3=-g=c^%T?wd@QXu4d<cS3(Po#l7*QLiJI6pmk',
        ')>l3uw!<&(==a8<&+(b;mVpH|$Wun~3Jh}))R}<zF_a#v?(rIzR6m)-c}iUm8I=gsDIOjfRueTFGoPN3VYJ4E(re(RP(_t-D$CHD',
        '@z;e6knz7&jRB;2+ZfJGHMH2B6*wM9;l$t2V*8uG1N8ZbV}ysSr*Pwchx|(o`R5tKN%N}w#VHhyDulVF`J2Fv@)x(l__tKmR9Mab',
        'q_3Aqd#X*mR(iXtUTptqH0_O%nbZZ~i}Y}z$QVw7^yPAa%4vML5j%|^Y=Cbxfrlt<1KkVahOys%3^y{^P-8!z+pp|ohlIT;z4*px',
        'QX{=ocdDoVIdYfua;51^MlGApC;|7=23%mK2=`yAy={Q99|gSFA-kLIOR@cv`@(f1D9@!E%1$n43D4)Mz5HMJKEHuJmw!Z$fwrd=',
        'Z4b1Iz)XS78G6BUf?SzCEVW$hUh7OD3Wn=~$!uZzvSrJpO58}DDa5z&XimrB_P4Pw9g0W!+B?MZ8z52Tdxlv;@uBEO9fRl?JixaL',
        '9fteHV1YDZnCiIGd}|fcKLOJ(RZQ=Y42G$WB*Al{*g<2M)-!o9L6=S-j!<mBxMnhSXyW4+Mp3rDiklf`53M1gPBfY}__;cFWzz;<',
        '&kUoP1C<Vb2OHO{=WbVV1m?rn`9c-GUf~2_4K0PRgB%As2h-nji_!OkiblV043G47ROq^Zchxx>RwHk!2%yv?v^04@Lfgc?Tlxe9',
        'cz9*uLAcM2kJ@f14ekDD#9;1z;9$QAQ;R+s9n+@0l+dr>;E;@*;bo#Gc6PhLt2&!o+pPRRda`8yK}UCMx3JHI26=tL#!N~GoZPlx',
        '_T~6-Ug`6@=k?KR8Hw|u7F|$;upi`cX_fX7S!#MOTw$6`OXXEnaJig(!|!A84N@^sVZI`dx>fkTV9j<P%B9NyU!cMt^48=GOJA;A',
        '0PuzNYdf+01L+(XeNN)_`73*={NL<b9<BS{pqZs;Z1rS2J4fY{)!+>Qeh#;iJD9@x^GwrWtnO+x_;Ob6Ub5K)zSS5`S{dbE!bbOd',
        '$-b)e%08??6D$hu%V9<kp<hFZ(ZqPa{?07PqpeQUzr3;Vnwzt4%$&S`R{h(v6fYl;=6~@H);6e(Uag4v(qj|u4xY-Y9vMK<5gN$e',
        'ShKOIO^3r9SA9RGYp<2|w%R7xtRWZB+^Rw5#y^y$9iQBYY5Qlz_g@zUsALTE4^8oO>7%tObB%7Ayl8wzl|D*C#?^$%r`7aPt}=LT',
        'WtoXSN?U;#^ig)p;ktXOKFUH0SM^bL%i|fG>Z5$f;Coc~CI+YaD32NZeHDI94=24CT+v6_jWQTq(MMTk&_~%V=jn<K`Y6i``Y5~Q',
        '%?z&SqbxJ%qwJQyWpG6wWtl-AWjC_Y!+AS?8SB}`=;&AVp^;P`U^C22@<}zRJOKE#Qrc4bl)@`kE4%JoG`>_BW7u^smiyo(c(ZDE',
        '5n5O$+RCCOOBOE%k2fz_wrn|gmcs&}E#3h+C)1oEpR<+z&`LkjwdZWu3aLHSmGw1UBlYxjQZseDXjm+r@=du~fXTAyShX-}pZ^Ku',
        '`nFN7?xwlA3u>-*xJOmC(sn$q-i#G&|6uE~jc70LTeCkgMtQMG-x0m9&te1OM!iK=Mm5aI6r&Q{Ybn79eW&V0NWWH9qV_`Qm>r8{',
        'WD#9|^>*{3%RBBKUdj|JhV4tSEfrt&e6%Ye>DU{8G+*<k6enGqEWP|`eoU8~W=&!f+xi6!DQs``M)0t>X8u7w-rW+q&RMr~Pq_U-',
        '>mbL3aRsqGGt>P%d*>!(6%TV6lCU~3xJ`0!P)ckQuW(muk+s%<iExXpcIC)hKt2<Yx@(6k6n&`}cL(<&c)2uEuP;@-KS6vP`5M0D',
        'c&3|~Kx8rt#)@(J9Z_lujhRDJHKmO=o0UCE<2c-~t<oVDR%!X!zL7bRo|@97n`Sq0%C~jsG<sxKW2b>y8;x_v5FeiF&j}pNP?Idp',
        '#VLNBeR|Aj-P&Q{(p4*Ci(7by6$JM0JA6b)VTfm=@J5-TVdgQ-ESe=W^9~u@B;A)<8y_V#Y|ujd9$N4?lnTA(STqflq7|w=WhZuo',
        '*-5X5L60(P^wpumNnic$xuOK<?c?R?!E1~MFeY(##hM)3^<y1^I~zt&<yDSlWmngmFr5qciotG0S#iM&aEHTXICDWzY*vv=zY!z)',
        'xfEr^1}%_2EITZHxF9e-tH`x*<cPkm!lMb&t+(HnZcpG3ri6sFX%iBX^0F}v>7B5)Vl64k^z-c2yLUHFzYaw_KS&tWp}BWX@7_7y',
        '%`=NKYQBR5CSs3Go3Q6Zp?}tB*M3ln%jm2aDEmNJnRHOaXynNLO(~x9ar;AMxWoLAC7tu~I(Jtx(`T9z-C?-ZB<Y9Eo24HoSq*L9',
        'eyGhP?6Y|@_MK!s{AE$K#M8SL6m&%vXrOdlGfB9LtdK2yo6g7$zMc>ij*?Jol!Ds9Sk2uAV;gsj)hluL#ZjI@jHitacE{jNjBab3',
        'uuY7~aMK~eJt7LbJ8@5>+0t`tfyZJExJlSTdM=epFE-(*^<31g$~S-b0e8Tca69QkDJ*_anxZixg>Br(>o!)-jG?1V@KW$CZr2ae',
        'aOv#&7BmxV4jF2t7m)d1EW>}rhU=u*+c$1VA4|=p12_$Tk27^&GL4kMgP&~Nh+9Y}Hez-BD$TilL%Oz(t`rx^Blvj13z-2OT$xq}',
        '#u^(Z2luEL8($|I4McoxVlaWb7$<l@oML>7@B@7R&iTqd=Zii+&q?P$mlhRxFaGk&#oh<9v$IJOejqizBNePJD_f1X-1%IxU!9P!',
        '8kc{L$5hO9;qn)sIkQ-|&4q~B+1=Plh{N3XM0mB0^!ZtHx^$V7rG#s{#>aQ19^W*;AXH%VtTCR94wJKxuZC-jlvQ_CI|Evqz4X4i',
        '28NP?YG6Q$eeAxPmV}amYDqwf1BMpMSVMt6&xCt_W`WE^!`E1q!wFjDD(wXll|DE&wJ0fdc7W5oc_k0pJD5?Gkxp(NPL@5kk1HJQ',
        '9qZMNpO!u@#Vg3Mk<<I3nH@Gsk4Ne?D6VE_Hqs0@Jf=nCw;h`U4rPGDJRydzjVNMD*VpKty~a~HTgT4vFkB)zde)|FNZ$BkbG_|4',
        'gudOrpWC}pvu8$lH8pQw<!bl7bz^IDYrb#aF;gy0ocw2xZi6N1@Y1uxT)pk0d|G}X-Mn<`?s05=JT<B_rN$8`xfDtkxsE)&hjA0X',
        'hEj*iY_Dog@L56|=*OEwZ>6^4wlUCJBU^RzTQPeX6QL@OF-?H{Mq@K)t~$RlVd#$-k6Jpr&2(vPQlv+0h}~qHutbj*Q3p2fdV6E{',
        'v>m(4@apUF%g#w}OJ9}l+?9s48M#gS4jR<AX^#%ualx*irG0=E@RnZ%ScqI_W=mnT{D`a&LR7suQ>_w4tne7<OPxeS6%z1IhY$5V',
        'JSBY`wANi>+l*}e)+Yi9ZQCv^=!*2|unnD$bD7O{mJB;S(ckYKC%<+hS|s+g=3M=K8*LdQP5vYCV{8SIXDdJABcVhM*A;DMa-`~7',
        'fLtA6edZXD#JfTm)qV1IaF4dNMNH{J)R4QCOYxX5F<LZh@tti0?aIpL<8F8u&RzWG9O<_7__uD5+hfQr58f8Jobjl-&IK(^(Yb)q',
        'j<V8Cv?Zv0cSTzQo*!dhN;MsZ+ILLV-$@S?-v!CbcJ9oHv{v=(vH_g!;8($woy<3MZ*WZv<2{#J#*u>2MwR3BszKikJk)qI%4Fa-',
        '#pE}bc==N(rO>35+QyaoQtWY4qp{^}y!W-sxew*cf9-M{aH>%`AK~v`yPR?;=hT0w&vc`5-sEimOF1OsKa}&EQ8`xJwAU_&N<oET',
        '?Cf6G_}oMkR#zWDudXH#CWGi&B!VRWw-W7ttX|@e)kLG-5<xb|P?HktoFP;Rh)`peP*F`x>MfJ<Z~K2M(;l~~QReY#BJ-*;dAt8w',
        '=KD3u>{?BH>MpZBJqx<CCzvz1r7l)&b%+dYCf80l$bM@*pz54w)j>~#=?_v}`6pM~P}4ggKeE#xEbWtCV&TnVfA7%Yv92GL8HAr)',
        'uy8iRA4%teqWgFFsy~hh_6#WKQW<WlETrP`gV2NF5n4YUwTfqc9mrf8SDl{YNO5gA=~7JSTIc?mftF_}oL|KA05|9*khbC;&WCE^',
        '(6xN@G5<iXk)x!?E-d}?Dt!nt0PcU%>n{jzkpU<Y#i9hd1M5_awt|y`sIfO}FGh*HXrka?{BwN$4T=kce4DJbO8%HBF7z+ypT2lt',
        '=Z@<y4*FjDZMvzd19x-6go#HcPMm=4AoxJ9#n3L?qrLO6=^grKkv&@bp@oghT5ew(_raxM289Q_X(GG{ekg=&ZSxI*CqcBA!|Q1;',
        'r`O#8_&gK%p9s~b4FQDdCVt7_9i-<bS`W2o5CHra5?~BB(m1GHO9Ijx!<F><eI;ezMcuWe`$?ho?3wDfOqs55gocWyCC7J_rFqg^',
        'W$$!4#>XXP?=(4>ZJmz3yfZASWLOj2WYqdZzcC*xo2UVF6E#HLM2*a9p3e4AJHK5vVR}hfPC<O9=0<y2)%UPW_LgehL!xyth8ypV',
        'DR8us!cF$BjNt)>^IaNln0j6aE4u)k2l6u7J2FyHtaA$~!#=NzuV-p{jWeIpeswG9QLE&##?5M;?li2Ph&$M-{aN9neicPoE$7BE',
        'T~p=E0b!E@7-Fki^FBWKdEnsgwksQU8Z$jHC8tFr=g7=>w`QZ}b#!s`!`nxkp6fN^=k2{xhQvknYgtyfpk+il+wG_8`D`rUR*O!g',
        'Q?+C5#yXOtBNL;VhlIpO6=s^WfY(ENwFwChj!uppxG|pYw9=gcJe=*qF-vz`lwNUCA)b@9JF^nS-Qu{(o6f`xj7m<Kxjm~WEaQFr',
        'uDj=)xzcgRFy{&RxXI<lPGy6#CI^mbeR?+jy=*y9Ox=|MzS*-C&~PZNlqAATAegRIsY=33Ujpc*mZVtR<9t2Y1kXX?Z#Wiwv*-7n',
        'x-?_jwp{y?9!lAO((WtncevPuZo%0Ly0k2_9rn{F#Rr!3I=He&_TCowkFw=^DRw!y(JpS$`14ckbKmOGE<@Gq3Zj^i1fw02M7U2>',
        'C6}%G8P?ePN%rC_nEA2YN-upIs3=;vJGgtWwLgxe|BJ2SyoXCaeaBY*`t7_tl6+U`leuhM%9sNuq~lRGyP5<$B#tdem_0VVETCn)',
        'jZc)Dm6E@(?|;gl>#Te475Ue>c1Z18+KbbOAB-Df^s}?tv@q+ziA$|RLlcjvml*i#+7vi-Zgr3d>HxS3dx#&aLs8`;n0+m;WV%k0',
        '8NMP<s{i~@zY!fe3^!3^l)g{^28Bg7gBIp<$2H%w(LsG<gKz5?1qe@S>+)Z*|E|dsx6*!uThISwy1%cP%=(vgLYZkBX#x*;<sRHt',
        'W4Q6&Ssi<MTaDo+du_Gv<!u!j>UTkur}gd4vGfXjABCIXQ~RFaR%5u5{?&W?#(RMJ-7LJr^i(x{pY381wtH3hid_)jT4t*6tE#gm',
        '&Fn_~&mR-E`!)YH@~u8I{q7a{(%rLDuZ6GaOw{Jp`5MhX>&w@4<~g;#eD%9`ls|0ujwri!ui70l6jsYT&_H;T;j}5`jb?wZm=o1L',
        '@B8jmWw5=(*Da&=d7jN{lwmZxdu185&FiMtUj~ofkfKPEID%Q64xu(MakMR4(o}4u11lVDm0p?JkAgPV$;RD=cvSsJp07vMP7%(J',
        '{pD?+qP5Lug*Z6}&uYH5$mi`}VmAgmI|sYr6<<ZBr$<JnrA0b<csMz^yOT?I@qlg<yGcL4?;NaK9PIqQbSr0K4jypFIhZRBaT=(*',
        'z=cag;T`&3`XP5GM-O*+5wH%EqPQ0(SR2p<)P11}dUPXv*kM8XtHQo@V9nVv3^&I7@{Ui@I=@-;W|rT&BA<7DDU(Lv6=x~V6bHcB',
        '32-NuoI)tNcck?cUAOlsx_~`)r<GYR9oAjgV1%@jyN4qnjRe}_H<+O5jtKj$w_b>v{e^DE>2=QX3e=EI)stv%-xzAVfawaTX1x^S',
        '0E`GqLLBL9I6)sRXu9G0jsx=}cU81ovDG<lZuXKBmDp0Vsk=p9_pr>?mb+})q=yV`y=%pZfWpCk-a`gU-T6i*_UC5C4{Ia3;ug}E',
        'o$5VF+>hGcdAV`Hgt(-2^VZ$&8tSH7*Q0wvNq26ZZhZHWgsG!Qo6675{urO09*_TN=CUJoUgzYL;2x~)-f3RQEdPvY&_F&x^+3eg',
        '61DhqS`$W3b8uG$GiHTe(NXJ4zQa0dM;5Pa^c{RbPoZxpQhzhryy|Q?hEYs+<$SpA2{%JGh2N?_C0-d$b^a1aLv9)Lc_LmFvryS-',
        '<xb&CiVf)q2R2}J8*&VH>5!v)iI3VlNjkRg+Eup^>*NXW-fCz1i)7u%uULjYm7XTIm&d}$o?!%#s<Y~VOAo-KoZ>-#r6@FGd8Lug',
        'TmvBs@>&;l#88(kc1GCe^-iKg)*M&&X>@5VEAyB7&!krlr4;EZje?p@25i<h8cgdUb>@oL-m^xr)<NrHOY<9ocbAn(omD#m_M{D$',
        '#D5GF*dXiV5e5o;jJ%nx@z*-X4%y@U`h-`y$GAARUOB96!RR`H^u*uu^mOrNlqFBmQx+iR%I^r)HIgKN#<Uuz(eJR5wbhQKSI#L@',
        'e$SQJ?}D*NKQ{JI?QFq)e|ArQTW+Qp1oFH^eugRyEBkd>$ns#imd5bT^;*tS##}~lb<9<lMXkQNU0k3FfBA1Wm@QmIbz0iekF9uT',
        '20y0zK^bEKkLiAejbI7=mVbR`yh!!^hnsawDdrgKwUGU_P9Q=}7W@<({@~XE4k53FgUR?qssoYjHMLbpVy)<GGwE^JUJ<iO2DBT3',
        '^uM%1X>G3lx|33R)Y@Hxhcs&R*(X9_Wy+$yWnbT@*J_#6R^92*!%s2#KbG3!f7P*E&GiHNTj5)cmQV*TTei$(^tKiNbQMBs%em`w',
        'sa5C5{--Lw{QOs~*+4!|AfLmAd}<q4+TmiOF=Z)UPzOd(FZiC65m7yQMGiHWWG~PhO<1DG4r)obSLbQJp?aPh{;D%is4-b%$g|FI',
        'n;j0TF=o4{RK7;G_1mc9<Af2++Dlu0va&u^zwJ8Z#G}bcM<SmG)I<4ItW9?|I#F1i^LU#&s%&<?ifb47RqCkHNH3XwiG<D2`qs?1',
        '`q%Gn7<JppM$#bHm9dKPAME$PHEac?I>shktU|lAI@L@0IJFF&#&nUMVNbkjmQj1VQB{Rw)yp_mR~hwK6J(s?r?EAHh1?OtPVPC%',
        'DPf#Cu5p1DL|u@tm0yy6V~6hWxEb?I7^iOR(7(!4CgkN6z8RN-jY_G>y-?#~Y)x2Le>t=^f3ROArRBKSQ5)0t*`!J}`6+lB6{vGE',
        '<teZ5)VRc2Yi7z>)9Te^Z@op<Wf@X`HkF+z-N@z2E>&k!rMBw+#zroVw(a`;4S>%xf&W=;uSc=#Shd&lzwGk-4?8>mVJ}BL>%;2j',
        '$y(hcE+&{5BS-*2QA5wR+3y~A@{{vZ+pU>D&>^uG!{&pMu!Y<9?d#O)%Z<}+++DPz%s*p5T;s%OPqpaq_L))SeN9$&O7AdzS(Ii}',
        '-`YVmIa$Tlyla*&@Bw36a~qb##Gm`x{<O5{yB&36!3Dg}dp7GDQZ~GFcxi2zpt`JY8crq~!HrKQ)73bDR}k6sWU>j|Xm^X8&|BYB',
        'cD2a&Y;S_9YR1@A?Mf&nXrde=+_<ey*84Zy4Q|^ez~4WsUz&I7^)0DI;bR8kL3nuMu3mUqy6(~Nu>oa19pZX)ZD>)C#ZUcCno!E%',
        'Yx%cKcG7ylNJ6#bsqBbx?(DbJ*$$RQZy!|I%OTcndo{Ki^Wam(P-wJcpL8E@hnatsdB3AK7y3<hYFS;jLJfO#MsWR}n_@#$wY{m>',
        '6#bX$Nq<cY*JVAlMpIA~*16x5;8K)tupzpVUM-Hex(q38>_4I}GbGw6xli4DQO2f3fAZ?s%Kr9=vdqkkPSlcUZ5BxN5?;=oqNhXM',
        'O^sl@jy+WF-7#-^hb*6)UofqNVk_qLtjh#yRz^lvS??*0+qH7|CVON(CQp^qd{=?G(FUAeg;{l4^ZONQ)s3AgacRMIAZSd;x~#G4',
        '&wm@k_2<4ReHA!2nZY?dT)b)mCx(315T7dhs5+fl67fMXbk`l%X4ciTQL0tEj10aia+c_rU@-F9!KHUzQ<tgRW4lPt_v@{_8q1fc',
        'F<5@B5;cRZ*XjK=7G;sa!I6r|S5pZipD6}l)sx6#7SzO*oq?`qlops9W5m%agl!K{H215vXs_H1j$1aqIrb|p)NnhjgkbLJ@6y99',
        'fSdE^s|Mq<ozk+>%ib6r+9kyPQJX@%-V}$w0EgyG-fJtlR<hdZ(IF#hq7eEa?#?jm(6m>cYtyOkQbHaupj8c-YDD9T>bH=qz|E43',
        ';iQ2<g0d7i-;%-24Di29;O5B44z6ndRypw>3iHUOY*nbHD7i;P=(n?+^!4Y((I51KiWbm{DiWhR`jA@Y9e!hqK}v{^aTWE(d1uTh',
        'QyHP0O24AF`WsW5)nBE&Rgm>AYV^0CV)XYV8T?EoY;8t=jrlB0{Eu}bokoAqA7{eXpl9NK-OJq+0i|wAsjP0(2Gmuzy6g?BWA#uK',
        'uJF;uus?T2fpd2l+}6-?Of-SpRat_dmK&%A2-ZSm@^?sY6hUp)V$~w8YE{7Kw7-5qSNX{sFSaXd=RtopCM=XK2I8LshV^fqvuIn#',
        '%x#N$v>7})B>VKdgoFWQhX-%^?ArD2ALaUXyEyto>5sSCkDK{z(2k7OvwG&vX_c`PJNk^CIHYOw<lsh)f|Hv!9WiyRPo!H}qx7`g',
        'BST8JH+9|7%yZ~>oAOtd6yOt9jXKZF%$(V|k(Ju}kgrHzI!ExXVQQJ0llOLDBQ4x4sAW;vkd0++HVn&O+@?cATN}LNU#aJ%5oNJk',
        'yA0bqaPnLE#l16%TM7fpI;5AW61J0As`f0Zlfz*9AUZ}Nn3FLjxRI0siM<C0q)v(0uztul1CrnWAoZP%9%Cj22Mq3=L}_GHe`3k1',
        'J+NeikFB;WJkBT7v5c5o_>3qS$cVDXvQd{AnVB=XG_pkXS$SrbJOUWpl8-dC<RL_`G07!9!PJrm&uklW=G4{<wd7Hs&10q;MlBg_',
        '6hiq1rn+Im&n*9J{-LS0CM7*S$kIQr+jg?LruP;<7v?EnsjCL-$SOnr=h#`ygZvHC{P{oF>Vp%HtMXTDFe~TTRm&fF@Ev%ebYq*$',
        'j>eO$|IIkFW7nQOUT@B8+T+Dn|BH?0fu(6>ojVn_`fqlc4d?vniZQcc;jWHrjEr0S*!}3%0a%uPeg3DhJqvPHHC9n(uwvn<XkaP%',
        'GCKiYG+)lChNznsQ2Y(k?s9Qal)F|`FR0VohYcECWi@PO=ui6>DyRGv>(fFJaH$!y=2ye?v|*jn*BWu-OiWOBo#5)yqeFo7hh$yd',
        '5Orm2(U>yN=F%PMj^5teA!X&$*J#tqK6`xju7fAKmdzVTKCMi-bX85uZB(pbW8>ltA3nC&)qRYcYtbxU_cnui=45AgcgJPqy~-{l',
        'my~tP2utxSYoqka%nmlxbFtdm&J=vO^T?02E5<@ti)HK>^@qTW4P^5@ECwVuX_EMcug8?WHEd+(#U6S)f8C(B43@H!Q}EQBF=dfS',
        't&(e4%C`C9=kBWy>CI)ceoy|D+y?B}Z?g+kqvj-DJ;Xy89!|=TncLofS6=J!BLaQf<fpak@n-(0ck|vo9o{9maYCyutyT_l>fR$g',
        't1NTk{E3;v4vyOTAs5kRXp5G9QEv^Nt&|0=f?AfZQ5lAn596}zvy94ErIee^9jGeIs2o}>=Tg0_-B6Y*JyjWDsF9On_34#TroK>T',
        'L(Sga>uO)vs&SK66Mg(A_hZH5-)Z6Q$;~<?=u7`>ao@pZQE`ds1`Ir+?ZPMuDh`+bI}WX~v;H69P?tS)y_EwZ)U-E2bzb-lrgkRu',
        'x-WcdQ>zZu&wnayFFpUM{05_YE!)TrCVR!aQ;m(_X3`+Trf;>K+%n@$TcoU*QXM<JO$J=e2t$;qjRw2AZ8C69rgj+m<rvv$SXP@!',
        '1`=p&h{4D}gKk^xMi~@sOQr*;XeY9td8|)8(O5gt{$icVi8v=AC+I)uCeoczz@v{I58qd4NxB>3VT+sAiHLNSoBrR_Bn>-?+zn+X',
        'QMIPOmAirLfU7VLV`e8s`ddU^c8UQkyDAHm_~3@`_zCkJXG+r5czi1D9cqDcq2BT&cY_(39P|CV%{O+ZGlae;XXtLI<|b>g;h^eo',
        'v0u)s)3&~;p#;SQri0HbjU=QTD6Zc^LTz^R8N8SAIemEgb>%FOd{wus&U5<uC1%sJRYP8(L8WS_L2K|HYYkM*GPvKPt_U-YFt~qT',
        'e@`{J;C?UL*HJaFUQgqywzED%>eV9BTVkL(Ufe!ChExL!?)Muo+Rp9|=<lg!litz--3KSnss1<J2giXh2JWVZ^AAkm?APb$Sq}1C',
        ')12*7&qR+6>FN2r2PSak*XoArc9Q4HzBY>dH(Q^c!uba#@Ej&{!)g8icT-wR)gE*xHJ}{hXlv^jrVd`=3&2jza{`lc0-7h<yQWUa',
        'Tc0~A-K~8|%C6xjrRO}qhZi_<2uI;r7t*duBc;zib?ClhG55R9;?qCpMk^&L7{HiVKO(N0-_}+(mrAqGFZS}Cv(!CX`s7;8@-Hfd',
        'RQ~YU>9Xl#CXZ)$VBQ1|C>#0$@(CYDEw$TW9DzOY&)5!Id@4;^B#r0eD!(9!6{q1jrt9Xyw-B>Hs0kN=?Pyx~zjw9rEsVbTNIJbr',
        '`d3D)B}iR)jrdi5Nm^8#CJ~kA+1vi?*BGgte`@VG0^THq{?eV<lV{hZckqbsUcUSu50^IbaZ(9hpnIhI5wDVla$X!A^dU#7J=v%H',
        'j-eebnHlr<^;O6cgKuH0iqld@`}WiNI;L1+t*55<5D!0T3Sjab@vmqWH8t0QUxi6#|B-w!<vGib{uUw0Q-9G=nQt*T{p~^q7tzsf',
        'v~-x>@?OF|s$q;B-09DuM==NFf`4}hcZ~!2MEc~obOI-V7o}y_0Umi(T7h}#D8|PoOUH2X<n3>A7j;9p)w-}ZxotFGOQ6Yeevz85',
        'gQt>fR0NKo6pCQ^#!F8>lAd5Y>B*1MmD4)u3+d_==?@;~jK6tve8xm>jP5h8wQkYG2JBZldqR#M0-YMDzfZ}2;FWw>xt+__9VAh@',
        '?Q7P|BaPP1th`Oz^D%NcZ!VlfHf*F%f0m0{#b!TMP0e=`8%2po`FUO5yy(KBg_nv)UtGB8;^-r#b7w7DGHc#^(&6S)>A}Q_xbf4Q',
        'PqBSTiBv8*;wFEeJNLJA|M|6R>_jZLh<iu)h|`b(>^HkPG+utp&*wMSRkB7^Us8r5{QTLE*1Yv(-rVxH);yl`@yDxI9yz*lO&Qn8',
        '{)F_;oH@9`3HvXwcFr8>#oge`*RNj=zI@{b?XBcz<URgv)R6AKV+U(mgiVw+Er2M$x6868@7+Cq?3eq;KEHS5anXcHMZa<z_ew7-',
        'DzLa0i!WYC;rDRHw(a+}@1_*%EdRy5L)`Rv8h*W+<*COxBAk0?^Q_rhH_x8E<(+oP$?X=-#g`WD*|u%Z!uQ^NSNe2&|6aZNZ~uZ)',
        'CK~9_4(K3~e#a>P5E+D4!YI5KqQ3%U{3odQE+`gFMLW<%bOZf@{zNi1!*q!?7{}pM+yxilp|}L*y^HZ${4V~Hvw<<>7;Y-Jko%Ck',
        '#yulil0rI>d@`7fB~!^lvYKomd&v=Unq1}y%;CNHc)o<+&EMxG!AbBIB7~MghL9ul6-EjZg*n19VV`hOC>OP&w-_b16+4T)#3|xN',
        '@tXKdqtygxT5Be1=4ke7KGS@y`CfBZ+eF)5+g;mFJ4!oAJ6F3xyHWd|_U{G>4bmE9H<;XDQ-hz)9L<u=hMT=__Sj4^w=qvN?`=NK',
        'e6#sU^GoJGnm;suZZ2DxTR2&GT7+1{TeP#tw-{^jw#6xn`wfMLO&dlvY~3)kVb6yB8%}Dtyy2CGw;Dcf_|j6e9Bgo7EiYJkS+%h0',
        'W>su8)oPK|TC2TQH>@P<3D&c$k6M3ieci^)Cf=r_O&^=FHl;S7*?evDz_yufq;0P4DBEvrZ`uXh4YXTs_mSNLdtdvZ_T%k8u>ZvV',
        'wEboKn+^dEkq!wCa~;k*Ty?nb@TVhoba3qFIL&dR<8jAxP7R&fISqAs%jqkp%gz?gt(-?YPjFuCyvg~P^H0t%8*z<n8|61T+t{YD',
        'TjQw4{Tr`ne5mmkjc+x6-1wg+q={vdxF+433}`aC$($yunw)ELy@}4n)uk1<87>Q5&NbCGO>Ek>X{V<1njUL<w&~TTx0^n3MXq+P',
        'KCZE@tz0|1_H`ZW`hn~9W?s!gnhk0;wb{;Q-?&A(Ww`Zm8}GK-?V`J-d!~DT_agVn?#tY_yMN+-$^DkQ>@mV)g~zv^$(|iO3q6N=',
        'zUjHtbF1fjo=-jh@szz1yheGw;WZ1~3-1u`{@$hDCw;I_sLu$W?LK>bKJ$6u+tfGRx5Rgo?-}1qzAyZm`1$&U`z85h_~rN&_)YYi',
        '?{~!S@8;>v$28yA{0Dza|F-^}{d@Tj@gL_u(|?KodjB2%ANqe5fC4N60s~?LvIF`A91l1b=ouIi*e7s!;Gw|J1J4Iu54<1vd!R0;',
        'L6Ad`dr(kNi=ehaor4O3h6a@cT?wiPP72Nl9v{3n`0o%d#3{rxWJJh}kR>4-Lav0|4s8_b9U2yz7@8hBDs*k=me3DEkA|KKy$~jZ',
        'xrIfBwF~PORuDEe?9H$b!;XfF;g;cP;hn>0hc6C48h$GLS@??x&xnAC!4ab)Hb?A>I3002(mpaXa!6!p<o8icqr9VnqT-_3MioU(',
        'jw+4X5cNxRbaZlb$LQSXfzid$Q=%6{uZrFh{cnsAV-wRfrg=<MOzW7;n4U3%W8R3F8M7j0Ys`l+U&LIDxe@bA%%8DsVz<Qp+`_*_',
        'bc=B<j>Xx<b&1;$w=?cw-0`?`ao6JR#XXCwjMv87$GgP`#>d8|#FxZRiJud{Dt>!>S$su;c|zBO-U-7JP9$7N_%7j2!qbG8iDIH{',
        'qHCgmVsv71V#mbX#DR&$i4zk~CpAcNNODgKN@|hRHmP$`ucRSK<C3N)ElOIO^lsArq)(GBC*5q>pk-9cfi2gx{Hf)yE&pyQwGvz9',
        'wGMB6xb=%<ujHWQsN|&N)a0z>?#aEAHz(grevtgMO<0>2ZHn7WYBRIVmNvWF9B6Z-&51T=+k6Y|W}9Ez{FQ=JB2xyZY)HA(mT#Nb',
        'wxsR8wvST1QbSV{Qm3V!P5mzQPU_Rtm+iQAgWHX5H?3W1yEW~$wcFS3XuC7*E~Pb2^GNegi%v^U>zJ0C);I0#v|VZ6rEAio)05I?',
        'rY}rik-k3t>-35YO-8SbAsI_Dj%J+6xRP-zqden9d%nGOdzbcp?IYW_YTu!KkM{lB4{txV{p9wu+LyNfsQtYT)*TW$q;we9VMmAS',
        'ndX@TGe7T`+p)Og#Ez>vzSHqQ$0HrT=y<K;t*mBQzFF~EDOt0!7G)jDI+1lY>$|KwSx>WGb`m?;c5?0H-zmD&_)f<=o#}M5)1yv*',
        'cW%@<v2)+frJc(<-|qaR^S@n$E;d~fyA18Jq05CXKXiG}wNcm1u5-E`?|QE5wXXNNKI{4-J1u)k_Gj5&XMdl4H~Y8jf4X^h3+tBH',
        't!=lV-AcO6=(eESZ#gY;=H{%(*_iWQ&f%QTx_fmW*!^Jl<K547zt;U;_wpY8J#u@j>9MWHz8*(=oau3?XK>HFo;!OU?0LNBxt`a0',
        '-s}0SXJxK7*FM)RH!wFgw^i=W+$(wJd3Jei^YZeh=FQFfF5e<QIe%9E>HN$2H}fCm|D7-Oit4qf*Ea=D1<eYw3-Sxr6>Kj!Q1Da1',
        '{emY2e--F@2ltNdozy$EcgNnjy$AL#?meaVg5Ilof8D16IM+Ujee(Ov>a(=Z`aWOwc~+<?j4w<nEGgVnc&6|Y>(SiNskno&3B9cn',
        '{z62{y0~xN)Y0D*KsT>FI8y0bvB11gJBE$MIHfBjrKrCUi<<ItkiAfh#%d<Yzl$*DF^2tL;(R6A#dk&vg(x(hpNt9(*H!?Zh8FUr',
        'Xcu_^&zGS${%c$)K;HZ``BQ#4z|H||44TOALH${IhamrZXdxGb4&z_Z8o2I<>tZfQeuRIOzl7^mxE_G(&u~4;4@FVrA_^1)v`g?p',
        '2l#G)TNdPdh~g+e;xEgOM5v3%p@rHHAPvp`u2K%?jMj3OPyv4#4aeWhm+=j>*7&(F9q>Mi`uyjs1?15S@|ccB!hJ6P6z((70{%Ga',
        '%)6l!{xk~beNl7%2--pZg><2)lz$VAV`bJBuj{!p@M|!#*EB+jq5}#OcofHHBVYakvf=}fWsU3Cs5|e8()e67Lhrd5XcE1OC+Kg*',
        'oBpvhFVR5$J2aRR<V&2je4cuAO1?<!<YV+X>491aqtQ%Kj9fG!XfS(c3MwV(s5yR)_Hl<$EAaMQGqeGG16+<)!d!PDABuWWk1Ije',
        '3ega967A*!(K5UOH5JoQK3R>HKplthGf{6&2l@9z13-pK`Kdrtpdqe68~7y1qq?7pMsvTREPe-?&0C{F76+J+i{27jp;DnW<k26k',
        ';rF8{ti0zSYiCd*|0mG!isJc&;71`hcyArNpU)42vaQi{7N3g7fRBQBG=C8#0X&Swry>us6(vC33t{fjj6Z=&xe2HRAA;tRRcIji',
        'H-J}*$S0^2|A7_*{;P0>d=dN+@F(#jpr<8T1kW}=J%{r)z^9RDs)k1k$$MxCKOe2<dBA-bTEzJPZ>FODP;Lgu&<wyKhsCFYEW>mD',
        '36!-IaINmAqW8#Ev<cF$h4ho){<#>1Itf3bZuFk4L@UTPfX`RcQ$9XIDcn&s8@w;18$xNuIiZyx&!v1jsB0<eP0pb9<Se9Hiso}y',
        'AkB{`0k1=a;IqJo;x)(%4+I%1LyN==l*jEw6L@#jn}3M%@P7CXrlO8eemqP|GXS?_7N3erz|V$w4!;7e<*iULi%&&UNC#v~X&|nL',
        'Ja(WwJ`#BF4r;(}KyShG-h2;`cWaad>082myx0O2(KtEH>RiCjKr_|n$+R4H9fIap_hbX&xoF^HAjI3joCBfBs;4%IrZV_o)Q=>B',
        '9N&<8L;K-D7NdS}brIfW*R`Sy&yS#f^h!O*VV>TTEAR|F_o@7x`UYcBKjteGFO(}DWRrQCrw|6|r^u7Jlky3!Tz*9S<Z`k~F6Vxc',
        'CzBxg1jzx|a)8MIi;~O5-{74EsGr6HuC9>J3AhebaH6=fJm013;?xEr)n0&E7NRG~=<YjRYe07DGnk>%8K3;Ua;58fbVWm^7C5}Y',
        '2dHVR!PAql-IXiV`oM1xf({66@wZ$S>I>I@;oFBRdIFwa`$2m>1mJ#r0==TA<T13PZE*-+hC0LjTK$#WKy<}1k~~F=Ng+z(98gEW',
        '1HDISs~H8fNrQSHkjrVEq5bVgb^<Lb=t?|l1w1;(1*5@wUa<7r;kh-?Nza?<;Pc`AvEXaD*Yk{TUBx%zRQr|j@;;Q61^It2moq+I',
        '03QhPl?rbt4tgBfJqCXr_0tBzdtCwcu3XMw^C8_Lc+U~;ftQ+g)$XyvUm+jx>IgJg2>1>)q6fr<Q|kS31xEa<;9>(5%49<CKULC_',
        'O7QAEm6sZRNtGOlStv@lk9_{Coa*zT?{U_sn7um|o+J4QcN4Bd<tNa`^`oBa4}AsT!~0N~l@l85M58pfppSYF6_Gw@KFG%8x@3%B',
        'j{5SCQ5nRKfPbI=h{@VMav%DKw%C@+7_@6+0QR^xnWKFolObmqQ|RSqiu?xD@l&YdeWN<Q0d@Qo>UdvYM??C2`7!sQ{J1*rtIL|P',
        'j2X*S4OysmujePFk1-D^fBEq!Tv(#qQ(B5eknXWUS8+JBn~$Ndje+<4(U9sgX3XpQt_FGguQD(Sy#YSV%og=CKMn77LS4B6`6>4c',
        'N`Udg5O&`Q`Y8(Q1mC@K?W$hw)$0U$)wjpMr@ORGhH;*U9zWVP!+l=^ejsO5#%Nnj$Acz(ngw_@1{pjJ@uv`<B|p*2o*~^>7!yGo',
        '0(qR*w<9Y2oC-fjtZHvd>buwTzb-kb-nP`ZhyKINAMW?V{q36fv|kZjfCkYJA6WCgdRz6XwyS!3TU~ytw@s`c0$%ee5FZX6<bpp~',
        't(`icv^|vGdb#C00Dq0z3(CX)Dr+z=xC`@wyKsFB*T-zU1#<xXmCdhiA$O6W0P+I)ldq7Y-oseKifl%f0Pji1f-rXI3?1fp7*F)0',
        'ae&d?45W>$IEaw>=}O<quI7c+)-mY{C(qBtOKD##L<no?`A_9azDEN&TjR9TkQ<q%MA_8P4gSz^<R-~V8QL|1z1mw->pTN#62!B;',
        'O8UD1v=1k&e0wN!3tUSfOA%S1DM%I@ingMI*hFk5#)*kyd$EU@Cyo}!iW9`i;uLYZI7gf>E)!RXtHd?pI&rhOO?+ScKs+QK5s!&q',
        'h-bx1;#Kjc_^bG@o0XfrTT?d=H=l~m6+J6PRJ>X7R>j7OcPl=q*k5tH;>U`gEACeOS|RIN>i*DGN@JyE(gx|S^jxZxWLai7ZxSs;',
        'YZV)Jz^0YhLCh8V7-2J8oX4<vi(#_~u-T<zb5uMoo)*6rFN;3{HqYG5-R#_)88)3Naw-N^Ostq!@eW|KqhephXBFR8+^o1$@vx#&',
        'm&C9sl1ilI(%aHK=`R%<RFPGYS<${CwW3W$az(3(bbxdMp9{HVQxEYR$UBR|S=bqTHzf`+82eNVtr$>|r|06WMYpzVED_R(;!E+F',
        '_)xqn-U0u;cuG7eo)C|L9F*`U`A_&X7*!*f!NTkd#<3V?3u)EoWz@D5XHj*}d8t=77P=~4fzW51pL!R_C2^^onjhBz)$zy08O5W)',
        't8d^|=o{>fHRu>_iq_#k+yuLzZP*Q!p{Ljtw<LwQA?}D%(a-2QZh<4vX%M?ZI2^q|7ttf^iQa|!IKvpz6}TOSBH&AjMaj$_a(mPf',
        '^+bcv5Ht*pK%>wYG#$-Ev(a3%2<<^T(feo%?umXyOVAFq4;?|Dqi@jz^aTBeo}<6e-{>WhuogGM9ykbx;uM@sdXSzZm*l~iU^m)@',
        '-a}iFj4q=OaWeXt6rjC03>`&>u>*RH{>8aC7*(K8usiOGJD}yL5;w-p&;e|Z9Z@;9!?w6R_Ci+h)igj3@a5RT_vL~Dktd>>*R5c5',
        'oPd&02+Bc)C=c~Q15j_IgYn4}RDvd;NoYPAjUJ-4XenBTwxbnjGg^uEqf_Vv`U0Iqf1+pTck~CkghgzD8(=f+jLp%f*a`Q*-Ea=>',
        'PV#YE(hIM`tMMAV4zI@>@MgRPZ^hg34!jfZ!h7&uybmA12XPtx2p_>m@iBZHe~wS!FYrlx2A{*{@pb$YzK0*+C-`^#H~t4#V3{MF',
        '1!v3Ia}Jy%=fpMPJUK7UoAc!&xG1hI*N#i)+H;v)XRa&PjqAzfalOzQv>L5KZ=nn5Jodx>H~=@tKG>J*!xeIUxk21cZWp(k`+&Q_',
        '-Q<4eZgID{JKR0)0rv~{kb6WTNi>Nev7`lwBk?4GB$Ae-6=_Y9NgI+v+A_N+w3Qo3TP&VhI%E6xERKQdd(}9HETIKd;smIFL5=gs',
        '18dc|fLyS@8W#bcs>DGeo$+AQ24t)VcvB1#HwX<zV?iRrQ7A}d6vVOtG61fWpA#WPG00>gq!|gYPH;C04Tb0Z*jug9IJgc5STRVg',
        '8}ehhjfLFD0A?ZJs8rJw<4|7|3~!Bs+(ICAA*2|~Bz^$g^@sanfS1%Lvw1Cebb)sUL5{=OTM^)b0sBz6#sdzDi?8LqplWZ{R>SHk',
        '2C$k90leu`<Fa0ze<};<+ivik4^e3`M(vs4Pv-+Quk8;b3@x!0>$&u89S1zZC=Gl%%7DA}zzvMLg5v;}^$_2TK7{ZPIt<~*=ragu',
        'OM%f9bPdv6M?XOPM|2CqJJ4!i^ba@=y}(c!Cb%JlR?rq;XhR|)jD{8u<JPz};&3uZ5XNcHK456$Izre9=R?>F_dx^~;(ieC50b^<',
        'A$SGESK^%j--UNUd^dg%;(PGd5I={{LHs;tk1%|94v2$qFACx@9F)bi=2|1dC3ABjK98FRPv>)oAYR6mLHsZW_2NF_jzIhq?h}X~',
        '<xWHVO9J?iERusb(w+2$cz-gGQV?inhH~M%5mHh+WFxo2;=;Ztyx)Z4;V5}f@ql3{d3fR2k<bRy@dDk8<r!yq_<HmZ0`xJy=NNCq',
        'enn%7kk7CI#UoK53&U9$JEE|77-}_Q*oa}M152s$l7$+^L*>dtTWW^P8E-k%5WZ+Dkdc#iv+Q=-eP_?x+uM8DPqbfYzs>%j{Ym>P',
        '_V?60wBXGEA1@{c?ZES_UA8dT+HVYPGworN)|uMf*Y`C_JKf5t&&h#ahql<gK~DQI$<_Bjqd`iQ)_aPfFPe`QLVvUl9Yd!<mVSn6',
        'HN+9n(}8T@j--GTLQS;b&4ez1*KtO(JUkJX*6Eky<=j!c2_!^yz|j-<BW^LbfjdCswfZyMWuhe>(2{l`W64BPO16+a%snD6cw2s_',
        '!3Bb!Vsx)}d-=Z&?j-n1!A!6g9ED^d6g)V3ZWcTTwAC<9Spbi$kTuZQ4yf#?Qn{H4ooV|?DVqjVh4~E12g()zZTkRq`vP?bGWre$',
        '3J+)PuPI%3Lr?#?+9#fZul5P_?6hAOp=YPPqB-=A&M@ZiracbyKFy*3355Q>C2oaNSbv{K@<}hC`V9Fz_LRTCDe`Zmr~EV2V}|@C',
        'Nb-B~Z|HsbId+jBK+R^zPth#-FM!XGpP_GIUeQy21nobT;X9I#BN46|xHf>R8C)&k+7PZ*aJ7c3EnMy3>Ht?qxF$<4pjP&1iF^|+',
        '1-}gZ3h-;eZw9{|{66_H+7G@=c0@<u`Z@R$;J*NW68tIf=iu1``7zdlZvfs5yg7Ia@Xp{}Wk>7**8sQ%$q#Tg%wD^JhZzp;4!#G>',
        'bb81SNgnuo@V(@RP{$?^M#>MMj-SKV(?tFg?tVu@!OxJh0lEz+G8E`AL#AyfuTt%I`t{2jaSD8&4Ozdd0dL0cEI}G9K^iPU8Z6N$',
        '`6`+&-$OIyax_=|3C#z;5WI+r<X<7D$3W}f(LDJ+S^&Njya4a)gLj@l`b&`h5~RPx(*MCqro5R4JX!$06ucHN$wVvV#ptp05h|Dc',
        'u#>zQC~ye3lCMDxzN8ofUObHU>|n%b1EW)Wcz%b~_g8>g0@MPaRsekrwV}|V;H@C%O>lP}a{nBygEzLx7u8&B(FgLkARUj;V|g!>',
        'dLBwWuI79VB<vBR?GvEpW5D4);GYZ>7z4fp`~>imz)t~BpFD*pPvOZ^c=8mUJcTDuL1y%2?E^0D2Y&?o=ipC({{s9;@Tb5(kRP(L',
        'J>cq%tg#PxU+{k5n}hcU9|(LggPh7Cr*g=t9C9j$oXR1ma>%J1aw><M$|09>=#l(^3j%zL_V6{@!q?~s-=iaZk&f`qKY{Q2FYtdO',
        'H;~6^Adk~P9;bmkP6K(I2J$!!hssNFYk4W@39UJ1A%U;g3S=f6?rflhma-f0W|Vvl!M6<MHUMu1-W<FIcxS+ahi_sQ$o?#l?^z(*',
        'v*3+ysmy6R(RQ>!q<J)QM{Xu#1j9U8z6+99^=E@#>yK(ls<Hd>Rp6l#|CL(IF-o;g{;L|~FYwQ4NaOMa`8Rp0yoEhdM=CNsp)^lE',
        'DSsk=$?kWl@SkeI;&-F?vnt%=@#z2bC$Fgk{zM6(EmZCwsO$VSQ$DNx-MaBNfq(j%c>YyeUh?l$=k>MT|FaJGAN8eIN~#NY_GhK0',
        'f2+=tAJr~RufX?#d>dr!KD0aZ?ictfUgxdjhS+By?bjf@E^m<!%U{do(018CZ}gsgTE4@0){siE@+HWB%0a+sTWVWG-bCvzzb)UB',
        'S3tN~-Y+kaKbAjYdp6ec2XZM$tjcrwLxl#w2Or?Nwd^b(m96WgSFQA+HpSTP3;HiyQr?h1Gp;v-T+>TsP1$^HtR7Q|hwurbQ#rKR',
        '%3s6R=HD;u{l_Z*ZokG$mh$C#^St{X@Gq|l|EKytG0NqdaVuKw&st4C`+qFwuX?{fDv2d5js813UG;q!r<aZz-i+0M0W7>icP8aO',
        '<Vo^b>4|K^_9^u#e_&xtd9)nQVx3huwFY;Jas3O{LSHb!Y@f+p9pAlz9el%2pwIZ3$^zvtK)$XcJ)@s|(r|x;)%`Mjg@+)#uBVfH',
        'u#WeevGB&L`mp~~UvXdls@~@h|A#zt{?BN=#uusY#cKUVfTHpzCh)5;swk>=UgKQH%%=MMF_q18^mE9%W;>(UI#BKN$R%t&a0Ob6',
        'HlyuqU2{JwGnxUNgV9&b`A;M4YBB=~U~{0hF#pL_=0Btl)UF$wb92nmbxW!@rU7RKb*AfQX5g%#?iOtBO({SfNBaRbkG6%l9XKnX',
        'i37wP!C3)qoKaJtPh)uE0*;=6a)ol5fwN?5P6E)(6G9(w0?@8Ggn{4$wi1Tfx{)Q&FB!^D0cQ?$Oog1<fpZ3Wra>M(!C|0iE~Lz3',
        'v(S8SmO$NJ5HA2{2~_S4@jl=zf!c-achUMnd;mBMlphH3LExN#1EZjRMc|qNC&oazCE(0}BNHH;1kMaNGX=uw;F<u3W<vdDgQH{N',
        'xj>=$;4FY+3n5$t&IUNQ3@Ew+oCw@o3*lyPB5-j#ggd}#7&je(pJhOy&w-m-;O9ArKLN)BUtdBff#ZR{B7|COfe7%|8Nx=m5z+vk',
        'fx5tJf7A%L902h^7~3}jZU-R;+!nV*8sK^l)Brf2i|m2>y&&EP_-_pyD_{~JG6`tNBtT>m(2z-h$Rwa4lK_!PfFc7TlYoXy0z@VO',
        '4VeUpOadA*-ov~JoR&!d2lCJq;&fGzLy_RLjJvLkiyY&kmT{3|T+}iya*T^w#zl^CQOmf<F)nIRKWbTt@seY_)S@BaV15cti$;L6',
        'MkB#BW!yDq+;vvDOV?iQR35(p&Q9g@L~wR0zbAvUQ@K7BoShNxozN_B%~T1P1Fo?u2lK!=s?x9koTDlerQjTmBtyd_L&GFP!z4q)',
        'Btwx2%w(bwIsy(e$-qo98lm&(0#Nh;IE<cxb74|Kn3NDEC4@-{&!mKBQbL%N(0vbTDV55K1Cy0TOjaClD2DHitzcT>Y}^A0I2Y$4',
        'mpbHzwhnat&jRKHF3PGM?yp{#s@H8S4bNu%uJDyB&?D+qQm%9@lU?W4mYQ3jUT-Q_Vyj+rnQSU+eDr>3wfkaruT}F3hBi3~T91w_',
        '1tNph>q+Ix&r_~KfqE@fuOF$`OX~HJp-oir5G~cK8!KHYALgK9pwYbm1DCO;&ut=2pEvDqbT1<0sx`fLGh=ZcBnkt?)6qY85FUa{',
        '@gk0J0;l2DaA&x)+y(9m@gu<`i*zPkNjLQ!dTvdN|Hc(q$MM`WZWgzW_(Lr~G60^1&Y)-DvQbNL=`3x3JOH2P3b@PM9qu9VQSy<8',
        '!qp8A!=rIAegluk6YxYlA1}sB@kO>z^sib1-6LwrO@WeTaI=URu_U&{k@%Bv5<wEwy!NrafYyVSH3SdE!|@0_4wv8sP^&BWCjJd-',
        'rU48da=#Leh@>%bB_71bsCFFkhZ@+(8(=O&-AU$X>>7;M<^f!Bl*=}_A0CLu;IViiUWAw6W%xFJ$R%)zTnYD-`-A(N#HyHGpcLmu',
        'a@V-)+z(LyyDGd7T>SvgVGL7B1Bz{RI$Tn4r_@mCpwFMyK~D=>7h`ISWxJhK)SxtwS^Y;c9^7SAP--7rqyB54#<#&4*Z(A?0CJTt',
        'P*-RD1xA70N?lc4=z6Z1DlHhespq3VThrs=o8JWPAfzH_4{m~60vuWz2>Y`=7LG!h9)T^Zm9nx=*YYuIA-oLrR3#XnS4$&M8kOY&',
        'CZlV(b*fBX=B_Am2(s;O#D9Gmu4)<do@e!%#JvgeZmgt!tgL}d#;SNnDPhXL1wa{!&0?U=QlQU8#!1KxxLGCL6h{r?ETvpa;Ezfl',
        'z|tTwlv{Oho$`>@fpR~VB-E&jk6IU=eTNjwJNPcXhwpPbPG)1B|683d)mAHzHEf{1aotM(AJ;BWtsQOE%%KiLxT}z+6Qt?_>9R>T',
        'l0)SRPr{S&6T@2`MsJC<rra*>U#@~sN!8%LO=ML6PGh5YIP^YtQRT0?TnYxc)JwBIMX;KJj*ECoTet<fA&#2}(q+a~0zWJb@~(V4',
        'zt(~g9VzqR#>-o&V=~%bts#al$C-^(=P~$dyq@mV!#n1{WBQBTbXPD8pTu9IaQIHnqd0t#i$d{S43~(u0FPRsUGTlUh2Dej<wJCr',
        '?f%K!N2DW0Y){En)xM?Qh!>Qhlj)wNwe%d~_hh=?IYPdNSCdc@EqI8tfM>}hjdbL9Yi^NjO`yh26G(D3DVk2Ck7k<c)@ZhA_G!+L',
        'fn>OLC4Yewlkxm5?Ex}{%pwcOQnE^OOY=a>Ypux!vW4u_mWZp!UUHBe(eiu_pAXMwkmKYuIR|B226qkIO(-jn+#~mluCe41c_hpu',
        '&&c1Tl2r2a)63wsyd`fBuCeBV<_hn|`#`RNe7MkGSj@-rNqh>QqPfL)67C3h_zdL=_}tQ3gLBckXdX};_`dvLek4DZpU6++=kTTc',
        'a(<1rhqed5kAH{X#_#6$2@my7TOvFZSMg>1QT_yfhSr$B!v8>P%s<c`)1KtZ`9HKJ{0m+Zc)^T$YvGdMD7XMc#|wT!un;N439W=w',
        'p@UJZi_k+T0M{SfP@#y{TPP7GYhr~NnhZ^bFb~38!eTxC!b)MK=7DCKcvDy>Y!bE$dxQhRN5V1Tr0|_^7Rvih_>rZ9^ukl-6#SLg',
        'Q{lPrQe35(rd_FBr`@F8uHB<Op#6yAuRRL`eo?d#ZAE9%RrC`5#ZWO?Oc0aBG_j+Y%`~e9io?ZXalAN1oFy(0m$IBXh#SN$;!bg|',
        'cu+ho9$|V_=fun6HSwl+Pkbak(^!gsYb-Ul#7YeZS1ov`rN$myV~{K#aP&SLT&yMuq$~yM+=;myO}?hDX0T?YW~^o+xM`Yl%^Yy0',
        ';Fg2i22%JAxNYEegWIPm(;U^D(43)Cq4|MQLi2~_1yHP<T_uo5GjP^gM=Awc53QdzQX33DPTNYGs_mfd0{kz~_SX*87HLbsP1erP',
        '&eJZgjb88tU6dEWxAUtUMMvPe9SCy=btw>kBty$hHtW_xxI%V-l%L78h3TN1%hI%k@PH1+8rV$-qf0!HhNwdK3B((*up0}%U}2za',
        '4o@#u!q>_j(hZ?-3C7Lbw-vuY+(m}|mb@ohLwuy9qgnzq1RNp&zE~Oq@V?Syh;OaD3Qyb0&>G^i(rgH)R&IoFQsoyAHmf)c;pR$j',
        'Nb`x#2Eq?%NIuuWNS@nQxdp;h8AhOFxopGAddbRykaVM^lTTR~MDr(eWIG7=(-2)^xVg$yU*T-!CKh(5^_7YNK8E6kMHZhT&4H))',
        'bbvGWl@8iO&QXT88Q+$vriG)f9Y7}N8bka#r3wB*_JdMJ>Ea=rrt@O7jb|aHEyrl<&%(Yeyuw0(QT2Nkp4UO?-1imqt#dCa-L6$W',
        'p?v`j$!dx-{)(2umC63_^bc7B@GX_sD1R%z2FTXR?<t&yxG@WJSa_0#<ZVWWwKT-R6lb!5QE#jaBYVz5c873f<wXeJsf76jewRuP',
        '?y4KdN~d&7Wp)2b_GT353@HcbTGF>DWw?)Z0T4Q~_+XtAy!9h(+ewOUAS<254^o*SsZ4T;SZ;@O!&nHVU^9kWC%Fye;6d>LJgMeN',
        '1nVQ#$|W?VOlONNbd*O;Wvc1YTyD&8qxA8Wn?aZ@4T7+z1ak-MBX?u)V8(YpwrALkN{jpyX!VQi3Xp!30_Y_xYY^jRnU3bLK}WS1',
        'f79)Oc$n^8CiBq8<2LLo_>EC7P=-#CY}CR0f$Y%HIr2T45>KQdI?BSkI;u%CfrUTQvapWPe+k`Ngs>^3xykbUNrh8ceaIjW7(R_@',
        '$gS0V0O8fjH4vVve2aZSl-gf2NE7-+;8Toi8`(F0Ru<WpUd+Nd41fAHeZb^drcX%&8Uoyo$tsl)7y8uzgr+$|YX@)||AUpaisk>3',
        'hTLXWmwQZp-eoupq*y`iXv#d6&sGMXPD6kLhFpL4<*uN0gi`vmuN(5m1E@UXXN&^#nXE2l@{r2%8K+Bwa4ZdhYm_qIRz6^D(IbGT',
        'uo6RA*n@=!b<1g6L32j`vh)><lGhpjKeAdo%P<SzvUTAQZqwNV`~&t4?q^cHpVj(e#cc>LSNzJjNxxYO7DlqLh=n&<oB0|0img}}',
        '!@@BvyiG$GS5Ye}IXIuLW-`qygjeEKFvePg*U?|r!SBK>XD5CS*)u&WXQofph`mWQ#=4?^YK%<jm~triBJd^Z>8=^z=Ygjd@>YUh',
        '2YwUy?cn!-KLDQYnjHgA*Y(eWzXbj}@IQjT1D>v!KLt-u@xEk(PK^KJuS58P{~f5&kkIzwC{J7IAfA3xg9w%F_Xum{c-oS8Co+wb',
        '^AKhd;1q60{-JV6PSG!jJYx_#6TU{iq55h(v}u?p=jazrXb!F<g!U=qa|qj#AEDedVH<=k1^R1FgZXUQCi4e@K6GV|uGGeYqbswW',
        'z|qw>j_Ha+k0jET+ehQVAn*l~b((z;ccoT3u#NaB#BG_yj)R&K!&{r_Tj+{#)Bss0(plpL;bQ|oC*o=Uk)Iqum60?ZLx-_3bPG0y',
        'j$>o!&1?+4lZ~NwvoZ7?Hinky7#hj{KhG)8@hX_pMQZ*gaikSVB^^i?VL9nR3P^u4loXK?GMV4mz_Eb`nL*}}#bhN}M>aLsPPUUh',
        '<N*1I93v;mS#pVdM}8!C$V2j!JSQ(3JU64$B9XV?ZOz7;jpv<tSKdnq=l%ImKAKP9lle5hBcILZ@_qP${BXXQAJ0$WXYmX8rTkLu',
        '8GaSNf!`t?;P>(e!Bai8gW`7n2)N_mPV?vZ%ltL|CV!8A#6RQz7LW0je5C+WT=66ESc8`hUJ6=pmf-BglR{%~Zs2@`Kp|X+6_SJ$',
        'Aw%dS<OumfUtzE?QWz^t6s8Gtgi>L-utw8SldQ=W-VwG5+cddm<AvSAKA}uFDx46`2p5Da!VkhN;ek*t{2{y$Bte22P7xo9ylAHF',
        'D_V<=qKoJu`ia3}q!?#5%WQ$!QnOWN8_c$dt;AHZgV_<Wi`YXf5c`Wm#UimpoGi`|=ZTBKtz@Sewu^hj1NwT1C&8U%CoH}be-!UP',
        'IZwss;!6$Eh#Cuxt;Sj7s`1kJYeK<AgG&J1)1U^~;BvwB(G1iK*A$zL*NivD#cZe9UbBN{M>JD3vos4dOEs%B8#G%qJ2iVX2Q^1D',
        '$2F%l=QNi!*EBaZ_cV_*&oqB)Dz%(etF_eHYa46bv_9HEZMZg8o1{(AW@tNUbF}%|zS_ask=n7^iP~w}IoeX~a_t)JJKAm9-P(QH',
        'GVM`tC%~Q2UeNx3?Okb<RmBnRs_wpT?i)rH*;H1QMK*y9ATxj<;DT(9h)76an2{~RfGpz1D!2e4iX$S6C?KLya1YTaXcR^Cpdx5c',
        'qk;>If&u~}OX{n4=gk{LW6sHs{K$D{>h$#O+qb)xudD8>?)RR2wxoTU_N9EjPTQMyDD7D2!gyFtG_6{AW>{Z7O~TADOFr$x3&XDR',
        '$qD;}zn4#bI4m3)UK`#NP6=<5&#bU0oEM5thL46%hO5FC!&kyL!necC;YZ=taC^8r+z(FTi;j2@FB7j2uNtorKReza-Zb7a-X?xQ',
        '{Ni}`c&~Wh_<;DJctLy=<f(rr;g{IbVvE@Ry8w3eyuStBMNnDF2i*qv7+&nw-e`Zsn(OAnXZGE(o;vs#nGP#yD-X|nqI*i{H9>1b',
        '!9u^*Q0yDGK<F+%N9uP6R_Oi!YwP_E5y7)=iqIoY{C~>!ec_oXeGQbY(rdJ4fER3~WlCQGWuvrEYwFt~L+)rGz1bq)P3jLj!m91W',
        'd#B}YrqCCh_@83ohlGx|yO6tOkHE_XYS@<@e2JI*1Vs>;HHCq+v2Xe!o%Q{QP@K$&y7;SrRz3rsJP?hu0(e0kf*SB%{9&cSD{dW3',
        'R+=ZYe}L@jZLxYnr@3dPrlJGqXW4P!11NiYgYfNPUsig)0hHIsnj~>VFhFSI01;V1AhYm+|E<t`_pOe3u#Q<!(z_UZ*9%-3c$E%5',
        '_V@t5N!B~oNXmD|#GmAD9(bA?dm5gVy-fMA9lnaaPipRPxk4X84`jee8tr=sZRaGWqenevI?w+G)QQgUArG1Ej`fFhl+UYGbJn6I',
        ')%N|QO?_8a%BwuMy2^ck-@Wq^#n98fG5jbCpK3KPk(!wfp8u2Xe8hTwpw13-2PoH8dY9HL_Z9G5P_Cu4pVBp2ztju2*{PJT)p_MQ',
        'NOpEmUp3)?l%I1itCX=ORS&)w{`}FMBXk-ly@9)Euio1P)V_jmS?s&e2kI8I!Mf!?7OndhJok46qDAz03~^tD0A6bWY(c23f;`+U',
        'R_UX@Y7G3m)r0dl``_WaEe}7it`~i9n|yDj(wkeO_R=wcCO?AJl;_wE@Xuo{;Wyfc5Dn=Zf3!oP^N<D4aER_FYPq#abFdRVw0E-T',
        'r#+P=n&5kaN|3UJ<~#_kvx}e`v>**}+!lWfo-njpBz3k@%=clx8dB3yJ>}a$QXWxNL;E?RB#V_ei@~u~YC%;!BAsepq7wSbw}wyP',
        'YO201N9^X|Q8!b1p-1d?k<z+$5@I?lGNLmbe1&tAp6!m}SxTRAtwh83#h~X$f^mq>V)!<FcdVb{!|l*RP_r`#70vJ_;XCc;phN7|',
        '!9!yQLWj*>_2j+=F4=Kb>_U8mtr(TsMWNlHx+IheR4a<$i*N(=NY(_1Q!WjvC`v2`{(#EdyC_zGmUlluj-Yl5Y|7pW{S-4sT+<5F',
        '9QW&#8|%PBcuU;O4N{C<q!_zEYk(vz{CkSGZ>jeK4o`89VuTJB)E^_Eg8^djzd^DzR}rQNdPs9YgOye^EqES&n4NAvk@9T6M(A81',
        'wjHIoeH7R(=q=bT{)^DqUcXS0T=2vd`9DLZ0V1X^JtWdxwdGvZoE_><?*u=oWrX(d5&PqNOYWdE8tPyv4+c#U75eo;&+<T|fqta)',
        'e>fN|HZ{#ZhL{WT<(R))vHeQ$S-JW|P$>23US@5VLW}GQC%z-~f$eZ-R}CJgj-j`X;ebO-`mL@D`A*QAXF5c1d0L*K<(aOT(8Z1w',
        '{qXi6(`^&F!flrNW$t@emJ^HPIu6=C)<KJ2$6fSy*F<XK7Qg>l7gj*msQ%omn)8y<qpC}P(DGc>r8n${iZhE8XFwN#M(^1(HP@l!',
        'J^L?3d%ybac@z;L5~>@0n#%|OBa#vPv@?#o;+Tgz9yiGdXR=1ew;WxWthrI&B(&A{c*wXQlSS?S!<gs)F$?0<(@jg$MxxJ)O?T7F',
        '^fd#_AX8vQnd{61Gu2EtcbGY5fmv*pn-%71^PE{@)*5*VR}T(>#$?sy@-ghsdOjxBOsDw0ut=<Iv=H_!>kPcGI>9_e<xSWN<_my5',
        'F}zu685@GTf=+mQ7Zab$Z1Te}caQHTcY1s)`*H9D+Qde}GW;RM(X-^a9b<S;+&P9lZ9}}Qt6U{`CaDos5mTpV@vqlZCk}2oQyCc>',
        ')buFut^F~vbhGOy^liwo>k|9GD2scR+6UHC_kqpSePA<nAJ`+wU0$c^_hcOH1xw`0rANE8$|2iWT2wTZ<jHhJPN!P<Hc##f^&YRS',
        'Go&5bpgx{bUnY5__*wC2&siW(K>y`+A6F9->(IAzEi<fRPA{Ps=5(lKx_9f*RcQAPy@lrV=q9vR&ZV_r7t&8gvY&WLPl<8jdiqJP',
        'pLi{;)V0ca%}3{%Tk>)X#+aFTdBaDV+4%)S2b;P1g(HTW`vwgy%r#2~jmph4j|?pwI>tOL7jw;XO4lfTy`XUHaPxM-h`a)`dHB`&',
        'L(NB0G0JQmf%@&E(A)0O=zss{;dvv?;nAa;HX}1eu2Ot#;m|xPm$HwplG229ywVI4oz5%cPvkS=|ME%`rvN0dBO%Mg4m!{?lqUK<',
        'CAm9~ROd81CCVQo{k_Y?)2l~XQNy&O0v;xD!rS(+tMA&mN$zF0*~cZ$#^0de%~+?j25Do$mf_s=S?MpPe_p0;nW-6LGG=G&PwwMm',
        '!K{lSy@*FzQCUWNhCHXHsVy2`&onTNO;f?$Ow-D=Hrb}FIp1_J9nFP;09{QFbERNFKf(V2W}wLzEEpoXP+$rLB}bVtW}LY}aB`yH',
        '<WzHuAmwy3Q?PQrS#8z{{%s{oWvL~#p^kL1_=x>&{{P^P!V+%XB=hHpS6;X^{}=hy;=jl-l75P_Rg28MGTxGm)#p+&R^QfL`P+40',
        'eqvAizLfp>sd=nbG?TS>w|-~pmDuS{eUkh36Pc_fv%gP9ZzJn0FE(98FsF{GZyE|xbTd6oA2U`~f$?TivXzV470a)r{m;oS(a+Dz',
        '4^OyWpHW(VBx9peI#Z%637vrCA@@`+{t^%WdmMDhU9;^YP6oo8O}a;GkM4@zlhUSIN}J8v=74!${y(5-urZ}&4b7OVsMhkHlsD??',
        'g`Lu!6eBiL)07>%L6nc&NwXV^Wm-e{wO%y;pXPJ&RG!YGcB1=TC`WfMmys*4m$Hg%wTj#mdU(B*?PN>4;#TNkeM;}GrT^mIlOsF6',
        'OtVGKk1#JSnJY3f!d-mtkuJW|$7Y-Pf*h5jR@9a*q08*mDI;kk+|V{Z+v0R-@tWCUJ|&`xlu23Cl`h9_ZK*G9{l()CqdOlrSMWMs',
        '&u{Steup>l`}_%i#$WL_{2hOftQq7HS+d@SHp7;;mF#J@I!&;(Z5?}#ZETy_RyNzVw-?w>^eV06{yd!C;au8GTX-yAPoMI5+D2c{',
        'S9~Mwq}{Zaui=Swh>oz~d--Pk`U+?8sr*O2jc@0wT#Xm;LVkcB<TH5*KP;Ynea_@od?y!a=I0erCevUVl9J6-NF!+ENpqTzH#7bx',
        'c}=tFE<L^Ctti83lb&JmPx^#Dqpf<D#dg}E=UIH8a-s$Dox&5&wW!WDxF*-)+NDmo(0r&AE^jw2cq4zvpYu-M!~6K4wKitcZCQJ&',
        'tzyrxHEF6n%ht0E?YTD7wzh5gA=`ysqxC$3Hu7NlfIi}J^fi4;`*}1^ro(iMndkE~PUCbg$AkG#JcCc?mE450Ql2|fo{LPZt-Sr@',
        '8CWziB?s#~+lH6f&a{@^;6i$r2h#iWAzw?oXb<h<Q9Ox$&>X*cJe6Y{=Tmqf&*JHP8b8A4F!CW=h*XB9NaWv1qF4^8Qut-QgX?Dt',
        'ms{vA|Ad{j)GXktg2IgyGY%T&O<KCVxP<J-sMNc-By&iUd7rDfbBUgoX$EjDS?e-oRqG?`Sdn7SLcyK|TDmV;x?4*chtVRGX{pd9',
        'QO*5Y^PtwC^}X5}*jG`kwU!`wBki?Fdz_=a-J|`?)qCgZ9rL5!mTEnEo2PkxMWs$`S&}00CKB7-Ot*1aF3*j*Ik)C)9>jm(75o%$',
        ';g5M6f5AI=m%Rk%I!N1zG*#qSPUP2w&*LmTX$7aOd?4D2^Hp%R3QkYK3KWyEH<dO0Yw9C$#0PrLhT<7bP%Ns9tT@r5Oxrebja`Iy',
        'iE9;TYJ>%W+I$mv){Q(~5MZUEOk=^4E=i2MA;L)9*_>_{-8;<J@;IKvlSTigiRQg4HWiIgw0n=}_dY%#w#8yyf!J4EEbJ7qu?k{k',
        '(DpNI4O>(6{VdV=dZP0UA$=V$&W%BCEdyst=A}DOOkyP}ev*{yn9J#iSfDhQnWS-h9-;G?C?mUBMs}MZ8!&Z);OQr7^T!1(>j_>q',
        'lsTR&&&swvZI10Fx>)k8tmLyQKq~aESpLsQMkNA0n4tLBN^tTWAaK;%tR&Z~zylW~%@izND9E}*a8;2VC~n)@_R$^ll6O?qxdLrB',
        '*g8qQDqdyFs=i=E7!6v<h!PTy##GXtH7_`|J=&(W(lwT*(5>8voAK2=j347C`5*j|z1Vic8hhg1$+?%<es}d18l^<4mxx&U+a5yK',
        '*~^5kw?KgnDJ{Bai=NsdM_csL79#%>@L;p*d~fkvSD01eN4+Vv6@*7M#3OE?JGa{KDp)bjO`$pbH&`*h0BhyduvUHvR?L5g_4BJj',
        'AJB0q2Jo|jM$o`)%9ksP>1v<_4HmkrG*a+i^%*Z}ziYJLm$lzlwBNPbFXUD!i8q6ir6I*7ocW@q7`2zR^LAYc@3Pq`enc5#xjU_5',
        '&8p>YIGriws$dIW#g|F_Z}}p=LatuQmz?x8r_Q?T;3uBP$xf3h<#RPRmFY2BNl(yI^bGxl{z}i&3$z+1i*Xn3%00OU=Ws9X!&h=&',
        '?uTElIa!}6W9lA_syD`E%)bDd`LWs',
    ),
    "bold": (
        'c-pMJcR*Cv@;H9aExQUzTb5N+WLfD*Tb894>Am;f!Gcsptbh#^HMUs5-ZgfOJsN8gjVAUO1!H-!B+*zBl|B1C=j<+E;(Nc(AHVQ+',
        '-JLymX3or<IWu$S-bVq7qI9Tf6hq~P1O^4itcd-QqJBX@;1v=ZlhCK)T{%TP@S-SJ(~yLu5rzfhc2QtpCPnG(j7e~CE=!i5q^O@>',
        'pbK&f#^hB9PrePHs5`L~#U3oltFAzBJ&L;jH~P0v$><5i=Fa6i(dXwBb=J7FD6dd7(_ThV7xC||r3j#B&%cQ9p%*o*bWBa%>R0>Y',
        'De96VMfGnRJ+>fk(5UcP6m`yuqIg%v<keL$&D1OOy$>31TAnwi$m+JgD@App@8J82vDGySYpM{@zlDFNXo^o!Tn4ur{i{V$Vg~&I',
        'F#uwQ9aN-vQv^@ZLx6}j?P>>=hk+Dc;&#7}rA=rjidI52qwQ>~`btYNUwtLhoCtm5;azYM7=X(P12BBq!-s6^hib3~tVR@*l+QT<',
        'ca}1v?9p`=mOP%s%)&zEGE6L$xmZZdctWv@>oBRak&#f}&`2aScXoA^%P`!~h<Pm3@;tL_#;qy97KFZOm_M@>-k1ycht_`hOk}7?',
        'i)+p^`t8f*)0|?x(#p=xzIwlQ^1|xzv#!-w-ruIkDQR%fzvUWd`=q!iZ!F@3M^XGBG<RLffTGNW>e&m$0{lWV;Rkg!>A>Kva2KP)',
        'yjJYg1%E=3zG69W8}RTUh=aSh-4Efy2e1re(!NL{*N_ebNFo}1DwznyB8<;m%J2kT-}QwXM!|D*81s$dRbOx(`fDqm{ifg1@$0W_',
        'Ye#1%eEAsMp_EFMUihxG1%>UAPb~yzfU8oek~+@>!D%(Tj50yviB$PX2+pYC7eUe|@a(6Ke*zDW+Mo9+Ec>Z5y|8F?V>w3s*n@@p',
        'vm48?aXnb9yRjUjZhw-OCqV0yb9zGU`sAFRAcN|`Ghc%TIDi~5zZ=gu4ITt%sBztRzR=)7@M7>qHy)>MJW<p+;L?p}pk`JeYUJQ<',
        'EV12KqNqzC0ex;nqmLsi(xHr~p~#}e`eJ8RXvpIu12$xVL|^KR>{@_K8~@@;?^0X^T@aWSl{Z$w7x0HGSN^R1yr3HX2|#T*xBKhw',
        '3OC!B9vXG^7JC-XK2ns{^(40vjv9+7p>@kjkbQlU5|8Oc34e5@<DdYXd<Z~%?%1r~K76@vcXCx^+E15q64=&N>tp7a88467v5wgR',
        'KTHkrRUGgQhi{{hB$AcS*$-%anjnuM)}bY4!)Tez#el3xk=VjQVycf;BErvn4{ZA`Q0_enUwyU&=ze+d!yFxWU2EDBc&%m4+-C4#',
        '(SwF<!2L$i4dAhT`hS*8p9f_nzdW6RX!t2#u<`6FY7nNuV<l#kJ|f}TO~On?f^h_5ZTQFIckpTBvsLRlCThb@wq(+?imA6i>yZb*',
        '5%l%g1E`kruj&)3;bzZW+mN1;p%dmG>B*Q_&0cUusX%Z)=!$6~_#idB80nWL`uiE_Pn!~IWNIK38qzEgs4JSg1?><0!PlpO-PWz}',
        '+Ucc>_8d9F?cND5TnV}YzuC#2?K1!4=a7!46vBpe<brh63!4$b#_NR*yoKdp-ZsE4UIZ08%Bxm1HL<PpyEZs(=$gmuP`GT_<h)@E',
        '5)9!ifpCVsiN09<ViUR?m@A4*+D~!v=~Hg^Gq~a*9E-*y+RSl6TQeU2=`+W=wlHam^WYa~(%H$j-oJkzzQw%8E1z*vr0Ic_1g!*(',
        'o<%IBFBF^5h+DZxhi1%pJiZvQZ9)Hj6W&<42@Ku?oPc*%1RJ+`#^m*G8#3CrEG?QF)rW5On<v}}oX&rC9ysm*04{;2H<#}lnc{FX',
        'IL(IrIe&AhWBS4*?A^%B_#Q|XrfP3B4|pIrnE~+xVl~{0cml#HKNZ2v)bI|F`UyPyshOX^gS?x|r?Bj&ZuY{W(~adAwXz2b@op+C',
        '$HIEB*mYw$Mm_%|FHfjzpPbVZH1bp2JOSn)6=?-&k9|*er6I(EBr<6)n-q4N5nAMQ4)kBLq}wvHy0B&LTuRT_OHb%FPe#5M77_b|',
        'RY>;Ag(?e0bcYhUhm<^JzS`Q{5Z%7W^sx01TVwuI9D~=GgNkv?h^LAz*w`kkEo~#RwBBQdOeGY+PR3u+L4WZJlp#Y?Vk^!;ev?I3',
        'hHMPzf6AYron7Tn<oaS6$C}OSTzw2aT(t}czC7`InjXBXKe_qy^}Du?DjxuUr{5{QvdRtI2s}D6I)TTgiMQvDo%P~j)}ue90|0#k',
        '(F;euB@_7;GYvfyT*zEpT|c6yhdeO~Dr9V?X~Wss8{an1ez|h{&9VL9)BZ)Ht6IQ-s1(63bQK&d4wp_>9PmG&m{D?ITyn<dF9P7U',
        'p*lOy<CzsIpRrQ5jj}-NO>Tr|K-40WtB4@6I2nkYjYJIb&p6Bh#}fnrR{jx{Bja}s3bG4`h5w<!&iS`9AAY{2Z1<?BapqTa#YrKt',
        '0lV)TODaby2GUQ!zKP}Vk3s92CT>m=tGxmzQF#TRFYf*WJLzy>3$H5<!fV`a#V$GyMQXK*IcOx&7hA3J3?E%H9b5*cuoF5VzuF3i',
        'g8L{+`wpuPU9%L8J%z4jRM&u0@H&qE2)^+MwhWEZ{I(2s;0TXxMRtv@qgRu4;HFxuls8!i=o)$wSqJW9H7ztot3ZijUwTKc(fgG$',
        'JX$#y!CSgVW4HjX8s&Bb-%8f1YWy)}I2qrJ;IqhDbyvf!$oQjp7QOI4Q8tnB%kT_(;eVpcMLQm#Y?RL#y!!N2Gc#02r($BD^krb%',
        '@N>(&yHmjM!qntq@D=m2OZT%MD{{{-ugji*wPU7y!PJvIy4UsSa;Y<&2E<h-YR<d>;zdnhCd2P~H|=2V(evxnIvs|n_CqTk@QH5Z',
        'IGSco!R9u24cN^9)u2N0ffmq_TVN*AZ_z>UNHMAF8m8HVu6EO0Eg@G+$<^23tR493xgZ~mhJN4=I!&<|UA&);RK#~dN^Kdnh~Ixv',
        '9ts~#gPp*%5DWxmFbnJiUrvGNk(C(C1f`0<6%#c(lntWHsdpTG)}|{J*x_C2gNNErU_0slomtj(5t!m#s4D?!;V*Q(BVV7gq!KAR',
        '>LATyk|;_I|L_q!nNri^O(wAxCBVrB!YBX|AkY^#f>~f3`kM`}nZR)u;W!g)t{<+l!{5h(RrH1rC)g2+Oqj2pGq+x)BQ+Ko!6FnF',
        'f=R#(eVhXCnZVOu!80b<u)7{GLq1e@(dk{=m;}YvE;T28EV(n(F!aTYM+7G#K1TJfj{qc@;TsCcNBU01njuIR#{I&+pW1lBCFbe8',
        '3y<OTx-pfjhrsLO*&kZRcG+^f+hTLeM(5W42maRO%52}zIKPr%RiApT{JzAt$U+ACoCID|e2KM8hvW6hT<LP;cB2U42eo#q)PB1A',
        '3bqP%Kq=NOxP(_N{a9hlwknM2$K5o}P=bHXOEn|<!=$}K@ih8J-D~;r6}&#3{b6xUmz>+ZIjQ&weA?y7Y}||^yg6t+hOn=2ZYYqV',
        'JWeXrM}gy<4?zV32f{4&RoAbTS@0Dgel!xM(_^Xolr^#vBtX8(ed24JHICC#<zJD5mEiS++&K;O{RSBcob`RfwG1=8{Il2Od*)cX',
        'm;^YwjE%Gi3kVObxv!;N!Rs4gEzt6F&aCqg8I<c94DlGQ8y*lo+7o%CEMjF@lqJ{(YAHvxmC?%Y*r&7(F9YT2D1u}VWt9XFE10-k',
        '<>rxaWI|nQL0nUO!pKEC24;?b+LUQ;<(%lvxuo?Q)OTiGX>@6V*$A<-xkGpsn&n&yfSK$|mCuyugPHHymml=l7wUM3laNd#TAf=w',
        '#hHjoe<Tx$QNyQGgS*G$TqITv|A2B48prOOJ<^^&-rE5V7<z5DtN~Cb*sAT8vu}q_4;_U6S=9~(96SU>tKbitmMz<WU-bK+yYS?W',
        '9l-By&^_R{V+TA5?}j{n_3ClRqqlDnC1145PGQ@7C}AxTN?23|+((qyDJ`$pta&@9<;|KkZ|1;`S#=W|r%xO|lg@VB1L2%GfZpS{',
        '7l65Qq4F2!%?A%|cHVUOFqR9M4_}VvnWm2Wt^zrN6Y9i!r>o)RB&Pc3`2Ey(pBRtg!Q!6L+TEj%QJp=b$zGxweQe^Va6dtg@l(7!',
        'K~znUma~-WI9KjB6kVH9w%yvW<V)lh8jXnLQa+NEp$HIRA)#3Uq8ZnA_~gqC)02+Q%Pd98`Eqwl=#lBkIbA6Z;7s@!q@H>U7!aJ-',
        't*77xFEfE~4^dTIeFs`ue{8R$6~0JaA<IT7bfhTWXAq|skj}p9%@I${k%d}wgd?MWb3`<_accbL`9aZ0i(96}ZJZYv+4aB}O!CQ?',
        '*a*4?!dYHv-5QOkJ|F&L6VYjO#o34}drBi|9U0QAb3wS`wR11sI&0*KZ1_5mb<1<0M*1MCVz4?(yF;QLL!2{WzvQe+ut`>pleEL|',
        '4ZKFncf4*y{!Izr0PTkAYMQTj$L*eUtL5t7^a*G;B`<HWI!lw1cnD*YbYtVGvowS);WF}&e=NOOt-OEvazS=Bm;ooUt)=^GwqB+!',
        ';gKZ2kdZitN`zh9aJ1rtB)&xw<%{^{y#KZ8hR+=T>Nxz))%{EO>uRjrFTsG1SKfE<a@xra@B04t)i*+E%}zzDPeYZbruyGk;M9tO',
        'bNk^NE1NZMTAJUiS&f$A{n>Ss&@!x>g_O4XW!RF>FLM87C938BlI*;q?kGc5d;mhA5gM}_KK~u*VCZ{b+FA!Z6}w-(88>bmJxKA0',
        '+ueDy{ftbpU57mj{U#Qa)?=&%2x}DCi7{5@zhE```5^27|G5dM2?d~!LUHawdU_UE3P-Z7dv~-fw}rO-m>uxD_^_}z<Y$AF&)E$~',
        'BiO=G)@o@_YodRl*hRgwh=}P(%+PPD$d0MAn>c>Py%}>}wXc0K1Kt>4Kdxr+yx;zU9e<3zx~&)tU)xYq0Vd0$0?ln}wojgLu(;;i',
        '#+-@8;bDb#p?CJ3xb9xNZ{n;9oJV;g`=ya?tO<Qf3yc3kp4W>!{d;!9&sS=`ZJ2kf9(ERFW+di~zjM3ma@PKZDL`CXof{6^Ji>;$',
        'w6_<nNKRQ->5~;}=i)!q@r$)<cH3nxD$YraL6i|i_7bV(YO*pJdkOl%>o5c*ZXZ0^v(L~dlti#jq;A_Y$_dVZ_h2hZ*}4*$9ruxD',
        'W}s24HTo8yfYMo=#2_7YYi0%zckX;KapH@eov?!jot-nPtGV6qRn50ejo;S5d(79+uQ@5TMYZCX$B1fw%7~&+LMv1wlt_dE4Z1>L',
        '+DH~$Bl-_5Ikt6k;m(R(H}5eVJdzdckEL6SwEF|%OF+DUCOpTHHp#!3O~7{khxO|f^MPH<g!<Vi67t#dsJ8ae7Ut_NpM^7LE+iT0',
        '3)UI&qOZb%_sCut^nn<@`uQjL0+^1>%_~8Xk=CW3e!4`zRcMuEXP4m^33)CYhv5p3<v%#UNvuBGw_nuHGg>cK0^85V1kWlJ0lO7(',
        'S*8Cu!F<_Ox@(4CRRnYSeJt{IWHzKCTCLv4_Tq}DibJxe)TrUfWPc-+Nkz0)4Sz~1qW{FRpW5~bJZRsGG3g$y)jj$c^;OSkvhS%z',
        'A1mmgY28ip81>gDX+EL;{v_{DsJ2h?{scIJBqU8-g>^yF<WW>#nw%jdfG`jaq9F&*!9FO6-@zuT@=j8jZAO~Irx`#@{zP;n&;brG',
        '1)hazzyWA!12-lK=b~LZU2t@XJS49}l^p?rKq^2XznEbVnaJ}m6fe&LYj^_w@Y{3v13Zqbr7!arexW$}SXUs)by0l7-BD#oA73w9',
        'vL7B?4219%?A*H(?KyuUnlBmMu2)Qd6jk3;Sd%;&SqZCD<?DEUK=nT`0T`ga6vXphxCuT+e;dGln1YrM3%V5c3NxCgcu(J>f5da3',
        '5splC{w@%xj#)60l#7=^6kGwC;I@vNa4)EY%Tb2b1STs^C=TEqi;kd2sISM~yN`+ooU$VcJEFf3D27*{DM|`U>1AE{ir;AiW|dmf',
        'v1&<UisTaci<iI+7NV(+LS(eSTZJ|(1)J#q;QgG?Aoeh;yQZh9?#B{)n5~A}6MImN*XkaRt8U%ni51RPS)nJ+%fr<Es4vMRWk4(w',
        '3jw?O!#4UBl5?;E<(xa%fzLL7xR2$9a($Mk%6oc^=5@Ky35xB^S;hX&&LxcK{%2heDYBbBXE`<+MMc9uI+1df_BxSr{pdtWb09_k',
        'BG9&-nP0Lwuj^eyRAN{H?5w;zck<avBcbA1c&X8Z>as{nALo$7h1uH<2SjCt`DCrkPTVpDI;B-v>z@v^`6@r$EeuJ6@W7+ctmmtx',
        '<c2fhf56cScEa;3VlUG9JRH+eSUE*!fIIw+y$D?3Ma?zbweD+<1>=zc%ICa!Yz)e$T_|5%?^RE<_dLzXJW;wMayAeO#U$sFi+DW7',
        'OjQF0Ju_$TP5Ij2_4NGHmldzq$44b)!Oog%^R88UuP?g(Qc>7<>h#j-9+58B))W->2UFs^p1W3<uPq&wU>F_jk$z~!sxLfZ)1zDk',
        '`O8wqxOf2g?w3`g=Q#~MWEICWG?om`s8U2ZiO*$ux`%v93Qwls8HFgHan{^fw4>Q$jU!*I)|*_VJ=G4w=L3{CN^y(?koM%R>hz7l',
        '{#p+BXm-CuZunv0_3Fhne&M(FS8mEEPs#>h@X*)Z!9R7iJ6T8h*aZ6q24|P2kII9$4drI3V@4jSYuJ{Uydq)8m8@kM#qB-;@VdMq',
        'bozg?AR9G2+r`;B(!n7vP0=r<#s?(nxMX2|Q<T54_fceEitJ2IG=O=PcSi_>S&^DoG@DPbYl+yUh2OuZ$P)*~B&@F7e6A?c>xRqQ',
        '5YKPndGF=$j)SF*=P)msT<)42nH}+qsofC4p4mk+e1t&_sgVsa8)nAMaJ0IfR5fmY;5R;TOr6;Xi$TNWCK9=q;!C;zpoc;e50n8P',
        'P@be?fEMBzM+qXnr6qcSt(!J&*nwU^YXf>mFN#4UH_>6>E*XiulK_q5)Z>+HZEYaDt!?Eg{NXNoD`->whp&Ur(RC8dbs8G@TJVMD',
        'nsEFkQD0QJ5?zQq(>~>4&>skh?W57TUwrWe5TG~w+t(;>c*Oq$|HjCG(1(%m2F?c|3wmrQRkOl?19g>3MQdp|eVa*A?LUBFBtQ!2',
        's@;1Y0^6&=hQ19hLnrLY@v0jMCbFj)cU%<!B6VFtRjaTh)+QqBh56%J(vrNmDWNtI)}Em^4lRtG5nvZ)<r%>Y1e1=G_yyY7l|<9W',
        '?gaST+Kfb42hsk_QguyBU@qqvDPu0;1bl{>atD6x1>6K*3xTsIJbtbH(3kXnpr5{b63BvW6ZfUVH|1a$e1v^iDD>oYa?O+{6@s|a',
        'VK8r)LAei&Ib(RL`ib0?9wsqkX+BMfRZCl9I*b;$8lY6g6s0OWvf>4JQoQS|OWFFWq{*g*8$Mej5Hy`1SKZ|0%r|iz;-BZvXitLA',
        '?_7e{w`%EwVBnH+z{eu^hb;IPoPR{SZ_{JoeFku#Xm=N*6&PueC$X`#scE1^#b?*%Mg=c;`|b9w<`K4vlR;sFO81uE1EKrBhsz(q',
        '#c-OX$&%rA*T4#3`8~)0^t`*U2F{x?H>3pI>3^<UkK6~1F91;<5Y4{=LA{X*1)7YAXorvETb1j^`4Fkv=&K~bdS@PO&1!MdfeK*i',
        '*qAi$DDq&EwD|DMfdjK5;#19NE8*IAt)YH3JW6p}?$i4BL1F5mG#^(O=IWia2wR&{*~H9JRmeHCr_<SrwZMwboy;5l?kcvIo$ybP',
        'uBo6)^g;R#Zuh$pyc760N4S)3;CbV-B93oQlLH+YIoNR}Y7wf>b@3k4;MUykpOBN2;NRZtmN6nZD=T@#rFP3XUzT0WsrsfW_hQ+X',
        'b1d4#o5x%%T<_qpq2S_}&A1YcC{(K`2rOL*W>ubVVJSDENnR)<`5}*&?w0A7mY0|2m+98h=AW3Io#@{Nm$qY!MeFVD*B2p-;x-FR',
        'Fc%Tb!2}V71#lN_$!jA%0>$t+nV>D_&4Xp&bMEZBQ#bj);hFY>yTAt3C;?73r23M6U>Pi(KcCn3x8Nqv7in)Al2jv+6kmO!fGd@a',
        '5CR&%P;{b2zkN4?F-G`igTLqk=03_KD1pk$3rR9%F}q<@#nh?gV;bmN7KKx%7FrZcnS#b6dtu(7-Cc~Ui>`9<r;mh4qHIJv#1;df',
        '3+ld|y!l>v`Q1%3zO4fACCEulOf@x4y_6AIwWny|-^+$9|9fHa@rJmnj75n<k`|^DyM#`d4zkPL>ht9Qr2^5<gD&6^`mRe+q_~2a',
        'wIunD<+R({ZaI6u;5+#TM)EziB2z{-tg*DW7IS2eS8B5ZI16&Rq~e<|l*ovw;D#8f@SnWLt9Z|7FFG@GZ(7>knP=KhV&=2cG|U&Q',
        '9%b<KiE!D&<{{0GmKk+)4!XUiXAX$w1oj>?9<3r0UkfFce2EMoH*No@edT>%&7<JcehU}sBIk>A)`W(1pjc6xWQZ=B3aOceey`6>',
        'C8423#UbHEnt$oHA$fTrA$hrq4<Wg^A)&cBn63p}4jQOeBwphq!zcC%8bEpSQuDx9rz$$bq5>C;yo~H46=e|*NqyRq3cz#?aIX++',
        'WPpoVIFqxW<%an1+6Sfm_DIz_G>{TjWNKrP^sbAfmY)G&(INm&j~#m&Di<w+$}{D4nd@%U*I!?knYr$Ief^Dfne?0G`&X<vc5Kax',
        '1Ix40@(K&{(lY7WtAXBkm6hMY7pqsp7Z<ClE&{#PGahcs$=UXB#*7Esva`26fPb~O-`r8V<7QiXUgG?*<0|GS;7Tw(hY4l8iC;j`',
        'DHEy)WW4w8MKX<3KH>t{m&k`niA?2|a`mZ#T%<A`KHBoszReOXXSt!KF0QhK8V{Gxniqua%`ey&{@I)*Qg;`JxpQVj94RS29MRO=',
        '?C5%6?6K<VqhrS&t*$;cmd-6b61BK>k(;-V+md<9A`g}1?~Rx@Z?<<}fcMPSIT8B`CtVmf?!u%==WA=vW1XWIlGWFES%G>9iEs@;',
        '9W<K@RSe1XRr~}L9G{iY-~`Oz2OZ~@^qI#BfT>k_WN@~zaaK@dhBd7x-1u&ej~i|Eo~rQ+o&Wb<VcNoEA6IbD)jI`O>ZUOp=#@y{',
        'y0~jYpb5j(dB5CHbLeWun-)GPJbXfE=!9_eVL_yeOK`A@OC-BDVsd!+<cNs+u(0|F*U%8TTUZ#LNgjKL$>Fa3+j`A=ea?&V@QRN1',
        '@`wS?qCGuhVmv*gxfEGwsLUlaR7(C02|-${qj;<XtBr6GE<rC004E;3zki+T<CBu&<C6*xGG8hdX}&SFKB=ib-f1bY<jx%kebUl=',
        'yi=2rtz;^#SvTJ2l!L08M2cCe#ow!(gxoo@43&2fs}#Tt$f;^PTYhJM+s;eyeo;b3Lh6LZGvz<+ZyS4l5-{~ImgnSvDM^L4PE}hQ',
        'x7{$ferw>1^AjeOMn&d3IoE7y+;v%S?woefg*AC&h8K~lmVl+XYLq?nRJH!ahsr{``RFce{j|Mp$B#XAt?zfXjXhKUab0WfA>CU)',
        '3y&`DscbFSp?CkjR?511T--t&%tUXSS&!~B=gqK#_4sfOL1>f-JZDXij~9__y~n+OB8joEp!2u<*)Z`z`{TEYD4mwd=vhv4rB_d7',
        '7wUo~Ajh!v_s^_0bj^cz|EMc(_V!s8Ue7d4D>F)~$3C9W;Q{31PpPvsXB-Fo1CFvZ+ioa#y&HPW_fun^)B!Z7Yy@>jxy&Hs*U;Jr',
        'hP;oHneJ@H$OvfUjpilj^_g{_*3sJ9**IlX*68s+%$W7V_|e&;rodB~32_-22?^;#Dl#)FE3&i3fcecwb8?P0PdMQ0xN+q41KHUJ',
        'rjOj{<aB@@m^!L7BW=tW#ZQGbHHC$hm5AR^x|QC?9j5uzw`k!kLy#)OYZT&@N1Zdoc}=wP1@jZT3dO}1l=FY#{eQF0{50-jQ^WUl',
        'qcUPL%1=%__Ip}P#z=TNIM6>NBp@(wNLF}QMrL?;2Jo*~8WFLyB5$6R)wIOXiz2qHOKh;Tng`ap`}=vghlIem5kWyC{30SSU!lr3',
        '^gc$6tU`=4dvil|y4#%|s+eM2Bxe6+iQczs=8QcdUM}7nCr{hpDf14#mzH90YiDcd<6!UU<>27;Q*~8U^@yCo#wqUQRaNEgDT4;(',
        '08vzwWcYA+*Ve|y*2=*FNlUF+8cB;}^mCDgE&wr<1u#L|oh2{<_enlt!t5bu+!H}N8eKu`Tr3lFXWt*>!a@60^>6ICp5NGW9}ch)',
        '<TsjXrmnDnaaG4q=vRF_0(8x0>hO!szouAmRn>pXCh+V~T!?&%K(5+aNOzH1YX;b-Tc|e}fy%|oTx7(dKhOAW@Ak;K^|NO^UApwa',
        'l;3UyPmvlY$xlYE^qxJUx^!&S;L&lFRb0Yt=cX|gX%pf8?;>`?*Grr3&b|-yt;`c<%IYf$EZw^nk7_%!u(WSqGwE{7`(zYL+~y`A',
        '-gR-5$Vix8K64#Pw*GK04h=0X2?;CWCUh0UZ@Xg$96K%mXVBWm-WS`Rx}@Q31O!q6IFlM4mX;P4o(j*J{b9x&_DxFi^-WHGzl?d+',
        ')gMW8FhX&UCscRqNn}#hwIV*6txzZxN<as^w|4D~gI3`Z#@uE7ty@g_#oWVRYxlX`cUdm3wjd0gC&FNX@=5FyuF_ff6dA*LFA>}d',
        '`^-<^7I%Nskiz?@Id*LylLV#jI+Ir?S<9!UeRuOF?1alW19-1D8%~$+h>aiXnSAMVv1=jvfp|sG&B6bqPquG$SCpEEf6;$>&2!J;',
        '5O~iv&N92?_HE2UZ}JIWeWRt<=Y4<9isO=E69^BKiauL{=nwpuemBKl@_mLW1j9~5;VcnxZ=4$wE+TmvA#8aFn-R{x|1lp8G9tZO',
        'uw(7o_nk>=l5FHtu*m3U*nv>?l9EP3WoRvoKzRqsk$@#``({+ThT4N!ij8^#3{d?zfpJzG9X4cK*wv>af7&v+vy}lReachIv#ZC)',
        'rqG|u>0h3}<NB5m{-^QYtilsBv&&YGt6k`fSA8(@qI<Y6kQcp+y(p#)SF>*H+>dFy8vLu_fzY1(B8>4=?6$BR8+rZlgtx~kZe^xO',
        'SLTgs^r{%|mj=r9nf_1UInh7>yB^PZxy0M^lBH*2bIgR*j)+nW^3S)~)*ecHjZ6N-EMp(V&>c{Fs8NOAsY`hmR#abZWPq4S56cP6',
        '%8rbrDLI|o@spP6xqYQas^eFT@{^Aak1rtgkznL+_b@_L#`y_4(q44lt#t)|!JR^#NSo<%Ml_ejw5EfLw7a9PjhDAn3R0}V()PKW',
        '(e(P@sWI+lp;qD`dzs&0EXf=sH+v*E9(9LwvAQF>$VEETB0X0s3%11hqgbEM=5B<YiXA$14v;9?K%fzA`V0IT`0Qd%cU90v<`|zV',
        'y^KCZH1mLHhNkS`6}B2tAQ;X6fw6i6!r0QEo`x_^0)?Vq9>5=f!*1qoS2FEjiLRERt>-n8wuT?H&{S7T_4&-}7IgIoJz5A%759}w',
        'aP5~r|2#-PQiOmkd^d(G;@}jLVE45We618+E929r(!PTm`+?$%aKT1B5V(Hr+9vQ4I@}wGXCXs8Pa;yekDGlc6RMo4;B3fF0W1Mi',
        'p>Ur=pJzKi%nN6$5TziP%$kv`*r8V>FYi85Q`?_R))UwFg!Mrq&(6x7ZQl=YO;1+_ENix(>O4KRW}0JT^2{?Mf$8YV;Os$zvVtnh',
        'RpGs#XUT`Bi@zT|EYf3}hyT<XZ&z;|=+8Xr>Hw}f%1$7GDZ5}eI|T9LKzUP<-5iO;A_+<Fai0S3A3Z^;tz5OAtND}ls+FpmG4W;6',
        '+G}$&Vn7)^(YxmAyoexe-7wd~2fu6ld(YaAs&Ucw!S)3eU}#=Sa&CTdQeMc|n%1da312(f46&rYH5#)a%kcB0a--tS`MUjv8rlz8',
        'xiEfnMbwnA%*kd>5doP)9XCuc^3KWeHu6tP>zXQyPPe!5>7(zBbkK~v%eOociHUIclsbmfLD9MiquSdvVs^1-WG*;mC2MdCysx}d',
        '^Hr6pwW(Knn%A#4qt;rxn%0EGq<E*Ldy8$vX@QYxYy{f;V}5xYzpg$aeMFe2O+<8NO~l@>#RKo^n@9UOM};MfjSff-j`j#m!l#yQ',
        '>_ysDb!Gqr5&$aCK8L@fl>Q=cAswCbnR`h7IJq-7*9Sr{o(@qQW=gu2gSP8eK!1Fr%R-9u7(HS}Gq0)sIM*@ftJEMO%Hclx2A<4n',
        'Zfa_tm6VtSZq8~(|DlhG1yff3HLmvW71O33NK805?W1o9dm6$%wi`R~e^^F1Qv3KD_!%KfA`}#FRUcbgnhT~KOh`O1b=r!*YsdYy',
        'a>_^Fu+K%lLY2QFji{VcccUKOI+|sR)7GAa_c9U!(~V&#k5YU+BK*WdMIya(biCZ5>pd<KQ*~fDlTFuXqA(^K_s)Ukcd&HEAlM%6',
        'kt!PLomX-Q|CrS9@KpQ)QrN*I`o_omqSqa=8KE_dG8}p?_Y!3WCsiG?HZPS*<w^uy-VMc_ux84ast#6qDY?(#q`Ko(PT=^~fz7?f',
        'KT%dFUsAMkC5!-H5Y7gC%$W!;dbCjF$P#5`I99?jlL&~<?2ZTl4#Q{n;&jG=uk`QsosXqZfn=G%1OM(E)rC(=ad)8{3<T+LJlF>I',
        'm%#%((6ucXgh4+APUNVm=xvSK8nNWpzz!bu9!*-^m$&qzzPuz78M`7IU8=HCz;XIO2AHLa)3kTj8YY+VyLuVE1mzDvIT|?yjAu?V',
        '!*JgzuDpt6%*oS=24)yc0NfVz`&cC2sR+}5>E7%y3iU@iz%O+W>;Sr2vi9QqrB*KM%iE8`j+BUii~#_4rZA&DofUxfm>3WllNyzN',
        '<h~+`-d7P8;HImmu&4KyM<SYAIc+9|+u2?9Nm0aeJJ+wr^&ptb#vv0Bk_xG&fTAg?Sn7-C0#o~V*Vsa1+?v3%tqds%g`KR2Vo_PS',
        'rH7P;AI=5YI@rSygsYNDVvaGY?r<OnKm-6EZvd0vU(0|t&|VJz0@^F!YxsIO%>dn1NM`sAzE}nH;UBB;DF<F7NH&K5l#NCfBm6T5',
        'A^qxTxvkDGUXcv%rE8|&5E>P8V3{JC-Zv_mc#1z+dvbq?C0fgPQr&)=v?!4~8YB``iXm6|J|1us{B&2}r6qdyiGm4t7cagyX)8SN',
        'N)%jL7Fs-TT)JPH!<|#RUHlsFwX{B+y9B=8Q}`S%|CYI0xhYaMZuRQ8CTu~0^jvx;&RGGg-qW+-aW;!xgS?6riq|nK60@GhAXRT3',
        '@hwbmZor>S8wQTwdg<3G_vSBuIcFvOdsEbkaJgfNbj--u8p+h6!g2ZA4x|IYgQd@AHo>ENlyLWE(;+_?45B~OZ*sOcxN60Aayo+|',
        'updx{Jb*IqEs2lme=G=KDp+^gaCnX1{>#;uW=y?Qw+&uBBMMJR@rV+nMMR}sF8<8i?%ShPUsac0T_5u+d_0xm$K}}$9$k`EgD3_n',
        'U(-8L?mGb6xcpNTaW9X+u!rES{$))%CQ;TC=EcrQPnw%p_kDw|yn2{ZKR0J5_eGxh0|rlTjBbnxm|rE|cs!QwS20j8(%&Nz-x-Hw',
        'n}lS`1N7fm`ox0B2@O=8&?Mp$8cziM9K~saaE(SPK#>i3Nf{}A2_@)-5<h{2L1~-hX6Qozq)z}Z>_FIQ^t$)K<$;n4JrMeJXwla$',
        'r@@LXz<eum8oBtj72eqbi@^$5OpgHOniKs1WW)Ns>y(;6qNhqp;ss)LBkCTm-}tDm?%{?FkLrM_T|s<YzMblwHW03Qw_w4$Rl;`s',
        '&t75rqSVwy>B7{->FJBr=TQOpY#Q`Y*M#D`Pwso*JZgV<Hx!@EtD(nos88T{LM<a65BYLN%<Ur{4=p}UX3mJ=eYlCv;P$<bVoS;&',
        '+(e`1p;0^%`*n}vp&V@soyN}nP{a6-C8O|t8IRdHy++ZG(Woo53p?jSEfW-Sdq0j@&?r4dpc*C7=c7@2-yH)rjG*f`+SYbu9pd&j',
        'q8Eqh0i6ia$%$4JzEi}KdPKdU`XIlpr@TbBQ+yD|^e3_?MI%OW+sQ8-Xg#8lU?>`iVs3gHnQ1e<lGa1s@t<_Looc#Bt1a|CrXP2P',
        '8iM@^__%$c=M<LoX!Nvb&^g+?p+zBe4O@4Vjvo=Rb=%yg)%|qfTAQRnC8-JYDY3bk^W>893DQOvr~XCL=QP7}PN{a&ER0(M?b5L=',
        ')zNzy8$^{vBp;tR$v=uZ>yE5id-(9`HAl85r==t(r==#bhS4j}pI?dJo<yhD)ul(L)z#rWZ-lan9l@SNo^e>O_*>O{s|j~WTQ0U#',
        'RWB}&L$MD~SuM8KqNL!}#rehKA8yEv4_#eajN)#A5$J1fk4zSv)EB__X4c)-J7oTkVb2$012$fKE$ol2LhsSG^iaL@#i?#3Hi{?x',
        'kUe+X<_C>!f~W*?vWj*mZ?u*K<c=y$dfIDSP<NX49IWOYot@m|h_(oAZKqjjEyr?UXHI1(K35HkNVle~g{$5#@OFiO`)Q=th~~eJ',
        '2%C?%yFecoue2q|5~7fRiqv<rNc6k+3o%o4Gimjms;WDyfgRmv!nIj5uho;gUtkoyofnVqxdw&?1|q%%jU2my0S-fH8dL|(DRo=c',
        'F(IhoP?cNhoFF=O58Mi$MK22hE@1Ve9UuS*FUgig!{=}-If;s*qYyS6A-d2s-VqG&*%FNpr3?%}R$xn+yy{Rx5XL7jYY9a7KngI7',
        'UM9N;e}gA>?10DNZx<0l6hfyUv!`;E&$-XgF4&*+$`8kW@M9~&+cWo(C4v88Ww~pYx35{Va>Y8<OW{73KCjUA@%8oc_Ve}u@7Awq',
        'TeEuQ$_*!jJi?q2*oTdXnzU!nq^SBmdw?!U{T4>a5))-n^2Efhu&9Z9_DsZY7ouDf6XfVEQFS*F=m0S|f{=geByUTvbGue#E7zEK',
        '()rPAY-?IXcv@O`L>hh6KRMaoFA4Yg)lnAAZ_M|qd(d!9Cf3J&p!AwMa36@e1EP=vp9^Y!2B~n%&&cjlDZo7EICWi5DDJ+7L{HD*',
        'AqCt6ViaI_*J@_V6F47KJ^@*9y~6w9Lwr(&tX5xRwLSSQqNxALld6FH<FhkcoCFGZ&#^IS*0FqYP8FJIY@88Nk&9N*#=l!a{Hotu',
        '0O;z4cY(CUsUA{rLF#r4Q5&v&&Fn=!0$0E(Ai;eWuEZzt#iaiNuNppzZF<Rw&QJpXq~Me`zXBTGV<udh^5gS>u_HVz;eAaA4vx>-',
        '6zx+Hu`407WpB!N$5Lk1dP^(d6eLeSpX6kG?g6JV-?La(R{yAdN>voqX}jvsn;aawy3(gKX$ZWdH1=^1%uWmmDxiUwo^WyM)!##^',
        '{oPHOBkDpaY)x+{k@;08CB!9VH1ALO=0s}C1TU9T>=}KS`*bAA09e(1ptyY-zX87GI$Fna9S^j?k+dm!L*8X9JD7$%Gs*&3@o~M|',
        '3S^MIA&mKpzKh(x1+eLl;WnU{><FdoV6YyI@2k?Eswtv$RQSwcF6`hX`QQ6Td(FvQf=7lje}N{1f!u3^9VTw52z}r(Y^GDZGml!H',
        'Ima{Fc4CV^9_xk1j#rH(zD#}8Jl-kZX_mPQk6Py6GSN2Lb57=RJUS3w1=FYpcr-qj){LG;JH<KAG`D-O#t)5-LVT>jqkWl9V8EkD',
        ')`0{ekO6}s2LX@z3|4|^Yt%UteFIsg9_}^5cGoMsrf+aU&jA}<kl)nRvZ{|0Nm+mRhT&=za@J#dNE{!@8J~VXP@jJLQ{n%&^KVXq',
        'mkzc&bwi0rC{Z<nFcP7vPSSG>US8gP3_d<{-HnNwWAKUi7(8puA|P6_tJ2%CPv6hAm=U9<HwTtR|DR66ufUy5{ETT;K0_-)+{^<+',
        'dd9g%t|38S@;gtDa1XN2vLk6JlHR{gWP6|T)=kdMoiZgicXE}7le341vy%rWO`I?xF|oEb(bmt`&ThmAEN8j$If}QV)%O5=`W66k',
        'XsW)ulXwF9a7uIe&cc$fXU+Yt9CpSghsCE?)*c>JadFPFE2DruTN{&*3~U_&1`mjzmX*FNGrhggHz~x~$wwrLZW@`hG->qa)YOOo',
        'G?#4X#<n3nx#L<Er@qs`+)z{h)STmLQX#g7EzM!BkrBZ6osq#77OL8p5!(nKJuUjWEfbhV*7}ScC!Jcg<YskpOhD+gr|o|NA?&QJ',
        'nlQ$Ga_#b4mC4b8!O+w^&@@rjP-PxuYGq+RC1L+5>)f?<C?GeD9WmU`-PEdNRb57lt#nRd^|tJvFDKQe`8xZ{Of5^>CuGjHah?mp',
        'wG6c#@|=RaC0a(>Mv)G(V0XM`J>hb2iJgs4uav(t?kH+IC?u;3#rXc}UOFLe&z7<Nic2WQlqzhw-7DWeV=v#w*SpZm=)Jh+3gqf)',
        'xEH*_%G5Ee`bwGNBJ7~ek@?SE`M!g_LppmMQFQPV+uCa!^Aj|HSHK^N8Ek7?moC120gZD(c=s|w^*9L{NAHFEZ-6M6!U!uA8FbBB',
        '#788e8bl&lnI3t11B!rRIedo{_TwliK_2|0mIKvYJxni)<TDXA`rp1amMzV0K9o7(=OrtCopu03KbTOEQ(09MHFhBP)#EFBUHxWt',
        '&TRc<#$v#2sdxs;zV8ZeU%j$+{PI{L&k(wmKFu9R^6*xlvDP6@3K1tf;BJUH!!`Vu?7H5Z@jfoE|C=-Vy}jP(JvI9F(~}N-Ref<r',
        '%dJU=f&5B#N>pxEoPU-PyY7)w{b*bHoo&?zN0$D$D(V*?Zt3bOD=LeOOO%Sqx%wOWG-FN~k=U=7JcQjXHLi?dFNy=rw98frhNipM',
        'H+(UB&atNJ^Ts>*ddqCRbQtsX!DEM*FRC5awsMop>eFe87t;bAWKvT@^%-fnYPRojXWgI5Ci<?bb50}y((~78;f=|>;i{F*+rl%V',
        '5<gzqqMnXla*Cze_JpboL3LE5Sz!M+7myMu9C+>GwsT2`56j(+ZS~BPWb?PISWz076l`Pte{l{1`!xId9b@+tq+L|p50!DuwSGDq',
        'kJF|X({ht@JYuHKE%>DG0OX#iX6`iZR|8@Zas^_MSSCgLjwPCgI>k{>K_q4h;g7r_5_&H9a5vXM#BFHjzx(#iGx(q|Fd#(>$myU(',
        '-UU`qpK4cmu7B{K%a0x&DT*yF`VU?qXgg&6*miRNCVoN@B7_1fsSi!)SoER;6f7u*D?l;5sw-GaV$q*oznbAcI(z#%ycg*c7()M?',
        'zW(!)`}h0CmAkG_zyAEe;nKuv-1(<`PS59pus;XgIuZ5g1nqrv;}JLaK{k58JlSRb*0wfa%6eQ>+zPfQ^7?EC9dr?B!#9*V5Ah+j',
        '+kfUE`k&52d{7>egz}Jkz|65BsrgtQNkcLQ4$KUTOt<PuL-=bHcU?VId5D`7x;uN4Jj55-q$`Ts@g5@UHgWZ>06mZ0>HT~FJOs>H',
        'Q9dfKC}te&Xt+K7F3=x8YSNV1PZv*oxDbdZW|h`9jHT=ERAwcP$%?JpS8(-W_I4+`z2jz|EKEN#KWR)w_V`k)9bLGd?T7r_U~;Dm',
        '?jj<_%Jx1pu#iZA6L3`g#f*ddNADZ87Ct^ua<bShWK;5*{hM~C(V?L5{u<b#S94<Ow%;d8U9MR<?wT>P5#zT<{^VzN73t^s*X)dX',
        '&#t?dhs>^~Aw4BLJs=Zy)L*FmUO6Uv<oKy~n&a2aj00vxNtuadnc&rlXkX_jx%0?HQQI~oH%NxeC?CB%&MK)n*EKfYD=rXGcS3&U',
        '03)VsKBg}D2X!J^X%BJzG!HjNFB`A#5@*M>?vC&ci9696_F2xw5pLeDL7pIXhO_k$shLSwp-WA<SBglG8dYBJI3~?ZDl?balCwS}',
        'Reawk4}@5fvqY5U>T<_OJ>oR3Tu=`f)g9Pjls%1444mi%^Z_*D|5r|#XaAd1M7M^$&lGYBeCmyR%`tv;iG$cW_Eje~P0RO7h>%H~',
        'HhUENyQU;MOQf7ad&~-3J7<T!p><()wo-&C1g-N>QiWCB3*-7PUDJGq4ec5XZk8n#WEPggj)@OeK7O@lk>dXSli=O>wh{R?^G7$`',
        'YWVr&XYl<J_z?4lQd`!G^!)rctZ0AYvtC_N@GLAOq%36=?3nP=f=5q6n+r~NcJ2l%{K~wgnMH}?4vjpzGP!w1%mIYe9qsNN|DRZy',
        'j;<Q8IXNOKJShowjQOVJ#`EC0r3X4YkAtS@DONraB_U%Dj5)C_v1L}mDiv0GKdFWO1FNLF3y4qeS9}gk+?<^p<?`>=w=P;Yx_s=K',
        'H8TO@oIJqFA;7+JZDVcfw4^L-HF0R}u82z>h{3xR<xa1p_ahq_psMWupYb!EpCW)VALM!TSPxu-Q1?}K-(HB11sK4WE-qMe0hp!+',
        '#pVwLrj@?I$8OR`6#==?p3e9*09V{F4pLD}R97WqRF$}o;5=%7Xg8EpB~{R4S=1+RJfW75%nT^anO2IN-QX>QY3-`Ha|g#P>T@w3',
        '^*b3i3BA3dH;}Vz9zILyZomSsyrZ=zCB<ZCv{cZ3^6{~PxsfR)=__BP;_Kp-ZVaEtcAs7I6^F*qA>1$TlFfMA&`warPGiD(n^oO@',
        '0`;k&$Vh}6mxMeqIrHVqO^Fitjk@{+MTO>iqFuXFssL;@uJr{XgSS9mYBP2^dvH*oonyX(S(P0RSXRV$7&}FaUsPI2FJeBb*zpYD',
        '{Xy2gI`>miG7?EO@dj=_u~c_I%1OWh`q0J(sk-{cMnyxkjCJA+hfW#gooi`h79jDBHI`}d{%m|&&T$qYVi_TBFDV!_j2`S~pEu9O',
        'GHcLK5FyhE6LGY6rIiid_mY?w%oqGX<eOxekM0&W)vYIdzNOq%ee;QkkB^q|eH}n`MylqRHZ!-~diBb;S_UGwFhf%>L*2wal99fW',
        'M5mN0$3BZ+%mr3DgZqQB0fQE-wVG8sP{h1q=6yc%>Ts5?B{DWQHnh=k4>A$iSkc-2QfDQ84*;#@R_k7W$vj@XFw3db6>&M0(`HzH',
        '6QxT|7x8VdQhisIKEyI}Q}x#TM_)nT<dW)?G8!&{zXFuw$n89)1#-o@COY63xNO?U*c%yOoVDB7GK6yyQvWM<rn;+^*W+rj0D5<b',
        '5p(kOR@u`lO`(dF$j8U^jSs+<wc<r;msx0TZfHnu4#FIU(r;gG9NJy2y1R<IpF3j02!#@SBHud+4XZw0p1d!nT3#&;EA~uH^?c`<',
        'lH&QNrYntkQ~gC=VB`&FyAYXeKuG4yeB*(!*&~8SD|9vGGC)NrS2HGzlss7v*^3X2?1jZjl55Uo)I5MuKK3ESIi3sP`V&^xN%jS)',
        'EAn%v_(wVAx;UkQJLXA)MS~5s{v6s^_`=9sSlXJqI@iVF^Z+wo{PdD+3eZiwv1o3%--kF2xrk3n%9bBni~!%1GP1vx$YPL3g`ac0',
        'y|rt@K%ueJ(y!9rx4?0bhot>IuP--bpsd*2(Pe;VKT9jWk^uh^`g-sS&67)<kd7|#2cb;xd*yZN2F;_$6MU$6#eF898;Rjmo6?--',
        'fzJrsR<Q}3=W5CL%kPqq4=n>2o&x)D_3Sn>{unGm;};SiLY|s}#`iomg{yr(m#d-3xsx^T8e@Z2#%d-OR-kLn5E@Ao%gpEIA-}bN',
        '_^meTIy;@fDc&{K8Ntt};PeW{svG`CH@uy0V+;`7TM@;1GKUa+EP=14pDXblC2BbF3&>C5MW+sJNdX4uVe*d_rwPLi0<+;$c)t3>',
        'g;|QChz8i%dJ3L7atOGcEL<|gE#1`d2aC`%jjju~?cYc5d!_A_J9BIC=WQ<5?`6(af1N)HhMtq!xR0>*F7d5h=n@qrqcr?+v+(@E',
        'Teb$xe=Lz%5=AqcSvwLtsP7FRxBM8gVjM9W8G-c7)df{&CRQ(2w84w;ZPYA3|M{Tr;2<C0*129U!VdrX+2e2e^#}d>>#O96@NusR',
        '9T9sU(W%6Als_su=|%K(awm>uPjiMI<(Pf%rA++J!Wte{biQ@`{(SnK{P@u?E8&lDfl{fuj;$v6;+b58ubS|kqJVuvn4y#(+E);x',
        'G<qLK_w<u!`r(KNRs~v?azRf`Mcy5cBmd8JILDZZ;3;_b7W}Qr#r}cBF=*3gi?@XYTKyd$%1%i2?87}1a&AF>j)DQamxjiE%cp15',
        'Te~!w7(ykyv>puDS@UjUYRQGP{0;dz>kG>t6=Z)GmYbL%8t*;?D*R&#gWPKK2aJgNB6-5qrU(yDa=J%mz)e9jYv&FyYthVVNiQgd',
        'UuMQr!*MJE{>Q9b4K&pT7C&nU$pGL)Up<CC+j}xgw&KZ*(6=3wd-L_&$zP4R0M~cARBQ-D(`JLBRxj}@UHG&mUjaWknQvX<)&Tyh',
        'k%z|Z`!Rip5Ko$|%v+l~G$A@HO*5CvTnW|^e>Izk<}#kl1;r8ENwgvbBy&`IdRMuIYq5TBjzO62P`wX1r>mx%9Q_U4(5YC=T<+!!',
        '^dDd^Vzm`6xnn@tq49_h&>!UY=0QUXukfvehY2bRS46Rw*l^M*&$lp>n6VmBs`5*ev*?*y{KKo_J(_%H^L4E~hMb1CH&q=f8-Kkp',
        'q*6Jl;n~5FKyE9mKUbCeb#7kJw1}9gz9v~qBdsIAYe>gzY%H7U?`ZYT-qm-;wIv|qh^!^H`t-QSz&qBm%sB7lf!h5UvQ{u>JPXV*',
        'UB(Fz#}la@|K^YU+jeEW!dK<rR8r;&nTG^^vFjI_$BZCPK+A)Uv^O_^wqSxtKC)d&K_dj;jNmgcoZha67ZUho1pk7-cc|fW3H&*N',
        'KOyih)bI-$IFmx)J5_L$8yTtTeqg^rbi>u-O%&%<beRztPVd&xC2&NS8A;%K)bK_E$M`o9_+B;qga*zE2z;NKuASmM@+HraX@!CQ',
        '<j!4`(i_rQiuIsBd_}4K^BG<kmMOcRb49Xysg`L7m&F+&cpJTq?T6qOh=t<udyv%(p>Q#nl){mXs+QES-W_U2Oz-A9smP4Da`k;I',
        '#E~$gmMz?wGWbIay!rdaov`CtwdIgbJ@JCSc;P8tsq*-Ex6sOI?tP5o;#{N4EJTCi8p18(%#mTcWd}1gUGuMX#MzGiYH{+0;3WGx',
        '*Ww~ESg&g!?58U*^Ktgi@*cWK$5>*hYcSZ<z*!O;W6CG}Wl2zvJ&pJmqI^Y<5}Q)`-5g_f^`-1-_!qnf@4`-*-2<_7!>%u;zi)?U',
        ';4a__^sC-frlwy``+Vc6J?v8?jc7#^yM!@8Q3p;e$#GP7X0Kiq64LhXYOd<Q;gf{~@+DR4%IXdlMJ}zB*VpC^aS``tA+L{Q9DOP+',
        '%pufCFJij2Y&aNKeX3^k>BiEXB^FK=U#*>VQ(ksDb35Q|&<QH@&2==Bh7e5!bKXolSrw^t3(ceZsx^gnYUTx{N9r7ClVa3@M>--8',
        '?@>=0*ju(`?3$N2^2me>*C&E2p2^l;#ToIt7*28_Y{BOnGRJMpJiBDlY4}hQA3Idu;OgX#pKps}5N0^q2Z!OiKWH3l4U?<TV>y`6',
        'xDkj*fjnG1e>42%X2uV|(9BnM^txmAoQ$#+MNY86CSnlJBE;RrLzH&y7x?`_>$lb`t8_OWj@i(XSiIwgmp0#LQhZ>##6v=MF}%{5',
        'RN|XH>FFiCdZ7OGusil+AHVd{NJHaw;jNjqHx>rZ@lKLvT31Y#n0~%-;#&9}?DMf_56-hT5DlL_yXEClYujgoj24#<6|_xge%yjJ',
        '&=p3wLTd%w>1tNTS2QJK-h$`cL)Bd)0-3oGpqC7M3B$mb?CCN4%i%QihF6;*Gno$O{fazQ8^wSnC~{kDKMOwxo|#DJakJI;Es;D)',
        'b@XYeik|tdy>GbT8>v#J?i|_Bh}LnrG;^4VZGStVme+7kc|TdBbzo&dczB+uk6}M8qoJW%y0$~jhsUzQ&=@8r!De`0ex5KQ(B8L?',
        'sYRcJl%S&QC{J%dg9N~o4sbM<*f^Qk;ZAH9MICsqEdF?IF5nOhMX#VS`^Uhk=nd=4gB=C_<mQrR3ZO9_;1D~id|x@5wGQ(G9ms3K',
        'vsNmABl1|wD0?M-a&$jt0r-h)Q*~GYVymiH{-e3}o+!91Ffo!Ho~oO()B;T~EIfaJwqE~P+B!C-=5{fx8KBva2!47WTQfYX;+!aV',
        'A2h3hA{Qfxjq{M<=Hy&Mp#aa6CK^s%z?TXI;uQU&&s;`HasoLYAWv^r#bfF|WA&=)<wh()4&7&{#6;MiS0W6F9qc<mkTyIjST~35',
        'V=VDB0-V??*~`I95;DXvAT+(oz{J&ZY^1APS70kNlO|;k!!{vcF419p8_EK-(GbaIF?t0fFMoyEL?`g=k<^wde*~M5t+e&KNydLi',
        'hw(BHd>Pe;x{Tp;7lIow?Q|%A2!gMmOhE&Nqdb77B+NB1pY#9UXM&PZ5fP*CD_d%A;pA*?E@fLHMvo>yb7yCB3n_jMkn$0V{MT{5',
        'xDF=3Ud~eG3dvIuM5>#@9~cDs>!b}#&o>_u6E{@P-K~GxfcO-%!AbG#9v5$&n86ADcEhclbYcdCx(&BOd=F$+fCJn-H0$q`$Cdaw',
        'F2k5D;4N=I$#0-C1>1}ADZp8V9^(U^@Z-rZFX3<KmlYHX4x(RJ#Wk=-d4XWr@~#O%++h{9M){29gNqdPUV<c%!2Cjsx!+XQCLoWb',
        '#JcR2662CUk9J)wvJHn2W6S;=8C!mMSit}-J)K!(DcQwv8GvR&!ill*6O)6Ba-!sZGynq!xCkY-(jj*EnfUI^U$h4&!0|m$ZHKBm',
        'IOMd4_N4lW_1e{1ULhf#TAZME^Ol;3%<M2+uo;OWUon5U>UkP+kzB;kT5AgG>I&C#2SA9^n>SAIuxi{KG>-hgFaW+K-%=oD#D#Tr',
        '1#8A}M;%|kb_DoA7St^27Trkyi0$1Rg~(tA>{8x>T?%A<;4pQI*^hqXgB`$`#DzZS?KPK%dj`Z3nI3N6G0+FEm2t(tRis@8xitM_',
        '^52SY;5u~m&nVJZh{p24POwUIHOp`*s=T>71i(j1YFtJ=i3jsH%MaVO$RyoHw=aw<{(Z3ZVDt2{)Tn-axPaj~cCEK2^0a3BRee!K',
        'G%<KgP-b{>xGdKmY4;kUvz)`Wu^V^gsVH*;5d`1mt68j{AnZ$vdgTj*L+18~sghw;&IWb{I(;xB61!oBayK7q0Dj<cQn{6tPJiB?',
        '+MaXi?T*83ooTk^e!YgN#HJ*NfYiK&+AWh~+*0hF3d}?WS(9!}CtVrjiNq;rCGN!yJNQ|Q%xun?Jk{<Wo0sa?TGX8e3p7o>Amu>A',
        'o!J&X!}<-Ci9M<#vRXq$wfl=_<OZ!sG1<DR<?gf=`cU}cshi?i-k%&j#A~?Aa4qoIIBTT6nT5UxzOe9FJZ<bgq${KmHX2dqgZ=m}',
        'O0*s_5Tq|ViP9ziPY69bI(~+unT8eI0YN+e{=~I%-{eXAs$8G^?v*4rE}KwaHo!F*{yh5G#i>&-J{$crWG1K0nkP#dKR!v?+DxA2',
        '(24NRL-_gNAbu7eu1X>a%;AQUr@yGuXjQs_Xc>sa7G$I6@j!D_YgXo$Mftl@eNbFC6uvYP^z9R!z;sH60p5^dLkHTehBp^(gJ132',
        '0ob`k4<-n-uq_*tXZ;UH^bS!M2t7Ll*~ewzzBaKIv19|0tAPROYGVKjK@P_L$GxG>#&W`R_y5EmT!}OeZh`3ylS|-l@YMD;5CDc2',
        'Pi}ydp%sK^Mm6|pw+J5(90?`(mER)#=A>^<YD|;9IZK3F)#vv1m5RXh36?f>4TJz^j*>)6IHhrN2`~WxZQI~U_<QN(M$iOqVf+Z+',
        '2ZWCYj)ki8M5dODK_1J3NKCCFBLk5MgCb5ui^ud1zrcfvL!+H?=cKRN9m&xQ%P|U;M_UA04@n-xng)7Gy^Vcn2l?mo!&l8p+y$?l',
        'aNT<Dbo4^tT+i<Vm_B_1CPoKlNSv&(G?9;F!x3#hI6++}mFjxsGv){vMeZ=g)kG1n?7g3S)S314isG{$=+dyTlG4zyQqWNn7FJvm',
        '8dkzZ<-Av9v!<bCW$0T`*O&N1XlXH)rVZ3#?C1}qPK9vO6~`1OPSPK8KUCw&p97o>T9Hj@Q53#uUfoHw`EhWWtllBa;QK?l>n7Xf',
        '2H;1oqpQ7`SL|KVQ^g24z5wtUKo2%P0l4|c;jfE~;S=U07y{d?M#5D<Tm}XMGgZ}xBGoXw9#mzhp1`xW&(NXSgkPV+whO9E=I@nH',
        'YxkF!7`(A5*!^q#wKFJHNwlw%mlT-}3=j9)80m9t{i1_88VdZ+5d|EGkiv^ppM(~wyy_G5xSyXIa*;_QQPt;8>F~q#&ArKPUi`yY',
        '9^F0|K9325uacsYBURLs?8js3#<6kEHIzCt<TKW|XG*GhQlzHVGk49JI8PJ(Hv*&JO`v~%WZ>#Z-;8~6O_%F5SDvE}lR8*$et8lN',
        '9-xgFj7b!Vid%<!Zq3QftxV;*Sqk^^3kLn;OWj7|3J%USnQXNCI({my*L2sbosb^IILkHZ75&{!_+%!D=tH-D_T%~fmPrnKzj4fN',
        'Nv~K{-~f-OB}VqK43q{0$H9<~1NwQ*eYH322wyopzVfSQ_3#ip$LI=NDx#`9-G-{FO59e)@xSua%@IDZ(3j9GBkf*oA2W6GR{y5+',
        'H-~4=&92{7&^JC_2X5-ujB_4m&jIQGEFjfS-E=T~{k*g#e=miv;9u_;-Qb3V5gEhWO>usY7F#J>iSmnAq>h2Uv%1V8^!+vQYVK?d',
        'wT8Pu;#ZWEyO*#rU;><lM&727mGe&07TC^ERFJ5HpF5Gs#(}?iLF81S`^-f|0%{{8xwBj@;w;aaSxk(vb~gd;9#%oXMMm@H>A_lk',
        '_VHkUd0Lf$e)SN?|Hs;Qz(sL<kI!to6G5aSy*B~rAWZ}W6a=J+2m<!rd+%Li*H~hSU1N>iBpPdCN!4g#G_l3ll87Y=xBGuHyLa3n',
        'p!xp(H=mi^eK$Mv=Dj!l&D&-UK`JI}(CU8L^@Jb!DCEs7XDl_Hs@ILh-}Tmqz2)^`q+*saQ_xNMf7Nm>1m0W5%tkk9jBTtBynhA}',
        'Bae5*r_6Z10iAEquUl!YnpuQJ){tD?xJ`c`i?1mL)^=MQ6&1No&T;k_n3XlCr*>@1H&BePRkm*Gtta!7FUQzrG|d_`D2tAHLqZ8g',
        '_r#a12e*%6<U@3#YQ;#eYSfO`buC_|hdzfI!dF)s8wa0Y1N69kTJ*MhyUVpFOjn<pSU1TJ)kBf4mz67P_#ZL5#bG*L(6w!$0U0f0',
        'N9nY@wynUo>%*CUv~tFK^A9LISo#>?&vfva2Ji~K^i#=7E$+<qu91GS0`Fm%zF<e;y|_5T^bOz@L>Bb+2m(<EIoJEXI|ZzsCA|N#',
        'YjA4WvSmk?EnAM`moH!TURwyF{WAGi241($FZ@I51qzaE<aK^3_xI?20?0i&h;Hk{x!zoIHJo`(X+qZb&+(-+9i*8iukTZMBHb?_',
        '@yAFl^x<rR0`EuQj7El6MauE<cJb-U$<^_zBIWp6xGw%-IX+s8^Q$BncdTvwMDMM`@>}q4n*49flj?T?=R3SN;Uikfw$Z*Nujq4!',
        '$XmikVjp@fWjD$z`||%rCbj*~Vp%>!4?+2#?IQordeHwF-Tzl={hvkE>&Z8$d|l@x-FkAqCX;(5gi`oIt^ndU(aIzHn9Q&N9?Xpg',
        'xSg&Y-pXqzyg%mxaC_Za>1l3|V!6ej6C?8<*BTwWYe<A%&&FSETe$mtX!*?ieych+>-WcDasMN6^*P$ZFZ%aGTsnKwrd-;&CAZ4K',
        '^UOY6pS5OEX5T}%hhM&)WzgGu;|N{HCSwI7jm1!R;^+!4-67SYrLqoDnk$&K*BE(V?Q}%m_G0_wxbzLoN$5ALb^6@?Epr`%EXpQk',
        'epS+9*S5$$dV>+>ic?avX35TBWx>$pV?1DL#s@=^l7=j3J9!`$D*~OZ-7O99#Zv>5UA^1Jnc4M^bxjH&`>smQhis??LivQDGE>qL',
        '(+pSB4)>ff8)tObeZFY(AVR(Ve;g7Q;0JAUwB#}g=xDxZeeVm?6?y?)C3-Z{_N>T>*tWrXRMC!aB6YpLXAP>X2SQi^A-*my{U&$U',
        'V!L3Eb_soFweGfWfH=lEkq{!hG&qGpQ~T4xyZTqtWawc(!;<UP=PQV8ady15ygsjtNT2fy$UdoS&6W31WqPJqlEP-ZOeC(SX193r',
        'hr!xWV%3)Q<%?Ix`1nD${#x_H<ew(Zmi+sEvACKQAsODK-MacamIQ})vci$#tJ~S@jtpDAqZMF~4RUmVw?e_hVu)KHnMH9_h-uy+',
        '-E!^`5gs0qW&$Uu$miPV$dcn$iXTMpyL=(fV+Lk^PQ#;DKA+!j%l=fcP4=9;76EOpVT4U43=v;emEsL+u-z}?<~<PqFtT(se`VIh',
        '#45<gyQu}|-9EKtSI^j?eM^1BO3&hcEo@pZ95ZTZraj=+4k2|O;1*LqZt+YF)Kq!{Z*H~tLG1oZUlSx*8OeG5HUpBavuEcg__z5<',
        '#HN#miZ3h66dYBYeYAUUaG#}-!^G1B*{9}~>iUp1X<=ld5-Xy7B7Nl%wa|pOqNh#jM38O*YRkbI6I=+v0GCFS2)(qp%>7e*Ft6$f',
        'Yc5k3-vI;?y46N4tfi+isuzv`4k7X`u|aO{qM9(X&$6gN;^j}OvrVmm-wPc-1+Gb)dvK(0gdf)9=i3;X#5W`)($)kb@z9g;ro1K8',
        'gqMm&0#fbOhcgp&t#w}+j%N`#jD)nUwVwgJsWu&9uMkapi7?(9(x0nKe}Fz5=1aQtg{=gRD*2^`=^McH)BhJnz;hK%h#kDWwp;?y',
        'G+4)R;)?2W(VeP!q$fsBBH78Gr-k#=^x<rPHeV)MKG`odp(pzTb?~(Y@Mf|`0NFQUzM?jkLi9BZYG?*9XXMTB$RP2gcusEvWpCMr',
        'uKn+K?zE_Kf_O{ZP9`$L{v{Kc!Y&&-YjcL#C18G5hY8FX;rX1>zkA3#SHOv@lB?kk7Ts65$v<V=(f_{^f1&?(VB+Eb8A|{Dlu_}B',
        '{4d==kASjgWnB`qfqjBl`u_x;tWY>neG(Y;v1DxS`tI4n=ubZR#NOgMwpFGIN!iX`hEoLuF)BM6OcN$uim_|!Z!iW299}YKxvdJ0',
        '6@cS81;=05r4+}PY&`tWDl<(jj_*Bx*bjIVn_jwRICrph^bRtdJ8UKR9(#M<U}tTY7h*VXppwBor{kI3tW7mPU{3s$+a<%9?_@pz',
        '=1?E_yPOj}{~tlliG^s7-$asjS5e`meW$FOrJ$}`@Fh`vcX=mydNpmoWn}3qoVWY0vnea)gckkLe+d4dYl2%^NQaimo$-#qjy9=1',
        'd#tpMz$OgabXshO^*z~R!;YVuarPTqkH#($42tSvZ4{W~)v{Br_`OP<RN&ROeR%VBlXSXE!Xv;x2c-%|AcqU8b(+|}4W9*PsxP%s',
        '`dJ6hk?ydM30(mGxwsq3KTDQF%`%X~Ji7bvSTY6pYz6)Y*O;@Q@gHk01AI=sI#hD{v&h^R-C!qZ(^vGpLTL@2CJH)zFB!gu?(BRE',
        'GkimM62LFwJZVe~xCzyVm?<_igmbrPzNgr02JpT5a5h0NozLia{uKM0F21rmtJ5#@7S%#9TS#>)X`2$Mvhe&b`?c&?(%hK8j>e|%',
        'y=7b8Yc@7p{;pkHz|K)43eJp*_P2W2)j-ahoYrtPbC)%6*onW0PbM}HLHzEtNZIninlr_zZC80(8-Z}UFgc7G1`Ss<;uhRwM`H2^',
        'xbgDbLyNksFH0`4_j2-aTVU-VKDTx-8+vhqPHQEZa<#q2#!zd8Gp6wJs#^wHE17(>PHSbm6d;-?S}RKkT+v$DE)Ah@qP23I!gnh0',
        'RTNINR$fu~Ckp(k7S493a9L|*J8DJYvewFZoz}{BsiS6<PHW}7PHSbmw3WhTt(EgSt(EQ44-_tIt(@0st!zi8S~zFHou~cT5X~x(',
        '`u%!R#?d)tIs1>HRPr3)lOVUG<X=L!%8BwWJQIO$!t7Qng+)?pyagXotnEYl8tIMEmihCyY@Rm{AK9{S{`T$j7t;5K@Ld@dq?<_6',
        'rK$MAT;QQ5j0vS{q_(xJlB$#>4`05lTP+==P2FJCv{=cC%rBlmnqO$sWDWWwn9A@<nijadF3k_bZFtx_(<Qh*qpPw`kZvt~2sFVq',
        'O6u104ZDn(ZKd-c>vAycBob0;V9$=%<>FdfE_!9Kc`F<7PrahlnGu=eVBrP(oSvg`@rbk#I2g-Qb)v|4mdyO9wgd1$w)(|(!qIro',
        '$3{!Uc=2SHYZpH*kH~57>KN6sT|~sN{<#(#1AB#gh6K2`Yu;|c!YP;yvlX9PG&2imH=$#4m&`VPo}EkLiYAS7-w-z|EGjeHH_EG#',
        'wYRgiF>SE{6QL<x>FO=r2N{jk&cc`|tfDKfFjCIPcSJ|55?|Nes~|3)bAtLDLp40Z*?-wN!gO-36DvKtw`__{(Zbf>a5mf^HQX)j',
        'Z)}!Pn&TIi-^{JS!VlKYbF}ScZqv43-!=|5L)GT$hJJnlt{gK6^Bn8o=4@-)P?+vh;9EE=Hr8zB2g_H+jEMFN?(E;SW6$0}U7PvX',
        'McTQ01_T%-IGPycx;r<EXlkG7O{|u86WIny!tW>)MWYnxL5HG=Xf9f;*krb$N2P7F${FOyv`SkjvL5HH-AE_)BO<*#y~w8$RJ`6u',
        '#^Bynwn|6$|7hRfDt|o1JUon{Nm<nOrZpED;t`3hN905Y%)uFl4&t_R{G)P*+jZ{Sx3k^w+$jG!;{Jn&#Qk%c#^j7}DDT_1+=2gY',
        'i1=*LLa|~fw<^A=e{8IO)A%=baZUZ>;{2P&@zX1J2t%{HU2=+xb6mXJ4pVXY>Y;6Y-Eun?=eqf14J~^69h^KAH(t6Fn+@f==8kaa',
        '+^<h(yAipOkoUfW2gOeTqnr_To%{9cVo&g#jnfYt#2K@jtw_txPAhL3r%i@l<J$Hy872Py(HimgD3e|p8NC{f!p>_x!Y-p4^?5U<',
        'M&y}oa&y}tV^k{cQ!U`{AX8+Hte}-~fm+lI=5_I?B}yjemU!~fAv5}kR(fS!y>YlZ4?|Bg?25q~817ctV6#YlI%BI)*U)h6YQy|3',
        'P7wdbjqzw=0t-*Vjm5vk>EiR5n48Qr{k&?xp+h(YpT{l5BVzM_ei<44`)6eI6N6{W;LT=KtqS&V4+(Mi2o?iBV}3p?mWY*8l1L)h',
        '6cRKRpF`sR&;(xyo|DB+UmQ6iejx^nhjAkQ2B&Ji%jidwygp+FZXiCJfjxqPJUoJf#lc@35$}@^>!Ds!7cQF*K}J9aCu*?*W0jeW',
        'm1}sUnYWFZ3U0j3A~6e3kv8xLYZK|+3qQvd_b*jtTq^wR67%rVXX2`H(TlEJS=9W?l#~=U9zPe2?~6T_pE<J}e|Y~hvElN7faUnb',
        'XSiGCoTg0fqN`UIX+CMnHcm-NMGP8-=ChVC8+)gn^<hOTN;<75Dq7L0WJS^CHj$C(>5-9b@XeABi;6xhDOu66<BF2VR;?nzlbzU}',
        'Xg;TcbX80%B(J<HIvUXC>>|%K^)TcZL=OXE?4i#!^d;mNL|+18>_fF_E5yMYGa*enZW65YPoC<jRS-2A`SU2yMzd}5OGf9%Pmgxb',
        'T@%y$yOhQ%F{k@KMYfI>e8=_udQ1&#7u12vo09G9Yv=IB(>6MI+MCid=W#is{=Pu0N=h{E_33~(-)|i0PU%X>{|CN^?0j1&Hg9Du',
        'Bb9p~ORY`YSeu)XQzi^K$Fp)upP>h)_}a8?`lxyP1l#)#*~OjwZH-knL6U`IV<U4ebI|-@`-=Nq&6<$)@wdy`uAQE1XCCTYCN2}t',
        'p8e_zY<eOkq&=a7HeZ3sq;!F<kh=kN5SX@k%DO5ym+yq=3XZ4FB3LA}*2rAh373hbwbe>h0tUMbO<T9V`})Gz5C6b6<JPQC8YLbL',
        'cZ&^dyvZ~u!K2yEF=IE4YMyX$*T~oZ4voF=mH44}VZzSKi8wMiZfrr}=;q<Gu+!u<;sb)yYKmhgz^NV8@F|qOXMf<+6fL^iR7*n0',
        '3LRk|(M^cJYv+I3|J?NA*|9mkU8BM}$9J3@%{0QTCuAp2`t_T>x0ALp9_}0GP5gd*^MD&o{6lg>eOmhRDje9}=T+NF;^U*`(`euK',
        'M*5x`2KlO(T(p5wOVPLhsX9t~&TjDR4N@h4H9gGS903wY#d%l7PrZSSrl0&<{5Tb=^s7(tVB8J2nsMlNG3Y)?=_<Urg2*FNFbLJu',
        'y&yTsx)+esahkJ%?u4xOPIM>W|Ibk^`V6)2mdfAFW>SR{Aa{BG-57zaG1jxk25`EcPvCngT;9hl)s!=FbmV6Vt+GQcBT5^+O{y=i',
        'UDJ&K1%od)84Lk?>VL+GlQuyfQ}pw2bv0b`FvtU|#k<(yKW78^T!MUd=;u@0NLH7R6>g<BnmvYpe(&-;hJ5~b?|dxrM7?|_GFJa3',
        'A2$C#<nvT7A9H5Pd*?%BppuTC?c^S*j*+^4Zr01+8b&Y2{S33LIh3x+FOUm1@jvEkdE>2oZ{WTEEnkomwzomPb<P>|q=kQb-cVUn',
        'Xx^1G;ppc7n6o8Lcsu74HO1&Xa^@`lbIzZ<opWwY!Kyds`ZO@ez8@b$;cg&-a*M=D520Pl;{T*yWN%8&m&%8Z==oAW5T;BzYb-UJ',
        's`jQv{d_xw<n_k;Hi;G3cw8SZx2E00gAN?f)eMfcvB1a0pM0D3%<A2JFb?*0@h-@y)YQ}{Vx>ob$Itvyibrt$c+@Vog|%aIf7n|@',
        '9meji4J%EjnzXHR2Tn^*!Ir{5=3)S@(^_DA3U3%MqNPLD`Oy*HQLC*Z$H(?G{_|>W3HC5N|Ekk&*v=kCVJHgbh6xDS)S}5i4xQ>Y',
        'tDyMEQ29aslYa$Tm4;%cpzpj=&dMqc?Qqx>e{R{#c9|=G9b@!Dyl$ul!MvO_Y0`S|*c^f>ih@J3cU+GagO?Ho2d08ERhPtw6)n~;',
        'Y%QUUI%Np_`6T(53tnCYE$6-2X{0B_t4L3%)$RcJOapiYs#jA2h|^oeQFxi;XP^;Li*^BlvkCfeJ#B>AH7H1YJ-8fSyYD3L%P5+9',
        'WWOo2c3r@l8?QD_jBZDr>nT1I*JwA!S8C|~YpjoNXawDV-9OUTIi`J@z5Vcyqk~87rkmoa<ZM%q?z)}W(vZkPYk4Pj>cXz+9eqJi',
        '^P^Lo^>(~!?sOUKK-Id_L`tL&*WW>t;b<&@8|-lD!<#DWzjQ}QIe~=b{Q_nZ*|X>M4`M1+_U(-(`3PyYS({cq9zltYcH(LAH@yv~',
        '*UItAXOZb*MPutK@kc}DIo&FYP)XPL?+JF=)fRbeojFI(Lb<_Q!JDigQHx;OI^^ZW7&|5>1gh^Cw;8&;Ltc*-w&uPGp)StDmlnI4',
        'Ho==RC$Gzi82Rk$(YeDT{R*PRiZ09Ay2TM~0kZy2#|_T4jpq!_Gw7>T&JT!D7bnD|r1Z=)Y!46m6~>tMY@V8%H1t4AvM)<^67Vf*',
        'w@9|Z=dCCrXsJggTcjWY%d*U-;+<nHGIA1=i$|A-Zwv029cWZ~Y>BJ;gg)Zm_EGk$LYDo!+y$G&mWLm2+B+fVBj1TXO%f}{6Z$E8',
        'Ilx0ps>2cjIqzcUBTup?Y~iYCJrGS-m34KdieW%+cS2h$E4srIf(74>%lge*n>u<`zG+s7&9vivT-~Si0Q9;Yn9}4sCe}O3(=4%H',
        'VkkDT>U{0^@Ux3c4y@{L9X7~&?A0k^#msq4!tGZDZ~ML<Gbg$K_~zDp=Ge5LC`B(Ui>{=_R`0X0^Wiy8k!QMYSiMRoS_&Ux6Z$<!',
        '-H8<CTq2B&sY<&O!*)AzYhFxL{?UhPx?aDux$}_YzxHCksq$|=wqNqVt-D{?d6_@A_O@;0-DX7ltQBLShj0&WW^ChWZ>r?MYyPJ^',
        'm^PZd?~zBHtB?%JgYqqB`@)#DxptmcizX+o8!e`F?OL9AYAqzWTos_8%b3)n1gi0>yAV(Yz?H88TuB+GYP?|<w@4yoC>B}#i!{0Z',
        '(@4dGva$vmY-@_AWPqM#CWAWjlbLg>pXo@VzH!7iwGSBh=d}$E>fgav4G{h%Jq@#yw>Q)fP;I3>&|?0AVf$$SZ~D&tyuJExgMGN#',
        '_XYRr!wurszAw0!PtxwF$p6-}&nL&1;in1QApP3+7x(JJ<@oQ`576IF)b5NiD#LT!<bAp$#?qaz>UnS7v)&prUQYYzWrHSodi8gh',
        '$xd4JJh$#yVa*xtwfYmLyZ=FYGuygTt~sZ*c%PK@=CbvrY&eUZRDa6a{XoJcx*y2O`+&Nc&b#K6wazDZ>oTo505{}<-YoK6^Qk)K',
        'kJsKOC%RMl-Z|AVU$l77ob+ai@6M^#`QfB@=ft6bVkX-{$b}J#J8DMs;&6mHJ>)1`bq5a7!^6;p6N^~T{*n)gv2N8L^YeDA-i^ZP',
        '&F`&^ESu`tKit_lyuatvvdFdfqNj(sxP*n_k;lSXw+;(UOAEDebF;B=abfRX#~E!_q`g|<6t0;V?zH@M+KM)~?RDoc2A*bW9x~SA',
        '$kt)xt+3VyTy3n~Ty1Pz0p|$7Suns^hpfC@8L34!l$t0pCwp+-b>Pf2ok5VD0hmXXPxTxS=Ik6cz;jwz)Y^M<#hG~IXJKh+6bBnu',
        '$llG3y-TnSbG|O_2i#vTcM2!iu6PBwi=T>XH4o?Op={&oW(_DKp7aF`Dpb=EOSRKG(Qnp3|D$@ZQ^P#5!lQg#Tv<D|(4P1qRu+^Q',
        '1~s_guB1UMb$V-+-y`@Y-I#h$9&=aUR7TI=>p5iJhHD=@nxnei$tWv5sCiJsH^#w{&D#6F*?P5U`=P$R{fos&p0&uC*FH5mxHLg%',
        'i<^t*vg$vzJb@SwdNXIx=(0WA#vYvL7Zj+O(lR%?R|?Zjvnru?i(!SVYt^Nz&*H)(Vi}7@t`<WblCuH=Bb?*p71IsI4XE{^Aq|F0',
        'UlOeo;ea?Ap3ka??HzCC3Li~v=gWF<ne0V|=Z})J@=dNnU3v<=$3S>izIv}clTN-4mLs3J*W6<!YsPaQYERx*&6hO`!&!k@!COE|',
        '%OdB<R{fZZjJuqQE7eCab9eDb#f?9jlTjf4D_+O<;#-a?6JI?dc5XFG7!F1*0$y(GK=Hc82t1;v*)64^Fh7tnLwaZB0jGBOu#69T',
        'tN;G~Rb#{oc(*0YQ%(5-NNZ*IQC*n|=*eWC_dN+uJ{8W_cN!k1Y$V?s?>ZA-by%iPTd^CIa5`YMs!2~$9x;^}NbQNJWQ!uCOy(rb',
        'F8B~JEFz|oWApTce=L*Cod8;_mNZg7EiE{*ithK`XX(jj6T8$a)#5|S+4=R$XVomZJcHRBe3SbEuv(9@r14rh1B8q|?du+YlCq9^',
        '>qPvYG;ei{WARnLQI2Zuk_osA9kVDF#9&O*l|EmAoEHf(AmtmRzfqNLU!gABU%_M-!T{c`UYlU@*h>$tjMeJ06;`6z%%sY2gkR^w',
        '9ODYA(}tK%TdA1B&xkhi*o<hRKoH8OgXH?wbGpo;=4AO&?GuqaJ?$FyaV@iss@NI}iZTYzxSN1MvwAS7W4Q99c05+rqBFxzsBIQ>',
        'ux4LOzfMc-tpV+N(Rt|{!b_kvX=h05F6SRUr?OR-CQYt?%@3>UKBz>jxO1;Qdu*bwvd^@Z&7tr!l>U!IU&dY;|JACs<h@RBm3H_i',
        'pL}9Sz;OVfCm|r6q<Ygif8)k`r1J#Qxvov8*3qa19;7!C{Q%FdGc8iKpOsNlJz7rQR<*4BUZ$zmWR6~GS>`=bwp{&I%Bx@0n=+K2',
        'U7<_4?s1+44t{%#_l-h~+7hqdCLfp2U!!(uI&hcewXpha^~q;24bEP&(mp6a2KS1#_qNsN9C1x)z?-!rQQ_jn>gm-h?Wx5Y?1Yr>',
        'LdxUcCuQBZT@c0hN_iix{U6bko%GIT*3MHM=FxNCqW)a&BPAP+#asIWkkdT~Jl;Pi?Riht=Wpfod3`z6V@pxy+uwBkVGC=a+iU(5',
        'XpqMLq>P9hpp{X!1Zt(RsBAIGdrjP>OFBC9KLMnRvdvU^x(2N|=|1E$T0b9LUp_d<Fe5#N)}N75qG$YgB_}t-64gO*gK|aAdyDGx',
        '`0;wVYW&{jF)bnMp`R~)cfN+@6l!7jJ2I}zN~ZQCEqj=ri0CinR-dF5eU-hIwM;bW4YYeL0H0|9uc)#2BwM$v-h29AcAfsGou>b?',
        'w<MoXq9t^d+vkY4P_$~qsL0tx%k496|NY_eA3P#EI0b$0-Rjijh{f9v>~Gg<!h+I?x1TK9f=$AEw28`3_KxkIFi6h#WrM)9(B$$^',
        'B_BdDGU~Px3=W2A7v5`{A@z<J5oEfob7ai1Up$YA^S|E$82q*JEex26&v_qfFKe-)%k-|(-o~O%>!Vt}>%$FBXqQlUCD*3L32lA2',
        '-X0t0ueCNR@2zpk)OrU|DK!*jYLSH$nOSIKa`NW$kk+YweuJlmr+>d9Z&du0QMrpc;*agxIpE!WG+z&aNmW5gMwYQDHg<AMs?)})',
        'c1KUZ6#fbK)L=(%F*8+byHwtpW2Tak13i@nWi{AMVm*S}*?WjTim!@^ZPb6G&C-&*t<7$U7eciLQg0>a(XoH+_EELFfBJA`_fN5#',
        's#qJv@|k32A8J)&PphWMRPVP)D2Gg4yIx5)CAoz$D*llr7o8j*MsE1bEGLIf^SpdUwx{M(3xeM^yZXA?erje%>+!Pv)b=?Z<Ki8~',
        'N12IYMN&*;DlwtzVEwl3RprE=3G1vo@mFoXgOuQ1mSCApJ!kkuEE`$eX;5bs|In(zd{TsmR#1eRwqH|Y<5I9mTwj}U1jhx{$i8jt',
        'RmSQHP&rbE)qCJ(NOEqEOtU%>Y!OOs_$)%>`CdN5N!E2DS(Bo5*`w8-TGxkbPpcD3%W!5ag)=(%6$3b{le}-(5jxwjB;8zg1d2k*',
        'hO%c<ZP?bZnW|0P3{Bq3qDRf#UVTB#+}bp;O^tMN9QAR0SMllUuz{PBY6=|xT4!VSwa(ma7Ngj^4Ib>p@nS`sf3xV?1di>gvp_4K',
        'vK9tGiCn0Le@*ke^c$*roHfAQjNm)=M8EUm#^1tsKS^%Be26dh8IvP$56wA8=4ZgMWv@jKXO-iSYD>H=Qi;)-+#XTApqqUf*b+z8',
        '#O4KH)09eGZ5Y%+XbU~W*nV$g<DlSb)1*lXq3s4j+p{_hYhuN}O-g2f7SXzF2P<-ys}E-z=;UjV4Ch)<xRDP2!T{b7>DdKV>;Y3-',
        'Q$jM#Qspz0waA&PS*X@hu|6p{+p%Bh%Df%Q{9?6c$s5^Z@o1e4u_FwNs<T^M-cS5j#~%4a{!6W8WQ%uLMxNM6-!Sqm%^S97=0{qA',
        'sNXO$&50lVKa`iv99vy~w#nsvvqZ*Zv@ZMiDsNrZnPoWT27zy-Tr<<{DZZ59%xwxc*Np&18^A59ZC{WP0kL^Ow3ms^3wAAx{D@s&',
        'Wrvs=SD-un+998wAv!@;eu-zum#m$b;2=6A!Uui26vvc(xH~Iz_sX*79l8f(?CF>k+hxze{_PjFzr3;Aw^=Rv-RieTeDHBnpE1}Z',
        'XheKq@3w9G2F8uTk-mdR_H>EJ46(5Z$&7IAHgrg!y{TAXozyC%_n??D(>(0NDrfiJ_YapY9M}P`ZfF|TvsJ5}VWth`9)o?%9;MPz',
        'yQZ^sq^B7jy#3_tJ?zYgJ_pOolP9$aDU0`yw&WY(U#i5I>)o+Q$bz_Gr-rvpyx|k!n;sTi5YrHIgT;!rx!6Q&8DcBjN%>+#O#)G?',
        '8S>~*CZUZ!ZJ7R9K$GyCLifM|kJYP&d{dIJWq0OBYnv35xVaVPL;%&YgMl7>G6+3JvYl)R8Or)}b~6#giilv>KwE6WsEu4Z_bnr2',
        'vVGC8kxYU{4ePUyO?k`9qkn&Hu%SH@ekC3Ae9O(KVb6rxgk~CY5wbm#dTd3@nuW9_;agH!)z^eeq3P%Gp@w#=<oNhg8vl9SHm8*}',
        '#%<hPm<4^MtXJ%0f7Ye%OV6i%$~)IcKZ~w=*l?|M=_|IY<&*u2C6obBH@r}~zAfvT#@GI<QR~IK+S^w)ZkgukK6a{45AoIZwf}>0',
        '>zw|PVnt%xPQm}t#5LC0J+P<i>+-kA&1x#9;TF`|$DZdEIvuod<_)_^G4BD@pr+*yIb~`&!IP$}uh>?54n&_Vx_xU02U(wO@>`i1',
        'Xs}i1Sr#1Gf##{Rf-p?BejwZL=BLrt8R55$ck32WVc^58&@<w_bK090U7a1{(%ntPH)3NPj?EQY-YYhKxrcjLHN5L+vBKV7)QG>z',
        'CB--8m04DA$IY^B-}uUNw=%FvudeK>s+`~NE0*xqLR?GP>Y8OuQ+RhOTb8*v^_dnFHe+OBY_A;r6Z=tBd3jH<BDrM?==r;mJtNtU',
        'IFxcel}vc%qgf}A%usokq<SJz)3A6XRmf85n-}Nisl4O7?d=mrd-;thjvO<rtlGf%7-D&R&6SJ#X2vi27s6yaIxsf2hM95eAt#3(',
        'daboJPFg?dPmY7xf;Eu=zPga{oH8zADH6H~5wpF=zQX1s2Kjr3w`-l*WopMU2RiS%(X>^VMO2tqylwN)T$>Il(V5t!?Z}xU+YUQ1',
        'c=vBiMDslFa3^cCjEK~3NXgS*n<uN6mu{iOFi*=Yy}WkH`R6eQ^)uDakL1rd*OKRU$kUnZ9fXoWpq#7CTXQ-&^XlAf{jryl|G<si',
        '&CGIw?CcZ9dIrM?m|%kcA$T}P+quWsR3lQdwsY5Pv7&iELYxjAr>2c88tdcpeKkI2?~hMHUe^C#bn3D=ueA?B>~++%(?Rrncw}gw',
        'L#yw@Z8fxsQS;=ia<VF)bAlS?uFMB(?2nrnZh~{6H8;4Opl7Mx%&xIPEmqXnjApeOhFXgjp4Dcb-ZE>!)@jiKi$#C*?^5u#S&Or^',
        '8MG*Hsnw<hK3bPS3r4H=DC=ede(zAu)YHwh+*O}yChqe;X=ak$UBIYJixK-Sgk1G7GOwx+Bk><*^#7<Ws{K-dyd$c7xnM8rt9*A=',
        'vEND8lsvRnJuRwjdE&1|@9v_`>mDwCE?)0=8rysIKKET<@!SbI6P~Q^!WYZLitA8eUV1I-xA|b3^>f;5?xu{-YSG>!%C#1|7>XX7',
        'Y`a%GJFnO71aszw_;jMD@4Jjnlzca^K8q8zeW!rJ<Mb^bz`wVu#z8Gm^y{=bWObfY(?6xgy;FZo(b9v~VFzs;6zw{A-l?orQ=K??',
        '{zUssG}+*J7u}6g^v~W?->lYiQ+?j4F{QO{LG;|1Jz6}8{uw;))#0_7KJU{$6J0K?{R*<P&jkq+<ehz-0Hf%99lX*2UO?gZNqq7x',
        'I(%80jU-)s0<ScH)A*=Fvxy5*cIwk!=V{|pc%=cnfJ$8nNgv=YavMrkbVDM8yohCLV*lFOI#|&%VgwF14eb^_xzxt9QIn9g(&&59',
        '<r!fPCFQTD{wh9JahPd{$Nhi>Y&*H@O!2z-_~sPP!yB2Cn(&{*iUko$UIYhU`W2(MM%7vq--h`!qdJwnwn<C%^cnY&Z<%=N9N-W?',
        'Jix}oR1XOp>T!f*WmLWmMo`1cdiN#xaH^UJkrKm%jn#0`)C#_Zg)4T&IOf9FI6z!1J`@*o(Hd9gR^?fUK&-()i~j}O8q>I;B)U7x',
        'x?>zV@9ei{=VBZ2>9b3ho{3N4wFKsbrX}-5<yq#aCI#NX#0oOU!CgZoC+y0y5aPo9hjFNQ87GJrPn{Ak!u2v2Egr(HHGgS-!o}i7',
        '#+NaJ_c<}0<KcZv<$e0e1n=+=`3rLxH@|oRhl_13TF)Nh8|KlliBUn8m5VqPa4BTAR=P(_EH-NJ18$y>$HkFv9+1??w<Xz*%J(T9',
        'sANz$`4T0C3+Omu4=FJh&+nm@oUO<ge_g|g3&pMklTa&HEB2arOS}T;hlv+oi^p)}o!?<}e*C6*3xBy)ypQd+j@rgt)SPBwHDTLE',
        'ku-^3&^m6O^3_f&IYn<slqHF>p;qoHE4H8b9MY}8CTg+ri$}QGM<3z9M~B5mc*4gwZhRbUv61l-Bk<RnpEtU>L7HtK%}YR|2FjO4',
        '>4$sSOI70-UTnyA(Y*To_Z`fx-#2U8l2*P{ddx-fr;!=`l9m++we}0+WMRM@o5|`@@|Qoks4>4TUUX~R*qe(M-yAF6>fO1#ci-~z',
        'UTk62FXH1#ld$D4Rli{K2@}MpB8yv}S-bX(c<9`Q4M5X0W*)PTKg_7uZuD#G3`&<CapSmkb>*ylN!dyst8n8!s93u6#li*8mo9z2',
        'K>Te<x2{W;cI&o?X={33{CCO}tUhmg9;>HJ5&!)&{K&?QN5YS7+NAwv`8940$hf`y#b)J`;gE%4+1P89=fBcae)UzA_*?tTwgrW4',
        '+ZHfK9=v??=;ebKfBhv|V{Y=`!IQ<RX#)m8x@po2W*_TPTe_?P=FGkwvu1s=bLOm1zK#wFj%pqf+#LV1anqVLn>KD-zh2zCKR-1!',
        'e}6%8GLgIxph;JtiNJRG1JTw9u|3cdkhv0&LyYu)yz=abnxpY(GrEYbp}XjBRE1U80y|?r9Dx&YCN9F=@nAe2&&EsfdVG~>z{D`U',
        'm|@H$W-D`<xzC7f44ca4vZZVvb~rnkozH&AZi1fwGxjU?HPkkHE|}}ZeZ*boDtHs#p7-U$`2;?LFXX%O1NpK1OnxhWioY!&!C7b~',
        'v=GvT_QGgkwQyDVOQlx%s}fb?RkKwGRHszmsD4rXp>|MbsXMBBsE4S>t7og1s@JPOQNL^u*PwNSyar<$tZVR#k)=@!qh3bajP4t~',
        'ZfM*vtYKlp(G6EN{G#E7hCemD)9|l`Z;S<FGh;_%U*mA&md5Rk2N^Fk-fR4WafOMoiKj`hNt{V*lRT3?CX-E0n|y2Xi^)Ti7madt',
        'u2Z9frVUM9Ok+(unD#IoW;)4qndwQ>yNx?E?%MdH#=9CHZ~Vl}$t>6`$*hCfAhS(od(AGHtIQqElgtavkD8yeu(8Ot7;CY?;#*5g',
        '%Y4f&mdh<aw%lcT#PYP2nU$TDn^k|Sy;eu8zOwqs>YmkW>m=)P>xtIutv|7TY7=OaZ!^K>Q=89hp4ocYcCanCon$-PcAM=v+q<@p',
        'ZL69jH`&$Xg&k+t#4gQlgxyNJt#)VazO(z)?oYcH_Kx<k_HFD7?EBb{u|Ht{rTqhk1`dJX`Zx@A*zfSz(aq7%G2F3_<9f$W91lC5',
        'a{R{ew&M#YwUeEbn^UM$vQvT6LZ@TSg0q=(Yv*$3na+D%8n`&SguAqL>Fe^b%Pm(YXeL^^wsr07I@0w6*UhenTz_(F<<`e-hr7MI',
        'w|k6xvU{O>U-t>_Gu%(PUv&S`!_*_mBi*AA+*ckN&p^+7&v~8~J*&K8yheCU@LJ$?(d!>?2k%tx!QLyqH+X;T{n-0oAJ(UlkDZT~',
        'PoPg5pCX_6KHvFz_-6W!@jdAKr=N?Tzh9JJvR}4eC%@i)!~G`v&G%dBchm1-Q}d=yO{1HpG~Lp4zrTrp6aOs#j{aNyKlMN1f6@P%',
        '|6TvT{i_1h0hR$S0sa9|0m%V50i6PR2YemyPhe<Zi@^SY+XA0AL(Qz3xi;(HY)Z2Q&DJ!#*6d!8b5PTu$e^U4te{~*YlF519SHg|',
        '=t9ts!Ht4_gX4m;gF6NH2p$(aJ@{zwsSx83%aGiV;*hx^OG8eFTnc#+q6zg34G8TMIy7`+=>E{JLhprHhqVjqAGRdyX1GsyP<T{$',
        'tMKgb(c#m>mxXT*|0^OPA|s+OqHDy!h_MkfBbG$0i`W?{M4CrBMfycXL?%XNMixbOj~o^`HF81Zy2w3|CnCR%yb<{z@_7`BN{`wU',
        '^=ost=7G%zH2*BxCb~oPmgxP_UqoMs{wexS^k31HF{&7g80VO#F_AGzF<CJaV`j!Ih*=l2C+0*f6Kft@9NRN?XzcmeA7gLCK8SrD',
        'hvJOmY~wuQg5qN1(&F;s%HsOQjf|TZ_f5QMynVcPd}w@&`1JVp@m=Bv#E*%e5x+QoZTyb-L-D8Le~Q1;!lXs>7JXZ+YjLl|KP{vL',
        'Rf1_kkCyQ*PbG4RL5a;1TPC(i%t`E+*d?(~;*P{W68}nkmDD_`Wzx8$=}GgGb|mdjI+k=Q=|a*ENjJd#k@PG{Og2i6O&*xMG5K06',
        '<5p>{rnWlSN=%7KNlnR3S)Oty<*$^=R8^`)s(tE|)DKcurEW>xpZZ1Wh18!?@1#CTi%LsKOH0d3D@*H}HZpBu+R?PLY0q1Gwk~Sj',
        'x%JN0pSC{U`gH4;ZG75<wVB;!S)1c+9<+I$j?#_OZPPu{gVJNu)6(<O%hLO%k4&GMz94-?`nvS3>3h--r{B#m%ZShDlQAS?OU8}1',
        'W^Gg2u50@yb9v_O%tM(!X5PrG$gIrbvYKSMW_8W#n>8tGcGktLE7?r8QMP%uQ?_4rM0R3!W_D3__w2#h<FmhLr*7A{omab%cJb{x',
        'x0~K>XS?e;LQYstLQY0bVNTbag*l(+ROI^PhUF&Y4$a+`doRy5FCedZUaP!zc^&e$=iSVA%=gU?&u^LEHa|aqYW{-!)%hFqzs|p!',
        'e>?wSK~BM%g1-tX+pF4Jw0Car-F`v)p9-56MiwR&W)*fU>{7V7@Rtrw9sD{(bV%%w*`c7r!440LqKcA>vWtp~dKL{WnpiZqXhqS+',
        'qCG{&i_RB)+cBnN&yE{9ZtwVO$BJUBV#nfM#UB^{TH;yKw`63=)RF}yt4lVP{Msp~)2L3TI$bE`N{vg?O7lv~O6Qa=E?rgnap{iI',
        'Tcv-L{#E*_R4Qv&W>e-~)~qb1EVXQ0*$Hr0%U+aQmB*H+mFJd^E8kN7dHJ*Q%FcnEi#kv1JdgHhuINm_!j`K`O=DjmR!zIO@5X7!',
        'mmAQ{Yxh@GKCK*L+(119;=<5P?h4tts4HKFRNM|p<olp})np_H+0u{tu>aG&J(|q*gFIqU0XGFD=&qpvABwti6VYV$XZU{+%K5MN',
        'd^^;T8!9d0x&rJBz=ony?qk%3=C>Krzlypt?a?x(F&a^Qm3A?WrMYn30oPe@JqFhm+(_id{)*gqYc!c}il%Vg0Jl6y_Xu2%Nw2v3',
        '(km54lLZy(s@@85HbMGV<b0S=G>18XGPwJw1pg#mV9e26{r~x?Q06aCi+5fpqY{H>A4p>aq%jrc!E++-0?%Vm7w!pa#igSd?gjGY',
        'a*-?d8=A&hK)e(*j9Z9GX`Z#+*Y$ri@N0%7!jRgX7m+h>iLAITh~b`t&yvKquD_txTri5_(omw-GaFGiauvQrNvOt;!1+X!3i-5W',
        'c<BJpn|gFc+Rs`@Uy}dXNhpAChI+FbkiE(urO|XJqJeCwBrzP?#e9r{z`HRKXaRU*hCvgV-_T61DZ~TMc%gC_t9D~6&?u%0nu7mD',
        'UP5bB$bJXBn2p+TlTZ@tgeJ15Al?=<l3R|72@RQMXgZgJQs45+Q75Jn@c04^=6p~ly$2Zo1kDu^&=|fY8pBOMv$zjYK9i5eF-d^W',
        'ugHOKjvQ12WzYQq{&!>z@AZQBQ@B2;7o?R%@0WwDE=SGbK7!kh`~e<L@0X)+b^&tY76FfAq$<7<ngz6t;#|=PknsZW9Y6+pu~*SZ',
        '#tHQ!_%qF<Rp3{E-^y4?tGE<25dNFM9Rqo?gfx4i@xoIyfWv4T_Ya!O1)xFPSkxQniXguN$TykW1aW;}u0igXqkQr|_Zi68RG@E7',
        'zZ`Al2B7&6e;LG|1<%(7b5y_|Lp{keI}gofw*if(DDep&|3qz>Z_yO+Ui?E;MtH>ZM)RPajpQ;=A;e8$D^UPjg;JThD4vZ*1K5_R',
        'HNJ^@;_WC6Uq!LF3MJz<s0H&B4J5o}2p=6#B(DNq*#oZ2Q7q&i#s#8Sz%7X0FNb=$9NK|O;OP#CFVkP<?<?sUp@DE2MRF&hUM)c_',
        'nFGj}n}G(x|1G%o$Q0rSLi}KOju1wpQgY8;K||p=m79#(!~cYb&EdHny^cXWYkKwrgjsLkR~Iyv9Z0@Jh6XF1Sbv&8;bT#2b_hym',
        '%%x7yegw0JP!3$1^5>w=KzpL{LHX<*XrFy#&rd;dTF+|WKk&?b>3iZEtU>LGH}a6ZV1=@X=N;fZ;vr4G6{NdJDrc@qTUaE0!^TRF',
        '+3nI}<`1cy4FFgHz!m{a0$8~8SY-?E?L+NVv2e`<`FMmfp$%?Na3Z+UG{Hl8!ogc2B#Ceh15Zv=H3Dy}Jez?h8Uo+SSMmijvg%H>',
        'oXOW6i6t@ewI~FfA*anzu6)J4iD!+_48A`;!i+@maJ|8GAXoGwcyevc-b6(JcLBNT2+zNP{O7>49d`@)!}Cz>73h@(a1Sm7jb>*f',
        'KPCt@<KwUmrSBA=Wiga{pY)iNnO_R!{0wMONmlKF-{#VFCJ+^Cc|qf!l<u;PrF%NwECW9f-X99SmS-)`xPofFvHrDRDL-EWAG=78',
        'g_H0;@Q?$#a|zO8Zi&nrf`b-E`s|1D)eOp3C!lshe|T>bzyYRO1M$|wd)Dv_yi|3p@$4t_mmdUp^#U5qf&75aI(oo8|FiP^qYPt%',
        'tKmWv@})AN_18d-Aud}5UU??+@|IsCUx^F}<B_)jv-AHdr`mLAooNhmK;Hd8ddONzPner<?JYe~B|;g&GxKEpxR&(PQ+R`|sGI63',
        'w8=-HZJmXNfo$}wOU9sGjpM!1O1LL}A++)Na9zzcLWOXBLu3s4n{IG_wKka}eIk{iP#9Ba<!6yp3T4~`WxS+U#!@KbCMe@2Z5eg(',
        'dr1$O6Vjuayss&1`ZA_3S8vHejb|-C34Qc=NchXGL?Qe}`I*pCm<D-BGF=5SR*_I16N!phKh(LVjOp{bzN=2&{;LcOhrVVwiZB|F',
        '@{LQN+}Y4~caokme?b3w6#ABIl%a!VLwhe@+bY*a%5^ZgvfY*TI8?eq+GH5#1?%vGwi%vFbok8#et?XTwwjCwb$l8qJz`b>UV$Kk',
        'XW{-S+)tIBXk||qZ>F@5IV#<SH1@Hkayz2HKLz-GfPV_``|s1X)W-k1<e+BTQuCSiAuZtfEIj}I_A~8Uf<Ug*A;0c#Ki6!l-qm*1',
        'Y;SAIPtCT8_CsDMlPiS#iQt!k|D;AcwNGw)2)(s(%Vhw6_1X)<!~ZI4FfX_Q^MWgIy${#>bi4&~0PU5IeNB)F{~L1T%wZg;LY&UG',
        'LB{M>7<Ze)d}TQuSEO;@qI6|^LdKdQWKO3JtNawY&XUTfRk_9uYOG_D6;6)Z%p6BNLVPIyA<^WJuk25#J7caNcOr6OC(5^UYUlz#',
        'v<|t5lAMQn1yAhpw4_)^3*AW++6OcA3k;-BXKDVHkmm-t&V?ie=+(y~NiY%21uMZ`a2BG4I3Zmq6gmongrUMHVXQD-m?TUWW(f<0',
        '#lkXSg|JdsFKiS(5q1j)grmae!YScv;alN~a8vkGc<o~9V(H@O;^yL2*{-rfWxvXCl}jtvR&J`?UAedNMCFajTa|Yz|E!cWEi}(G',
        'RpL-_p}1PSBR&_aL`jk;&g%qYp|OIED`1l#WC%q<nI1ONgqakZr4*ZWfXy}qo8!U>;jC~$_+GdH*!=C%(8a>VmSU4#Sy0)%a&+a)',
        '%8vk>EtPvJzo`77@@D1j$_JHIns|!MKykRZNL(Y{6<;XWpvtVuww39XDV0f;iIoYJtpSn^z6eswBOdMxAnhyyr(rwr`SLy7!FZ~&',
        'cV)NAj#@7MHt)C1Di}?v1mTVFx9~u?Bisi6lW;~jEqp0_4stM@JIx*A(zsOMBc#F=afQIQxj3!nyo}i9VvMVw84u;^LPICn%MkJ(',
        '<D)$JGx1Ccqol`VpgMj`wBCILIVq;(n}EK<o>+xG$Bt+v_Q&?v0d2%C=pcHEop1}b9Gl=woPutl>$o`%MQ1_m4&V^<FZu>O#O`Pl',
        'l*g9t&AOmq6pA8H6iTEvr_)g;>VSHnUZ@Z1hX$Y_XcC%&rlA>V9@>evqEFBU+yVWGK0sU09&{9)L_eVW=n47@Jx4FlOY{bbSdE)t',
        'HynV2a5Bzg3)v2A5!;b1L)+0dv;%#NB=kMnj}y@mwiDWggVAwx2wR~?=ru0Ffv6H4!>%|NXP`x>3fp04v=3WiYxEdfU~`-fy}v2c',
        'ng+-UYK}S7F9+lgqhBxNixOa@9E;*nGgN@eQAcP$x}j2}L0zDR4M(HU7&HqFLJ!b~XaQP?HlxL8Jz9eHqBH1AbPAnD73gpD4|;~a',
        '#R4|Q4X_cm#SPJC*ajEkd|ZIrv&Fa-TY{J2<#+{NiC5v(cs<^LKgOH!7Q7X2!#nXVya(^YpW=h~Fg}Wp<InL4d=h_&PvO(}Jidr8',
        ';p_NUd>7xxPw+qZC4Pk~vBa>9F=NhHGFFT=W5d`p?u-ZH$#^rNOgPhuNo86y=}cRu9h1xCGaZ<YObJ?nmZN28DY}d<VIS;=n_^$=',
        'g}s?Frkv@_^kB9!+nDXlZsr%}CUcAVjrpCq&D>?~Gk-7-n1^f_8^K1hQEYQInvG#&**LZZo4~eY6WJs-nQcX_#*kL7J87{vVkwU9',
        '>(hG-RNtlCGZ5#5a?irt<h*iExKgFu^T+|)D)$21hspOKk+wJ+C4r0$1l|k=iR*!SqM;y>At(r>G92#m0MZSvgrB1!#$b@ia){F(',
        'U@*pjcYDMCUFcga(J;971lVAZS{LL)QyU7Y4*|Jo298KIi7^ayMuG6w07$JF#4d*zL#f1fgQu?WJQ(1^-^$as7CaImmA>#!D7Zkt',
        'I|!~Zkh|<+YI!3-_fBo4s{XGVEk`eSCUE`q>q|3*Vo%<4f%@D_Df1Ae2ZEo*3u4vw!w5r*Yf5`4ZJWl>J~Rz{Yt)A9n*!A_%qMf<',
        'z5uO)`}Jr)ga^<e2#=sIAS5jUMn9r!5a&Aj8SZbO-ypmVtpi4{z%l4w47p(zn?Ps^bsj@o5e8udv}_o+#4Qnn6QTXWI1SnX3~gB^',
        'gxR<l!V+ADSX_>~z<pQT6Jqwli{X9=-U{$-cpKbr$2;JDC%ypp7x6{7zr<J~4E4(jF;K_C;XaaqyqK0uOT;pX%nZ1n$;^a*XE6uh',
        '{vdM@?hi3gF6J<ddl}{!a}4f}GiTxc91HlcS!@Af*!FB^xbF&MT})|bgo>bk@X09|dB~-FaCv7G(q+`(z9_NB;BI|TV&C$iL~6*m',
        'fbREG&Tvpmv=A2PqyNlM-UwX=4jG8N`g9xIANkWTgoaW5$_MvB3H|!?>w_}nU0a!#NZI3u_vI@GZKe@wNO{X36VwQJ?Sf8QOtsi*',
        'an+Kuw6yfF9BsM8a--#^mZvR$w7jdNp$2aR_;^q`NF|yOwEZ>KeYe04YCGvk<W`t`!$8}o$n9~0Uf)sxy$flXOF>S%fQ(o7G=o4&',
        '<yLyUuFsi;=0Lx*5`B)&f-K!4pB%)Y(4&ED;7qm?TTWXb1{v`=fY%91vyON)o?EA1gcmW#@j8$Y#Q{gZ#D|&r%xY#Ixv$lqXTE3E',
        'tQ)kLIqXn&G&`5w!0x2(A^R_9&TZ8>fAHh=p0#cl_fqFhgRkO^_{O|7pU4M+C#$;{=z^gv0&t6P${2WLiW&orEr80_3YDD==uFy4',
        'LfJH+Y7x9&43zBzv@HYbb_VKpr}XU&6z)sgSwp&RhaUZ;()XQ#TKfcgaMFJZ(1VlSuOak;wlKEvBs~lCGQQBy_(Q+m0w>^P+OKzH',
        'i`f#O`ef-6c9%}!Wa%%qgLDhZF<CkXlDtFu3w<Iz#}3kcDA{D`DVi$10Qh9-Z}bDqA38`6p`FJP)FbHx65y(WYXi6%!POY9CU7-{',
        'Yh$>Y!_@+=R&ce3Yohosl*$r)Al*a@z%K;982pFe*Mr{-evkAB?FD~OvPMVYdJ_DX;7@@+4gL)Hi}2rl=@C|gZvfs1d_(ZY;BCP>',
        'N!Hj6u1(<@Al=7#(nFjNz5slC@P*PtworP&b_8DxzC?NeWweJdOu7$cJPEbOUaEkne^77mlchX>ZUl<-20BcZNZZLNRQre2erYXE',
        'hWcp&_l-ar8i6!4qBXWLcnk2>;0H)o&?M<Dnj$?$Go)Y9Ebw!{3uvJ9C#3WUX#EeGDcwV}!OsQH!#jK6ohK0gTZsQH#Q&DYe@1gA',
        'yqO6+nhky~cr{?s7A=wHqetRl^jPx2Hqv^azyX{fU4s&wBNzi-9E|ELU}R_pqfkru|28e}p8#zHP-B3a0`w7-hCq9RH-(hf!P6y3',
        '{UllmZ)}vlQBpBSyQLpMIv%1&(k{sL66AV9N%<N`*h5O&CqT_dfWtk&KM^P}1pILDqri^=KOQ{!=PCU26#jV%|2&0%p29y*L1wgh',
        '?Ex<B1%DL$N$_8SKL!3Y_%q<|OAlz?ZgBNPjj<PaZ}2|geZl*I_XoZhK}wGyrN@xcV@T;Sr1Th4dJHK&hLj#dN{=Cx$Iv7B0T+0v',
        'i<VFu&7n40Lw&S{8fgu6{t49Y7vNta7m&w^AdeG49w&l4P6T<J2=X`)2T2QXOKAby0a|lRLl$bUDacF+JeffbEhHD<%>d~dg1QX(',
        'HUMt~z9D#H@V0;l2X$g9$o^E2@2Mc$Q{jyth|H<8NjusgtYHLlMJ@(o1l@dCx<l-O=zeDCy?(@gxN>(-x&l0u@Bburfb?RmlwQ<;',
        'ya4~4gydejEd3=-kT%f2l#z-=PC3n#PD{t6bM$$e0{^uZEdJ5E|GOGDX(akT{iGFjz@Nw=w1x8XeR+MqmY9Fn{%+m)8^Aw%Pds1L',
        'mX`F7bmu+auc!n5tiJeiPIci<f2K<OrFG`~uy$#B2fp{E-$BOiLAyhq{(xHXK5w1S-F*SlehtFw(gx{}bU}J75zRq!X@_)Hx=ndz',
        'LZn!(2~vd=0Gu|LE&^;FDZ8{rx+^V)aJ{rw`an7&9j5y;jiuevT##6W=hA+e2EYd|;Cf@pRyr;<u9sf5(udedqq{8Vzi>%@L;6C$',
        '+!S(6E0u4{=6hrHh)6tyPbi%p(|sx=Q|&$Z_XmCdk;1><-{Yl5()abIdFMai|F}B*pUVG4FO_Tht!Ry3V<rBy|1qBz_0~TkiNk3e',
        'Z9O|%T|e~Wi^p|uMrmsR4S%G&j?bhq(%0e>$&BtjYGeLP!xqvYDTdy)Q{cqH-5JXDQ?!M?Y=GGwgQq&`y^I~y;U~~%+>-u~`3sN>',
        'b;PIilTPcNf23vq9%|tM2(N4DL@X86`o1#_e|c9Q_J8Ut?nz(O`~QRgA<csSGg|LaBelI)t#t$_DjhR`Ux87@z<TF3wsp*GYR(^1',
        '=sZU|hpcP1Gk~u5)IN{=fUf&3MjxW}Xfs{E+=~wC&44b#=<DtIPZR89Fav5z=RmDs{!=8+f7mj5{teSo5=Yi2iGG+0oGFx<tdkjm',
        'GljAn)44aH0C5cI2k1Q79PTZ^nF39$;NBXXDbU6iIRbs`;2#HY7C<K_$cLOVY(!U>c%Yd(gkIoypq(#-{@{4J;)UsoQ6r#VBIKV8',
        't|8Dd1yV`{XAAU9gETsT!$8p@h}n_OLW{vQ0_v8)eJ5~@fXbzCUk0ubP`e!NJA-QkRPP4&-N7{i>i0l4z<~i!zJcJJffGX@-{Ihl',
        'fFq+I90Se>I5Qr?N#N{(LsOu9)4-9j@C=~PEO5rau{jXV17`-DTL=_g3{C*<eF))tZ~|~~GlW~fsVFzCfu9F~LMMTnYT)NZxPJnU',
        '1HQh2Pz1*Te+3BD*ch?EUt0*9;3h}~d<N<Qul-OH;Br&A_lL246X13Lvcj!!E2ILh7orBh`66Tq+%JLqGT?t>;8-Ur0Roi(6Dk1$',
        'l>iee0Roi(6Dk1$l>k`=1S$b0R00Gl0VY%e1S$b0l=m=i0;i@Dz<@kB!aZ3JWKbA5HRY}o<sw75sHR+GC>Pa~iwxzWnsSk$TvSso',
        'GL(yI)CC+vdC5>-s!=a+Fh2#SM*YAwM*YD#QtmdS+_hD>OV(H|6dsQNXQ6O<G&l=|-($g9C|sWa&O(p(HfSn1XGH?0gR@iQU?w<g',
        'MH*&<vsPqcE;wsF$xu<rP*KTHQOQtI$&h6NQ<-Rjj)KEfGBA~lCg>8n3>3W&4x^{w9H^AAR7zMXB`lQ^j!FqfrG%wYLiRV@V4g)}',
        '#fr*G6DliKI0!?1qbrz=a2_s1JTAgT$e|9oA*};h|1*a9fP=hhhr24*1<G|Jjl<DdKRInKLysy~QNEJ3OnRMJTWn^wa=j^ES##xD',
        'L}gQ6<0H?#YdjC8&uS&DKxmWWq4mh5F%a8Rxt^A<+)Vk(cT%o%mFr>U`mJ((sB04yJcLHd)rICR=TG#?{PmuB7`Svc{NK#S@PEf>',
        'y=MU-Cl|wK7dLv(fka`Tcx&_u_rSgITs)6q8J<xwE12`l*UV++N7jc8WV6_IY%ZIxyz?2k;+MD*YZ#81$V_EcvVKqskPLulq4Vf(',
        'aCxW&xYjgoSKJL>VmdM3Gq;%ste2dQ)EllYxDOtL2jdZVBp!uF<5_q<UVy)$`$4aj9LQc!3uZjzG?|&o8nKO7bJm*mV?)?bHdaY%',
        '5A6#`IY?f;aBtif_rt^Ra6B7I^&`HC|ALaK0D}k2pDe=)tR3sby0Kn*rDKpEl)y|{4RaadPE$wjuEBV1CctG!xNL^I;O=+`9*XDS',
        'dH4go5dV%JFtJP=Gn{$KJY!z6Q3__43B{TI%r)jZ^D~tHjsh=(s}JDWhhj=-K(MVzhi_%v2{jZtXwxTU(9(jGMV}f&>8@rqH3$tP',
        'TK@i&2X`nH<kCmIRsI!F;@`pPm;W@O0CJKp6IW;Xc}jupa#<Bz$a=1kA}tuWspX>|UDM;B&aVUaDa2yYPHc}`032Ex2)oif7KT8n',
        'zJNI`mAtY~*77lJAv|>DR3sQ*QgUM<HzLcOsEn>)Rw^?6J@cb1halU2di>Ys;iTk2o;g~sG0Zq>caY@Nh33_r%2+k;2qg^pHybEJ',
        'u$d3kSpf9;hH?^818!DJH^EUwIZG(l0{ElQ2e8yh4B=KCTqisvWgy&-Vq@Pbi<eRsj@Cnh<!yWi-^KSB4I|O9&i}1U-_}+tkTtBM',
        'zJA$;|9>u>zfwBVsx^c%^kS|+oNS1d1M%|Me71nd6&{1f;wQSd-1OcONJ*J(%xk8SC6cPbFAZc=`;ML7I}G{+J1Fv3Q!aU(Txz9R',
        '8zWGOLB>TKp)EWD-4Mf!2I(?ls(>Gjbn-6Ooj+^Ah>Vmua3iIUiK8;wRVg7kA8SiTsxv8kIbKC}<l&u$z+>|DbFwQKj8EeWC<N-s',
        'B@~UnVZu=i6UoG(4Zx!Wv<>RZQnUl=%YJl+?*2*4VKx)XTC`-wPRi;;wQDibNMz5lvG^SB?@DC9v!8SqFK2_;2;M;$4?Ig`)7VUI',
        'qv|r7r?OKuQrWRZst{E?Tc+x-xaq1zs*h9$+3sv#^<?fmJD45G{j6Tcj%TN`v)KjgGSy|(kE&-XkzLJhV7IEf3Jcg>?5FHe)iW-e',
        'D^UHY9>|_x&$1UGkMF@<19ubhvSaVEclEAb>_he;KbifTeaTj_RUD~$QJh*G#x(+GsXDIulC$GnAXP8UpYO!a<U+V8E}n~5UFO>G',
        'H~1S|vU~-6E~`Y9s5Vj?seU9laK&6_t|!-@8_JF5CUVocx!fXkiaLec&aL1+;x=;I`P*8j?#ka57I1sGgWPfMOHyL)GWR1XG54E#',
        'r+OcEpL?wC%01)$<wTw%o;TvZrh8-#!eHKw_u&KiFg}`3;8XPOGWZ<65L_p4UHRUm-26a(xXOzktBO)Z@slCE%+J)~&(G)Q16BJA',
        '*Z3v;N`4)`ncvCp;}7$n^WXBPA-`|=t27?O=O0if<1gPm;Ggo(g$1hq>dESv>iOy=>Xqts>dgdy^<n-E4EY7YSTGlC1t-Bn@DqZB',
        '2q9KT6w-uDA&=@%br<>ygN2d8cwwqAo2C>kEE84>8-%UGF5!gmsc@8PP+b(h7p@66g}cH-p^A9bW#Mn(rBJ0df_o^X3K|)e8YIgS',
        '9C>yD=cV!oDGPxz$5WTAYNN_l6{w0;omD--^;i9@8VYVSxQXBvffUXKw+P${a385Qs<x~4s16dTP@Sh7|4nrtDE2eGJ_dRG7o4c(',
        'h!m(BtF6^;Y6tK>>OggvI$E6o{LfJ5s0-Dd)Lp^#Ru5DUSC6fYUQmOcNJ}78N#PJ$Xtp5C{H#fad$j~DH@i>sA%q(wD}Y>)NDI?n',
        'GaBOjp^1m^w1#|t!COPp=}tpNa|}Wk8Wz*=YZ`h=4WWkKtAg6f{G=HOkn19ho0;2{=i%N#g8r60DK#ePiyEROKtkL|vH<uFaR}UZ',
        '7RSQ<{;CV`Zz~B}L;Slq4Z`_VYayIgMRX3rD-S`qr^*xJoY0t&G&L}WXU<k_fG|Ze0?2yFjOOx&<^mzxm*mU-K*Io%GP|1QyPbsS',
        'E6JF)HccRYRTZJn{i^l=sT4;+IztFnSP&;cI9{9%|K8OAw#-)=XbqVp30h_RuS9e#tTm|s8Ktp<`&Wb-_<`gDxlGf<K)6N&Z5ne*',
        '<4HqTk|U+3Ck@MJ_zelsOG?j^G`ysN+?iXIP7vOC17$(is-BQufQ0NWDS`5rq<m2FgL@lE1@OI9XK6@AliRC)g76s$u^kNyXn2~0',
        '>`qFDtt7;O1ZQ?5rCtXr5C2G}kn)nMZy@}r3T77gOCmG4mu48m{D^Y1gqHoj<Vh(&N;g#FM)WxdWtf|qrV!fF`<@ybc<T*m)!8nZ',
        '?ht-T@4wUZg0PH)csfn(oMr@s-;-Rh5ydTAN`e$pC0B|k(OC(ly~Bsna0rh}WS-buGZ;cgiRifaN`wn3ls?{)GlY3!4^l1>W)0X&',
        '%BOIWf)n8p_M>tBk#<4YkfzXEf>{l=rFr$I+&rTpd7YrS*l0G99YD=yB6}KtxKE;S8c_^eONdCa1|*g}q9GbQcWM05Bt*w)c!&Ip',
        'N73*p$&0a|^xs5w6(O`HwMl^xDSbfGw<96*p=LLc!>Sb!{!+D())zwI2NYsYY5~4TIdy^7=&KTp6mS=c`5;1Dz}bw-FbO3|fLs_7',
        'LVdTP@<}*lMQRFjfy()LQX+tpI`WL>wUeg*hJ?%^N~^oH6sM}zKsc^yDL^zdW*$xXOHzxO<pe{31BOf{t*skK>6l+MeIWdr<b}Ht',
        '8ODFpvag`>xt7X63dLuQra7emgy;|2=G?FPo3=dv06dxI=t{#v8eX8a^-Gd6dQIb3QcAv3@E53jG6YYW@<xp%<u$2ypHb;OMk#Q!',
        '5^59kd*zpq!YNu?jcFK0!+|usN$c-fT60Zl7)iq+H2j@}Fp45}QVMV}S-YfqRtPV_%V0#c0<R?B2f~|RPO}y7K$cXi%9d(RHKA`3',
        '{jgm0>aB4o8A0|2KM?$I@MFPG20s%#v7@&H{7Ufaz;6b>6Z}5#WRL7~@MIPLYw+Z((^c>{z~2T>*2kZMCueit&><$qFSzTZ#p4LI',
        'O<2+<9Op=z9Kex!)Pp7Ryn`jw@Z(6k-JX@mJ$nhlw(NZfQ`uJ#X0T^StzrMB5YkdzW4|NXX&h<CIQAl`#Vkp|iET!D5cVX5t=Joo',
        'ZyHZzz6DRdQPq>ngL`Z4Q=ku7$0O^q@!;BkD*)FM978q0p|21~>k_L<(?JfY$n5?Re*_>-!~zHQ7Jh+yb83y_q-sHROu>@WWB!~D',
        'azG&U@-}}~$JNnz;(z2S15kPNOvcT@bllvWj+>+DxOqJtH*clm=IwOce4CD&|6hAo9-UQ@?7OR%yRZfj7}lt4vWA=xmH<XJAwmc-',
        'MvNR;61GeTnXt$p+h9~=cnm@Y5dje}@-PA-I*78$Ad4f&5_t&7AdC@w;6q@Lf$*y8`|{-rgU-Bj=8yNsoIccP`u45cx4Y`sRrjmv',
        'l#tylZvT7bY31cY^y6ZjQ>v4SSG`nUm2Bp#f$Axhp)ysDny9Ae)$VmqtC?z+nyd2FVzu0xr3zG`+MqV6t!kIrrw*zjbwZs{#p<HE',
        'qHcIa-Wha-tgfi5`DuQduB{vBMkZP}*HJoFch=o>Z=I-9bebNbN9Zh_t0(DcI#18m^YlD-jb5mi=oPj=uhkpzAH}9^w2Sp-EZec{',
        ')}QJ_`j|eY&+7B~dwtCo>YMteLATZx+CuMycfz<>Dr2c(H<&tDBC#|vElsrPV7i(fCcz|`R5Qp7Gowtl8E+<=8RmI2$ILg2oX$>^',
        ')6Fb1tIR5=x1VO#nDyp;^O4zU_L$Gi5p&#}Hh(eSnak$7DKRDZhC^(T)z)_tY*ky+*0cB7X12ABv7P)8ewLr>Px7bvc{bjk=P&e^',
        '*j~1;O|}E=Q#Qk9+8jI4PO&qw%(8QBK4&Wwl)i`EfMpYwt#%jZP!!n{_KYpI7wr{$!x4^kDmvAi+D-#3jj%MwvB%;Ybi>jcOCpvO',
        '2iL$E5<V7xc3>>d2q(+QbtXB}oIGc?GtXJ*EOAyitDUvZMrX6L-P!GY>Kt;8Ij5Yn&Uxp1=bCfV1=n>eyEWW8Zlv49ZRtk49o(*N',
        '4>!S0a#P(w?l5<ho9&KwC%ZG;=iNE(e0Pz%%w2<J6_z#bdiQ-SAGtf-Jy<?-kGRL()9zo~@7&Aob+^RRo{y!fSJSJ9<vy>O*BVQV',
        '*U5{=(#z}XC3^$Cr@Ra=6HAUa(VODU^k#W;y?k%6x7;i63cU^9CU2{^%iHH2L}|pC^v+-@_AYu?yc@potzXfv=GXQc_>KJLev}{U',
        'clNvaz5PT#h1Qe*-DSQ|m$tFSR<Wm%9yHsLR%2u=Hl*i@z7(sKx+(3v`W)GW2JIr~qv>z%KB5oNUTJBkUZ}4kon%oP(X+x9SL?k(',
        'I@e7x_R+Q){f%CxyUc^wbBiWE74%@N#~HMx<!ekoAqtA=uefE1A=|8g=`Nxmm}Ya!y@Uv+x;;#`t9cu#)U9Zz);JQmRa5-eVag{_',
        'J9Q(npfuXgvV$gnXNyLBQJ?7HWMiAR7(s6Gu0VdlF|&i-+R$CA^^e$|fO~_KJ<M>jRW#WH{p=v5srE^vV=UT-kY#%!ebQo2c*?$m',
        'bg(TX>%y5%G1H8g&Gk2okE>{GY9-Rs3hiq99!1rv$wZ;Z!Kc9Q>|&&?><;FPEvAw|tHaDQ^?A&5Xv{Q1Q`bR@6rRxZyTsReDDqiA',
        'b0Sk6@;FvG!|>`L^Ch+nbvDu}sQo+-WZUOZ9p2J&$(GY(P5o%llagc2OJuj0_DBm2TFbEBjA2Uh(AY#_-ElG){b`<$*-WJ8O@E~K',
        'o8@@5u_jCVsKx&BgdxlPV?&nnhh`Mnc<N>3(-X)CS)s{Fd`qM4CnuV@#FOb+qR^1(Ty9xyy3!6r6e5`pV!EB%H?!nujaI8%ph)&6',
        'Y4kc_g`L1W$sgHas*uI*kR)_*$hOSaspNwYmHTKm)E=`9>9=Mt?N~O4Df$JbrZW}kN{gPd{Dboha;s&@0`H}dkt8sESZ_co9g1s@',
        'b!ZNknvapbWcHEYVMtRRH7IHNf=NeOWYMOR$)q)~Llq%?S7jqT;=F+ECF%&>#c%XCg|nm@^31Q-*bdXV-cgf;JV!dE*1;o4qXB0g',
        'UZ=N)4-~EVCiNltiB3P{^Lv+8gmPoi<FH+LT{4-{n{G3gX&!7WS-*z-uQntVxUbW``vM$AdXi*7);8qJ)H65(@*6{avNh!oJg8Qa',
        'cV@_bZp*Y6ulvJHo2VJ&wW(`J@6%)xJixS(rtbhPn7*&^Eu^vZ#qXx2F+?~t!s-r5$Pu%Zyh1~^v}dE}n}#Ke;hg>((jPVIvWz#h',
        'MuSZp<68~xiPEl?lWn02HOW9Pmj7&`C^J;gV!g;`jcCsG12z%sL+x0+t3Y3-Ib<}bZ3-9@h(?$Jj0^*aKh24H!z?7fz>+81m8rGl',
        'hjt(et@TPqQCh*Cn&z`D*-40{HJOqh)Zfzgga!H<`Ei83a-5w>_LikDA^X*vgolhFZ!v1UX;4D!7Q#1sh~7@k)@YmJ-iz;L-z4c$',
        'xs1#e37HX}(ON|d1*vn*Tw<)gfGznfpCvrnTb5=inrR!Nh~bpi;Sf(5OBA*+GUVyEDZ4|HXEa)8l6G5a$+P#<Oxn6ghnZ8TI}!Fc',
        '(xs*ZIjpl^5I1@v&RY@hFDtP99s3#W4x}UMbMpdyn=+NjjwF47zM4Gu$9W8?yrx3aj&PfiKSLw_z#L5|n$Gp#a{U86mA;MWc8qT`',
        '$zL<{9c;f*yemDx+SHv<i~QCJyn~$7PY{A>Tv>=vq)mB1xca1yrf&``Gdq~xVk!D1*B24;Ad(Q?US?Tp&2$OT;MemQj$kPJdj4}>',
        '-9P<JJm~o}5gpj;NeFs9lg0DwtHkm*F&*+-7V$TMw31t;EpY2L+xI0U*wTo-ptRc}boC~GJ-B*XhB&;;zVnh>>$rtv?4_`)vzdPi',
        'ec$6q3@JLv`2V9{^l!h0JlZ<AcAZ5x(OV>n6p<!|h!G-7<cdjRn#dEg#XPZ4ED<ZjYOz*q6r06%v0LC%puW9M)FE7na42u<Lx-%q',
        '%8c=C>{Z^KsHHI@R%5a_5YdlPj+n{9G2C|O8%(S-ite)XwYa8}hV4gr$7pWoH_P*eeyu%8_KvLV<gjjCr_3@#+TYzFo2aKlxo)G#',
        'e!)HSy^uuI7TwC}!9OqSTKJ->s7a9=GU^uDQSl|@nP0d?@;0huQXoG}c7mjG`C&<%A2x^c!{%^)*qTuGS9SjV41zggff%|9!R)Q7',
        '6orgEYKVLA$?~z}9+C7bAKvBhJ%)Q%!hY5HJek|nE7V>wbfxrJelQo!;uDB}1m}0P7S2OyJtIYC_v93$y^<e_6p8)%_eGlcP%6^o',
        '{{4`qBtH^KdVzc@h4}bkoC4#d>lN;L{myG{xob7kqsOI-=hKH~jTdv$)5qk9`59TGM~X!m*<;6uWg|vs4;8CM<PJ?2Yer{}9xpcH',
        '#i3#|)9p<6WMxknBMxSbP0tcVV}@sp7ALSFSDYD3?Zvso?czA%e`Va5^c-<x+_<Q=QjEu|(w~q$I$c%`%dfkKX<&7X2%cn&v<i<u',
        '5a&q$k6;=&1t5SOfwc@|))B4CG{C!ZC~Hr$%yHC8Fs@AUU+oM$y?#(C8i=kk3o=!qI;d{wX1c4Mp?B&c<D)N5|3=tPogQv;cf1$t',
        'EvhiL!qy6BD>kW^S805u`IWAO^7{ZW>*1hW_(7?tiKDHH&uJ*`MTu`Jnu}H<N<`x>(Oz^EaiWWOP&_2Mi(ZHTeMNuqIAXvc+(Cwj',
        'VIl*uU=+$imdHkw%oXFs(}?&}#B{{TJVgFkh!1lRD_<4c#9qX|Gw38$k+HI~>@IuDBsExN{6EYzESs$<`2G-_D}%H3>d($syMN(~',
        'N#s)+#hNc(!SR+wvThEGWIf1P`Nf=<AINF{AuK;XJifISN4S<|>-Ur?flPNfFqF3+h;S`?`(-$K71YiM)ahD?IgLd#(E^d8pGXn|',
        '#RS|1rivM%UOL#Xbp6Wu|J3>g`26(x(Gz~n&nUls62~UX$xNB71Z09%p4L5l7JqII|8+B{L)o<P!I`u;H%B>l>k?;6Ukd9}FRV`y',
        '_qis%!2e%kG&mgAvjIoa)nIKog7w3^dr<~;55|bYGAb;S*9PO1dvMfa>6$jc`Pz??e@dK{Mn=dEvMb7df=uSD=8AY_PdTlKW33o^',
        '2Zuf7bSDn`mG&YI`@^_*ME<4RLnFIgz)>U*1~JbK#gOC#ahHA}sercAN%6HfC$+39+siJpj~t+ehmE8&&Ib4S$sTpE$0y>rI4z~D',
        'A!B4G*;hVFS=`uOUi(YO?*;E%FCK%vupj;e2jFu!3SYoi@Mri2{u}-Wu2M7&#TO~5jPg_^6`}4?zfkq%G<C0PtR7IUR9n?v#i@AJ',
        'P4$p}lzU+?jFF$iP+268!vy%XJPlLj*YceF2A+}U<wbcJeg)Izb$Js6yaLbCUu%I%P#yjQX2FY48|uMpuoPa0H{d>43I7RApc%wK',
        'dw2=*IU@A2V1(01IVvp5DO-+}Id_e9qIjCAe;4mGUoPO&EB+LWd^*Z!Se%ky$v?|8e3nJA{FcwNxEgk%1;s&mcbscc9~wYIh=hB~',
        'op8Z%Qem@PEZV_g_!7>-dAI~Wz+aV8j;f%lsOqYgs;e5xJaxZns#>TvDn@lwUEocXAU~1&VXQn1Bjqu90-l!N$?xSA7ze+RH)II_',
        'yb8~O3l*R$jD+98Z=nvXgVxX~?72DkT#C>-1I2fb%#zc?VzWA^&QPFw%H47wWXnS^OnxE1go*Nkyd-~sT$mw$<k-K(=%G2_LuD8S',
        'b74080@lC-Kyi}oa8)wHR^%V8MCm$&ufosIJI#6{oaJ_${U6Y?SdI>^jVRobG2^cS=nD$g9xbg=RAjK0T3W+*aPhs{a7<l*QvpW?',
        'N8(-=gS*;5+{f}6dzK>hEaBR+Q0@0zV?Rv3Ms>Lg>B?Zsa&CEpTd4Od+?%kk25M_0B6v&Yn$H{;GPjqR&mz8eG2iiOklQM5CvJ;5',
        '{x84Wi7hL`R=hR(cF)RLPz55O6+}Zvh=UREOIQo<!f`kWU&A^07A~khblwB@oi6im9joH{wT2GRiBDRgQ&x_lw9@%1bhZkeo<cj2',
        'gJX}vJ^ed55dDZ_eEtUGnViO0M8Bddl%p8cC2$RN5bpxls>!?{7FgE#*7&SvU@9WOI!2jRh$RUjjGP?ANV+pxzKF7S117@LFav&r',
        '^7kA{-XYYf;22T5FQI(@0M}5r0JY1ae)*_jl~Kp4p_Y-f*HsNvLzMUXQR17T%(tM`=ke0HF%)wPgfnID#hEE0uoIO&3G0o;qw*$d',
        'Aa=($aUmYY@_S6jkv)qen}x_mn0f&5^efi++lZD;5ieWdJGQ}R#i=BftWr=G%bwLK^sH*MD)JC&{!gq7uL#M(G{(pFh?Aca0tdOx',
        '4b6HjvcM5ZV-Sm%BC@VTTxDb@6jxnTeDIFNp*w2xyAs+SP>n-!Rl3VoVR@ku(P*d_jwrC=!7-Kf2aXpG?@xWhd+{DCXUZ3#CA5X%',
        'kO^<Y2KX;Hp?a%+w8!3gcj(>AbU%@8g_dESY9Cyd!Ky#fy=nl`{fbcFKv<6i?vcbjlDS6;_rUeP0}qN==2OvbT`M-B9rY=;R>L`J',
        'fEICc&gkk4n@Ef4JQcYRJ|ZoKEu^imjkFbZkQT$or2X(mq_6Wh7z5yaL?e>GIGKT0R><Lm7IGxgg7QcKAMi7_GT-gYcPI1R#e8=&',
        'Us|_&LU=PGR2x-VqqAVR=E!*5J746TaDj>ovm+`B1&MMKaI`J-qtltft|GR;6EFbVpM;0uF}(T+^ttP4PCYpr;qNSuyPYOg&gP01',
        'HRaoKoqR{WE8mlUkRQko<rcY(P8Nd%=nF~EACe&j2EyZz27~Bt;M|R8*q9Q7qe`W53Gu%w+2_g',
    ),
}


if __name__ == "__main__":
    main()
