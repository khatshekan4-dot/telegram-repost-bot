import os
import re
import json
import time
import html
import hashlib
import logging
import tempfile
import difflib
import threading
import csv
import io
import copy
from datetime import datetime, timezone, timedelta

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("raptor-news-bot")


# ============================================================
# Configuration
# ============================================================

def env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} باید عدد صحیح باشد؛ مقدار فعلی: {value!r}") from exc


def env_float(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} باید عدد اعشاری باشد؛ مقدار فعلی: {value!r}") from exc


def split_csv(value):
    return [x.strip() for x in (value or "").split(",") if x.strip()]


BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
TARGET_CHAT_ID = os.environ.get("TARGET_CHAT_ID", "").strip()

SOURCE_CHANNELS = [
    x.lstrip("@").strip()
    for x in split_csv(os.environ.get("SOURCE_CHANNELS", ""))
]

# ============================================================
# AI providers — v5 runtime-configurable order
# ============================================================
# ترتیب عمداً ثابت است:
#   1) Groq = موتور اصلی
#   2) Gemini = کمک اول
#   3) Mistral = کمک دوم
# Failover بر اساس availability/cooldown انجام می‌شود و یک provider خراب
# نمی‌تواند کل چرخه را متوقف کند.
GROQ_API_KEYS = split_csv(
    os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY", "")
)
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b").strip()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

_raw_gemini_keys = (
    os.environ.get("GEMINI_API_KEYS")
    or os.environ.get("GEMINI_API_KEY", "")
)
GEMINI_API_KEYS = split_csv(_raw_gemini_keys)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip()
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    f"models/{GEMINI_MODEL}:generateContent"
)

MISTRAL_API_KEYS = split_csv(
    os.environ.get("MISTRAL_API_KEYS") or os.environ.get("MISTRAL_API_KEY", "")
)
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small-latest").strip()
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

AI_PROVIDER_ORDER = ("groq", "gemini", "mistral")
AI_PROVIDER_LABELS = {
    "groq": "Groq",
    "gemini": "Gemini",
    "mistral": "Mistral",
}
AI_PROVIDER_KEYS = {
    "groq": GROQ_API_KEYS,
    "gemini": GEMINI_API_KEYS,
    "mistral": MISTRAL_API_KEYS,
}
AI_PROVIDER_MODELS = {
    "groq": GROQ_MODEL,
    "gemini": GEMINI_MODEL,
    "mistral": MISTRAL_MODEL,
}

POLL_INTERVAL_SECONDS = max(30, env_int("POLL_INTERVAL_SECONDS", 180))

# هر provider برای هر درخواست حداکثر دو تلاش داخلی دارد؛ بعد فوراً provider بعدی
# در زنجیره وارد می‌شود. 429 باعث چرخش به provider بعدی می‌شود، نه ده‌ها retry.
AI_MAX_ATTEMPTS_PER_PROVIDER = max(
    1, min(2, env_int("AI_MAX_ATTEMPTS_PER_PROVIDER", 2))
)
AI_MIN_REQUEST_INTERVAL_SECONDS = max(
    0.0, env_float("AI_MIN_REQUEST_INTERVAL_SECONDS", 1.2)
)
# خروجی Groq به‌ویژه در Structured Outputs باید فضای کافی برای JSON کامل داشته باشد.
# مقدار پایین قدیمی مثل 1400 دیگر نمی‌تواند سقف را بیش از حد محدود کند.
AI_MAX_OUTPUT_TOKENS = max(
    2400, min(4000, env_int("AI_MAX_OUTPUT_TOKENS", 2600))
)
AI_PROVIDER_COOLDOWN_FLOOR_SECONDS = max(
    20, env_int("AI_PROVIDER_COOLDOWN_FLOOR_SECONDS", 45)
)
AI_429_MAX_COOLDOWN_SECONDS = max(
    60, env_int("AI_429_MAX_COOLDOWN_SECONDS", 600)
)
AI_FAILURE_RETRY_MINUTES = max(
    3, env_int("AI_FAILURE_RETRY_MINUTES", 10)
)
# Retry چرخه پردازش محتوا: تعداد تلاش محدود + backoff نمایی.
# بعد از رسیدن به سقف، UID برای یک بازه قرنطینه می‌شود تا یک خبر خراب
# باعث retry بی‌نهایت و مصرف مداوم AI نشود؛ بعداً دوباره قابل تلاش است.
AI_RETRY_MAX_ATTEMPTS = max(
    2, min(6, env_int("AI_RETRY_MAX_ATTEMPTS", 5))
)
AI_RETRY_MAX_DELAY_MINUTES = max(
    AI_FAILURE_RETRY_MINUTES,
    min(24 * 60, env_int("AI_RETRY_MAX_DELAY_MINUTES", 120))
)
AI_RETRY_QUARANTINE_MINUTES = max(
    AI_RETRY_MAX_DELAY_MINUTES,
    min(72 * 60, env_int("AI_RETRY_QUARANTINE_MINUTES", 24 * 60))
)
# Backoff داخلی درخواست‌های موقت provider؛ با jitter کوچک از retry هم‌زمان جلوگیری می‌شود.
AI_PROVIDER_RETRY_BASE_SECONDS = max(
    1, min(10, env_int("AI_PROVIDER_RETRY_BASE_SECONDS", 2))
)
AI_PROVIDER_RETRY_MAX_SECONDS = max(
    AI_PROVIDER_RETRY_BASE_SECONDS,
    min(60, env_int("AI_PROVIDER_RETRY_MAX_SECONDS", 20))
)
EDITOR_ENABLED = os.environ.get("AI_EDITOR_ENABLED", "true").strip().lower() in {
    "1", "true", "yes", "on"
}
TEST_MODE = os.environ.get("TEST_MODE", "false").strip().lower() in {
    "1", "true", "yes", "on"
}

# Batch مشترک برای کاهش مصرف و جلوگیری از burst.
AI_BATCH_MAX_POSTS = max(2, min(10, env_int("AI_BATCH_MAX_POSTS", 8)))
AI_EDITOR_MAX_ITEMS = max(1, min(8, env_int("AI_EDITOR_MAX_ITEMS", 6)))
AI_INPUT_MAX_CHARS_PER_POST = max(
    900, min(4000, env_int("AI_INPUT_MAX_CHARS_PER_POST", 2400))
)
AI_RECENT_INPUT_WINDOW_MINUTES = max(
    24 * 60, env_int("AI_RECENT_INPUT_WINDOW_MINUTES", 24 * 60)
)
AI_LOCAL_DUP_THRESHOLD = min(
    0.97, max(0.90, env_float("AI_LOCAL_DUP_THRESHOLD", 0.94))
)

# امتیاز ارزش خبری: خبرهای زیر 50 اصلاً وارد صف نمی‌شوند.
# بازه 50-64 فقط در صورتی منتشر می‌شود که گزینه 65+ در صف وجود نداشته باشد.
MIN_NEWS_SCORE = max(40, min(70, env_int("MIN_NEWS_SCORE", 50)))
PREFERRED_NEWS_SCORE = max(
    MIN_NEWS_SCORE,
    min(90, env_int("PREFERRED_NEWS_SCORE", 65)),
)

# تصویر عمومی نامرتبط می‌تواند اعتبار یک پست را خراب کند؛ پیش‌فرض خاموش است.
ALLOW_GENERIC_IMAGE_FALLBACK = os.environ.get(
    "ALLOW_GENERIC_IMAGE_FALLBACK", "false"
).strip().lower() in {"1", "true", "yes", "on"}

# ============================================================
# Dedup / event memory — deliberately stronger than v3.8
# ============================================================
# 24 ساعت حداقل فاصله برای تکرار عادی. اگر Railway مقدار قدیمی 360 را داشته
# باشد، این نسخه عمداً آن را به حداقل 24 ساعت ارتقا می‌دهد.
DEDUP_WINDOW_MINUTES = max(
    24 * 60, env_int("DEDUP_WINDOW_MINUTES", 24 * 60)
)
DEDUP_SIMILARITY_THRESHOLD = min(
    0.96,
    max(0.78, env_float("DEDUP_SIMILARITY_THRESHOLD", 0.82)),
)
EVENT_MEMORY_WINDOW_MINUTES = max(
    48 * 60, env_int("EVENT_MEMORY_WINDOW_MINUTES", 72 * 60)
)
EVENT_UPDATE_MIN_NOVELTY = max(
    50, min(95, env_int("EVENT_UPDATE_MIN_NOVELTY", 62))
)
EVENT_MAX_UPDATES_PER_24H = max(
    1, min(4, env_int("EVENT_MAX_UPDATES_PER_24H", 2))
)
EVENT_SIMILARITY_THRESHOLD = min(
    0.95, max(0.50, env_float("EVENT_SIMILARITY_THRESHOLD", 0.62))
)

# ============================================================
# Fixed publishing schedule — unchanged
# ============================================================
MORNING_POST_SPACING_MINUTES = 60
AFTERNOON_POST_SPACING_MINUTES = 45

# ============================================================
# Event clustering
# ============================================================
EVENT_CLUSTER_WINDOW_MINUTES = max(
    30, env_int("EVENT_CLUSTER_WINDOW_MINUTES", 90)
)
EVENT_MAX_ITEMS_PER_CLUSTER = max(
    2, min(10, env_int("EVENT_MAX_ITEMS_PER_CLUSTER", 6))
)
EVENT_MAX_OUTPUT_ITEMS = max(
    1, min(6, env_int("EVENT_MAX_OUTPUT_ITEMS", 6))
)

# Only used by a quiet-hour window; normal news stays in the queue.
MORNING_HOUR = 8
NIGHT_HOUR = 0
QUIET_START_HOUR = 0
QUIET_END_HOUR = 8

# حجم رسانه.
MAX_IMAGE_BYTES = max(
    1_000_000, env_int("MAX_IMAGE_BYTES", 10 * 1024 * 1024)
)
MAX_VIDEO_BYTES = max(
    5_000_000, env_int("MAX_VIDEO_BYTES", 45 * 1024 * 1024)
)

# وضعیت روی volume دائمی قرار بگیرد.
DEFAULT_STATE_FILE = "/data/state.json" if os.path.isdir("/data") else "state.json"
STATE_FILE = os.environ.get("STATE_FILE", DEFAULT_STATE_FILE)

# پنل v5 از Bot API و کتابخانه استاندارد استفاده می‌کند؛ وابستگی جدیدی لازم نیست.
OWNER_TELEGRAM_ID_RAW = os.environ.get("OWNER_TELEGRAM_ID", "").strip()
try:
    OWNER_TELEGRAM_ID = int(OWNER_TELEGRAM_ID_RAW) if OWNER_TELEGRAM_ID_RAW else 0
except ValueError:
    OWNER_TELEGRAM_ID = 0
    log.warning("OWNER_TELEGRAM_ID عددی نیست؛ پنل مدیریت تا اصلاح متغیر غیرفعال می‌ماند.")

STATE_IO_LOCK = threading.RLock()
# قفل مشترک برای تراکنش‌های منطقی state؛ قفل نوشتن فایل به‌تنهایی کافی نیست.
STATE_LOCK = threading.RLock()
BOT_VERSION = "5.3"
CHEAP_CLASSIFIER_ENABLED = os.environ.get("CHEAP_CLASSIFIER_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
CHEAP_CLASSIFIER_MODEL = os.environ.get("CHEAP_CLASSIFIER_MODEL", "openai/gpt-oss-20b").strip()
TGSTAT_API_TOKEN = os.environ.get("TGSTAT_API_TOKEN", "").strip()
ALERT_QUEUE_HIGH_WATER = max(10, env_int("ALERT_QUEUE_HIGH_WATER", 80))
ALERT_NO_POST_HOURS = max(1, env_int("ALERT_NO_POST_HOURS", 6))
ADMIN_POLL_TIMEOUT_SECONDS = 25
ADMIN_PAGE_SIZE = 8
MAX_ADMIN_HISTORY = 5000
MAX_ADMIN_LOGS = 100

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; RaptorNewsBot/5.3; "
        "+https://t.me/khaatshekaan)"
    )
}

FOOTER = "#raptor\n————————\n@khaatshekaan"

TEHRAN_TZ = None
if ZoneInfo is not None:
    try:
        TEHRAN_TZ = ZoneInfo("Asia/Tehran")
    except Exception:
        TEHRAN_TZ = timezone.utc

MORNING_MESSAGE = (
    "🌅 صبح بخیر به همراهان کانال\n"
    "روزتون پر از آرامش و اخبار دقیق باشه 🫡\n\n"
    + FOOTER
)

NIGHT_MESSAGE = (
    "🌙 شب بخیر رپتوری‌های عزیز\n"
    "فردا با اخبار تازه در خدمتتون هستیم 🛡️\n\n"
    + FOOTER
)

# ============================================================
# Constants: geography / priority
# ============================================================

IRAN_TERMS = {
    "iran", "ایران", "تهران", "tehran", "isfahan", "اصفهان", "تبریز",
    "tbriz", "shiraz", "شیراز", "سپاه", "ارتش", "نیروی قدس", "iranian",
    "iranian forces", "persian gulf", "خلیج فارس",
}

MIDDLE_EAST_TERMS = {
    "iran", "ایران", "iraq", "عراق", "syria", "سوریه", "lebanon", "لبنان",
    "israel", "اسرائیل", "palestine", "فلسطین", "gaza", "غزه", "yemen",
    "یمن", "saudi", "saudi arabia", "عربستان", "uae", "امارات", "qatar",
    "قطر", "bahrain", "بحرین", "jordan", "اردن", "egypt", "مصر", "turkey",
    "ترکیه", "iraqi", "syrian", "israeli", "levant", "middle east",
    "خاورمیانه", "غزه", "کرانه باختری", "دریای سرخ", "red sea",
    "عراق", "عمانی", "oman", "عمان",
}

SUPERPOWER_TERMS = {
    "united states", "usa", "u.s.", "america", "آمریکا", "ایالات متحده",
    "russia", "روسیه", "moscow", "مسکو", "china", "چین", "beijing", "پکن",
    "france", "فرانسه", "uk", "بریتانیا", "britain",
    "germany", "آلمان", "japan", "ژاپن", "india", "هند",
}

MILITARY_EVENT_TERMS = {
    "war", "جنگ", "attack", "حمله", "strike", "حملات", "airstrike",
    "bombing", "بمباران", "invasion", "تهاجم", "clash", "درگیری",
    "conflict", "تنش نظامی", "missile", "موشک", "drone", "پهپاد",
    "fighter", "جنگنده", "air defense", "پدافند", "shootdown", "سرنگونی",
    "intercept", "رهگیری", "navy", "نیروی دریایی", "military", "نظامی",
    "army", "ارتش", "operation", "عملیات", "offensive", "تهاجمی",
    "ceasefire", "آتش‌بس", "escalation", "تشدید درگیری", "mobilization",
    "بسیج", "deployment", "استقرار", "base", "پایگاه", "explosion",
    "انفجار", "casualties", "تلفات", "front", "جبهه", "battle", "نبرد",
}

BREAKING_TERMS = {
    "breaking", "urgent", "فوری", "خبر فوری", "عاجل", "العاجل",
    "لحظه‌ای", "همین حالا", "در حال وقوع", "breaking news",
}

WORLD_VARIETY_TERMS = {
    "europe", "اروپا", "africa", "آفریقا", "asia", "آسیا", "pacific",
    "اقیانوس آرام", "latin america", "آمریکای لاتین", "ukraine", "اوکراین",
    "taiwan", "تایوان", "korea", "کره", "north korea", "کره شمالی",
}


# سبد محتوایی نسخه 3.5:
# ایران 40% | جنگ/درگیری خاورمیانه 20% | تحولات خاورمیانه 10%
# روسیه-اوکراین 10% | نظامی/تسلیحاتی آمریکا-روسیه-چین 20%
CONTENT_BUCKET_TARGETS = {
    "iran": 0.40,
    "middle_east_war": 0.20,
    "middle_east_developments": 0.10,
    "russia_ukraine": 0.10,
    "superpower_military": 0.20,
}
CONTENT_BUCKET_WINDOW = 20
ALLOWED_CONTENT_BUCKETS = set(CONTENT_BUCKET_TARGETS)

SUPERPOWER_ALLOWED_TERMS = {
    "آمریکا", "ایالات متحده", "usa", "united states", "u.s.",
    "روسیه", "russia", "moscow", "مسکو",
    "چین", "china", "beijing", "پکن",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "from", "at", "by", "is", "are", "was", "were", "be", "as", "that",
    "this", "it", "its", "after", "before", "into", "over", "under",
    "خبر", "گزارش", "گفت", "گفته", "اعلام", "اعلام کرد", "بر اساس",
    "در", "از", "به", "برای", "با", "که", "این", "آن", "یک", "و", "یا",
    "اما", "هم", "نیز", "است", "هست", "شد", "شده", "می", "شود", "کرد",
    "کرده", "خواهد", "روی", "درون", "درباره", "بر", "تا", "را", "از سوی",
}


# ============================================================
# Prompt
# ============================================================


# ============================================================
# Editorial terminology / spelling guardrails
# ============================================================

# این واژه‌نامه برای جلوگیری از ترجمه تحت‌اللفظی اصطلاحات رایج نظامی است.
# فهرست عمداً محافظه‌کارانه است و مدل اجازه ندارد برای اصطلاح نامطمئن معادل عجیب بسازد.
PREFERRED_MILITARY_TERMS = {
    "air defense": "پدافند هوایی",
    "air defence": "پدافند هوایی",
    "surface-to-air missile": "موشک زمین‌به‌هوا",
    "surface to air missile": "موشک زمین‌به‌هوا",
    "air-to-air missile": "موشک هوا‌به‌هوا",
    "air-to-air": "هوا‌به‌هوا",
    "air-to-ground": "هوا‌به‌زمین",
    "ground-to-ground": "زمین‌به‌زمین",
    "cruise missile": "موشک کروز",
    "ballistic missile": "موشک بالستیک",
    "loitering munition": "مهمات سرگردان",
    "unmanned aerial vehicle": "پهپاد",
    "unmanned aerial system": "سامانه پهپادی",
    "unmanned combat aerial vehicle": "پهپاد رزمی",
    "infantry fighting vehicle": "خودروی رزمی پیاده‌نظام",
    "armored personnel carrier": "نفربر زرهی",
    "main battle tank": "تانک اصلی میدان نبرد",
    "armored vehicle": "خودروی زرهی",
    "electronic warfare": "جنگ الکترونیک",
    "electronic countermeasures": "اقدامات متقابل الکترونیکی",
    "electronic support measures": "اقدامات پشتیبانی الکترونیکی",
    "signals intelligence": "اطلاعات سیگنالی",
    "intelligence, surveillance and reconnaissance": "اطلاعات، مراقبت و شناسایی",
    "intelligence surveillance and reconnaissance": "اطلاعات، مراقبت و شناسایی",
    "close air support": "پشتیبانی نزدیک هوایی",
    "airborne early warning": "هشدار زودهنگام هوابرد",
    "air refueling": "سوخت‌گیری هوایی",
    "sortie": "پرواز عملیاتی",
    "interceptor": "رهگیر",
    "fighter aircraft": "جنگنده",
    "fighter jet": "جنگنده",
    "aircraft carrier": "ناو هواپیمابر",
    "guided bomb": "بمب هدایت‌شونده",
    "precision-guided munition": "مهمات هدایت‌شونده دقیق",
    "counter-drone": "ضدپهپاد",
    "anti-drone": "ضدپهپاد",
    "directed-energy weapon": "سلاح انرژی هدایت‌شده",
    "laser weapon": "سلاح لیزری",
    "radar": "رادار",
    "active electronically scanned array": "آرایه فازی فعال",
    "active electronically scanned array radar": "رادار آرایه فازی فعال",
    "beyond visual range": "فراتر از دید بصری",
    "beyond-visual-range": "فراتر از دید بصری",
    "rules of engagement": "قواعد درگیری",
    "ceasefire": "آتش‌بس",
    "deployment": "استقرار",
    "mobilization": "بسیج نیرو",
}

PERSIAN_SPELLING_REPLACEMENTS = {
    "تایید": "تأیید",
    "تائید": "تأیید",
    "میتواند": "می‌تواند",
    "میتوانند": "می‌توانند",
    "میتوانست": "می‌توانست",
    "نمیتواند": "نمی‌تواند",
    "نمیتوانند": "نمی‌توانند",
    "نمیشود": "نمی‌شود",
    "میشود": "می‌شود",
    "میشد": "می‌شد",
    "میکند": "می‌کند",
    "میکنند": "می‌کنند",
    "میکرد": "می‌کرد",
    "میباشد": "است",
    "می‌باشد": "است",
    "بروزرسانی": "به‌روزرسانی",
    "به روزرسانی": "به‌روزرسانی",
    "درحال": "در حال",
    "درحالیکه": "در حالی که",
    "در حالیکه": "در حالی که",
    "هیچ گونه": "هیچ‌گونه",
    "به صورت": "به‌صورت",
    "به روز": "به‌روز",
    "هم چنین": "همچنین",
}


REWRITE_PROMPT = """نقش: دبیر ارشد یک کانال خبری فارسی در حوزه دفاع، امنیت و ژئوپلیتیک.
هدف: تولید خبر طبیعی، دقیق و قابل انتشار از ورودی خام؛ نه ترجمه لفظ‌به‌لفظ.

قواعد اصلی:
- هیچ نام، عدد، تاریخ، مکان، توانمندی، علت یا جزئیاتی را که در ورودی نیست اضافه نکن.
- ادعا، مشاهده، گزارش و واقعیت تأییدشده را از هم جدا کن. یک کانال یا منبع ناشناس به‌تنهایی تأیید مستقل نیست.
- از تاریخ نسبی مانند «امروز/دیروز» فقط وقتی استفاده کن که زمان منبع آن را پشتیبانی کند.
- اگر ورودی‌ها متناقض‌اند، اختلاف را پنهان نکن. درباره دود، نور یا صدای نامشخص علت نظامی حدس نزن.
- فارسی روان و حرفه‌ای بنویس؛ نیم‌فاصله و نشانه‌گذاری را رعایت کن. املای نام‌ها در کل پست یکسان بماند.
- نام مدل‌ها و کدهای نظامی مانند F-16V، B-52، MiG-29، K9 و AESA را ترجمه نکن. برای نام نامطمئن شکل لاتین/شکل منبع را حفظ کن.
- نام کانال، یوزرنیم، لینک و نسبت‌دادن خبر به منبع در title/body ممنوع است؛ فوتر را کد اضافه می‌کند.
- از شروع‌های کلیشه‌ای، هشتگ اضافی، پیش‌بینی، جمع‌بندی ساختگی و پاراگراف پایانی «این موضوع نشان می‌دهد...» استفاده نکن.
- اهمیت خبر را فقط وقتی توضیح بده که ورودی داده مشخصی برای آن دارد.

واژه‌نامه ترجیحی:
""" + "\n".join(f"- {k} = {v}" for k, v in PREFERRED_MILITARY_TERMS.items()) + r"""

انتخاب سبک: breaking (فوری و کوتاه)، brief (۱–۲ پاراگراف)، standard (خبر عادی)، important (زمینه بیشتر)، conflict (درگیری)، geopolitics، defense_tech، fact_check، comparison، context، iran یا analysis. تحلیل فقط با شواهد کافی.

ارزش خبری news_score از ۰ تا ۱۰۰:
اهمیت نظامی/امنیتی ۲۵، اعتبار شواهد ۲۵، تازگی ۱۵، اثر ژئوپلیتیکی/عملیاتی ۱۵، ارزش آموزشی ۲۰.
زیر ۵۰ حذف؛ ۵۰–۶۴ کم‌اولویت؛ ۶۵–۷۹ قابل انتشار؛ ۸۰+ اولویت بالا. امتیاز را برای هیجان یا حدس بالا نبر.

حافظه رویداد:
- خبرهای یک رویداد را ادغام کن؛ update فقط با اطلاعات تازه و معنی‌دار.
- event_key اختصاصی همان رویداد باشد، نه کلید عمومی.
- post_indices فقط ورودی‌هایی را شامل شود که خبر را پشتیبانی می‌کنند.
- media_post_index فقط به رسانه مرتبط با همان رویداد اشاره کند؛ اگر مطمئن نیستی ۰.
- تصویر عمومی یا تزئینی انتخاب نکن.

ورودی‌های خام:
{content}

تاریخ تهران: {current_date}
حافظه رویدادها: {event_memory}
حافظه سبک کانال: {style_memory}

فقط JSON معتبر طبق این ساختار برگردان؛ هیچ Markdown یا توضیحی اضافه نکن:
{
  "items": [
    {
      "draft_id": 0,
      "relevant": true,
      "duplicate": false,
      "duplicate_reason": "",
      "is_update": false,
      "novelty_score": 0,
      "news_score": 0,
      "event_key": "unique_event_key",
      "title": "...",
      "body": "...",
      "post_indices": [1],
      "media_post_index": 0,
      "verification": "reported",
      "urgent": false,
      "important": false,
      "needs_editor": false,
      "style_mode": "standard",
      "length_class": "standard",
      "entities": ["..."]
    }
  ]
}
"""

EDITOR_PROMPT = """نقش: ویراستار نهایی خبرهای فارسی.
پیش‌نویس‌ها را فقط از نظر دقت زبانی، خوانایی، یکدستی نام‌ها و تناسب ادعا با ورودی اصلاح کن.
- هیچ واقعیت، عدد، تاریخ، مکان، نام یا توانمندی تازه‌ای اضافه نکن.
- شکل استاندارد نام‌ها و کدهای لاتین نظامی را حفظ کن؛ ترجمه عجیب نساز.
- ادعا/گزارش/مشاهده را به خبر تأییدشده تبدیل نکن.
- نام کانال، یوزرنیم، لینک منبع، عبارت کلیشه‌ای و تحلیل پایانی ساختگی را حذف کن.
- اگر پیش‌نویس فاقد ارزش خبری است، relevant=false بده؛ news_score را واقع‌بینانه نگه دار.
- رسانه و post_indices را تغییر نده، مگر اینکه ورودی به‌وضوح ارتباط دیگری را ثابت کند.
- ترتیب draftها را حفظ کن و برای هر مورد draft_id برابر شماره آن در ورودی بگذار.

متن خام برای کنترل واقعیت و پیش‌نویس‌ها در پیام پیوست آمده‌اند.
فقط JSON معتبر و بدون Markdown برگردان:
{
  "items": [
    {
      "draft_id": 1,
      "relevant": true,
      "duplicate": false,
      "duplicate_reason": "",
      "is_update": false,
      "novelty_score": 0,
      "news_score": 0,
      "event_key": "...",
      "title": "...",
      "body": "...",
      "post_indices": [1],
      "media_post_index": 0,
      "verification": "reported",
      "urgent": false,
      "important": false,
      "needs_editor": false,
      "style_mode": "standard",
      "length_class": "standard",
      "entities": ["..."]
    }
  ]
}
"""


# ============================================================
# General helpers
# ============================================================

def now_ts():
    return time.time()


def now_tehran():
    return datetime.now(TEHRAN_TZ or timezone.utc)


def normalize_space(text):
    if not text:
        return ""
    text = text.replace("\u200c", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonical_url(url):
    if not url:
        return ""
    return url.strip()


def sha1_text(value):
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()


def parse_iso_or_now(value):
    if not value:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.now(timezone.utc)


def split_sentences(text):
    text = normalize_space(text)
    if not text:
        return []
    return [
        x.strip()
        for x in re.split(r"(?<=[.!?؟。])\s+", text)
        if x.strip()
    ]


def truncate_text(text, max_chars):
    text = text or ""
    if len(text) <= max_chars:
        return text
    part = text[:max_chars]
    # نزدیک‌ترین مرز مناسب را پیدا کن.
    candidates = [
        part.rfind("\n\n"),
        part.rfind("\n"),
        part.rfind(". "),
        part.rfind("! "),
        part.rfind("؟ "),
    ]
    cut = max(candidates)
    if cut < max_chars * 0.55:
        cut = max_chars
    return part[:cut].rstrip() + "…"


def safe_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "بله"}
    return bool(value)


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def normalize_persian_text(text):
    """ویراستاری سطح سیستم برای خروجی فارسی؛ محافظه‌کار و صرفاً زبانی."""
    text = str(text or "")
    if not text:
        return ""

    replacements = {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ۀ": "ه",
        "ة": "ه",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "ٱ": "ا",
        "،,": "،",
        ",": "،",
        ";": "؛",
        "؟؟": "؟",
        "!!": "!",
    }
    for old_value, new_value in replacements.items():
        text = text.replace(old_value, new_value)

    for old_value, new_value in PERSIAN_SPELLING_REPLACEMENTS.items():
        text = text.replace(old_value, new_value)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\s+([،؛؟!])", r"\1", text)
    text = re.sub(r"([،؛:])(?=\S)", r"\1 ", text)

    # نیم‌فاصله‌های پرتکرار فارسی.
    text = re.sub(r"\bمی\s+([\u0600-\u06ffA-Za-z][^\s\n]*)", r"می‌\1", text)
    text = re.sub(r"\bنمی\s+([\u0600-\u06ffA-Za-z][^\s\n]*)", r"نمی‌\1", text)
    text = re.sub(r"([\u0600-\u06ff])\s+(ها|های|تر|ترین)\b", r"\1‌\2", text)
    text = re.sub(r"\bبه\s+روز\b", "به‌روز", text)
    text = re.sub(r"\bهم\s+چنین\b", "همچنین", text)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def key_id(key):
    return sha1_text(key)[:16]


def normalize_event_key(value):
    value = normalize_for_match(str(value or ""))
    value = re.sub(r"[^\w\- ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip().replace(" ", "_")
    generic = {
        "iran_news", "iran_event", "iran_update", "military_news",
        "military_event", "breaking_news", "defense_news", "middle_east_news",
        "news_update", "world_news", "latest_news", "general_news",
    }
    if value in generic or len(value) < 6:
        return ""
    return value[:140]


def event_signature(text):
    """امضای واژگانی رویداد برای dedup محلی، مستقل از متن کامل."""
    normalized = normalize_for_match(text)
    raw = re.findall(r"[\w\u0600-\u06ff]{2,}", normalized, flags=re.UNICODE)
    generic = {
        "خبر", "گزارش", "منبع", "اعلام", "اعلامی", "گفت", "گفته",
        "تازه", "جدید", "امروز", "امشب", "امروز", "درگیری", "تحولات",
        "این", "آن", "است", "شد", "شده", "می", "شود", "خواهد",
        "the", "news", "report", "reports", "latest", "new", "update",
    }
    seen = []
    for tok in raw:
        if tok in STOPWORDS or tok in generic or tok.isdigit():
            continue
        if len(tok) < 3:
            continue
        if tok not in seen:
            seen.append(tok)
        if len(seen) >= 14:
            break
    return " ".join(seen)


def event_key_or_signature(title, body, event_key=""):
    normalized_key = normalize_event_key(event_key)
    if normalized_key:
        return normalized_key
    return sha1_text(event_signature(f"{title} {body}") or normalize_for_match(f"{title} {body}"))[:20]


# ============================================================
# State
# ============================================================

def default_state():
    return {
        "version": 5.3,
        "_source_initialized": {},
        "_recent_titles": [],
        "_recent_events": [],
        "_recent_inputs": [],
        "_event_memory": {},
        "_pending_queue": [],
        "_retry_uids": {},
        "_last_queue_release_ts": 0,
        "_last_health_log_ts": 0,
        "_last_morning_date": "",
        "_last_night_date": "",
        "_last_queue_purge_date": "",
        "_last_posted_region": "",
        "_last_posted_regions": [],
        "_last_posted_buckets": [],
        "_ai": {
            "providers": {
                provider: {
                    "cooldown_until": 0,
                    "last_request_ts": 0,
                    "fail_count": 0,
                    "last_status": 0,
                    "last_error": {},
                    "usage": {"requests": 0, "successes": 0, "errors": 0,
                              "prompt_tokens": 0, "completion_tokens": 0},
                }
                for provider in AI_PROVIDER_ORDER
            },
            "keys": {provider: {} for provider in AI_PROVIDER_ORDER},
        },
        "_style_memory": {
            "recent_titles": [],
            "recent_openings": [],
            "entity_counts": {},
        },
        "_admin_settings": {
            "auto_publish_enabled": True,
            "emergency_stop": False,
            "morning_spacing_minutes": MORNING_POST_SPACING_MINUTES,
            "afternoon_spacing_minutes": AFTERNOON_POST_SPACING_MINUTES,
            "active_start_hour": 8,
            "afternoon_start_hour": 14,
            "active_end_hour": 24,
            "daily_post_limit": 0,
            "ai_enabled": {provider: True for provider in AI_PROVIDER_ORDER},
            "ai_order": list(AI_PROVIDER_ORDER),
            "ai_attempts": AI_MAX_ATTEMPTS_PER_PROVIDER,
            "test_mode_override": None,
            "source_channels": list(SOURCE_CHANNELS),
        },
        "_admin_users": {},
        "_admin_sessions": {},
        "_admin_activity": [],
        "_recent_decisions": [],
        "_system_errors": [],
        "_published_history": [],
        "_runtime": {
            "started_ts": 0,
            "last_source_check_ts": 0,
            "last_source_count": 0,
            "last_cycle_ok_ts": 0,
            "last_cycle_error": "",
            "last_posted_ts": 0,
        },
        "_telegram_update_offset": 0,
        "_paused_until": 0,
        "_pause_until_important": False,
        "_priority_pin_id": "",
        "_disabled_sources": [],
        "_source_stats": {},
        "_source_health": {},
        "_blocklist_terms": [],
        "_priority_keywords": [],
        "_topic_silences": [],
        "_auto_approve_rules": [],
        "_rules_gate_enabled": False,
        "_v53_alerts": {},
        "_weekly_summary_last_week": "",
        "_v53_classifier": {"seen": 0, "passed": 0, "rejected": 0, "errors": 0},
        "_profile": "normal",
    }


def load_state():
    if not os.path.exists(STATE_FILE):
        return default_state()

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            return default_state()

        base = default_state()
        base.update(state)

        if not isinstance(base.get("_pending_queue"), list):
            base["_pending_queue"] = []
        if not isinstance(base.get("_recent_titles"), list):
            base["_recent_titles"] = []
        if not isinstance(base.get("_recent_events"), list):
            base["_recent_events"] = []
        if not isinstance(base.get("_recent_inputs"), list):
            base["_recent_inputs"] = []
        if not isinstance(base.get("_last_posted_regions"), list):
            base["_last_posted_regions"] = []
        if not isinstance(base.get("_last_posted_buckets"), list):
            base["_last_posted_buckets"] = []
        if not isinstance(base.get("_retry_uids"), dict):
            base["_retry_uids"] = {}
        if not isinstance(base.get("_event_memory"), dict):
            base["_event_memory"] = {}
        if not isinstance(base.get("_style_memory"), dict):
            base["_style_memory"] = default_state()["_style_memory"]

        ai = base.get("_ai")
        if not isinstance(ai, dict):
            ai = default_state()["_ai"]
            base["_ai"] = ai
        ai.setdefault("providers", {})
        ai.setdefault("keys", {})
        for provider in AI_PROVIDER_ORDER:
            ai["providers"].setdefault(
                provider,
                {
                    "cooldown_until": 0,
                    "last_request_ts": 0,
                    "fail_count": 0,
                    "last_status": 0,
                },
            )
            if not isinstance(ai["keys"].get(provider), dict):
                ai["keys"][provider] = {}

        # Migration from v3.8 Gemini-only state.
        legacy_gemini = state.get("_gemini_keys", {})
        if isinstance(legacy_gemini, dict):
            for kid, info in legacy_gemini.items():
                if kid not in ai["keys"].setdefault("gemini", {}):
                    ai["keys"]["gemini"][kid] = info
        if "_gemini_global_cooldown_until" in state:
            ai["providers"]["gemini"]["cooldown_until"] = max(
                ai["providers"]["gemini"].get("cooldown_until", 0),
                float(state.get("_gemini_global_cooldown_until", 0) or 0),
            )

        # Migration to v5: provide admin/runtime fields while retaining the queue and memories.
        fresh = default_state()
        for key in ("_admin_settings", "_admin_users", "_admin_sessions", "_admin_activity",
                    "_recent_decisions", "_system_errors", "_published_history", "_runtime"):
            if not isinstance(base.get(key), type(fresh[key])):
                base[key] = fresh[key]
        settings = base["_admin_settings"]
        for key, value in fresh["_admin_settings"].items():
            settings.setdefault(key, value)
        settings.setdefault("ai_enabled", {provider: True for provider in AI_PROVIDER_ORDER})
        settings.setdefault("ai_order", list(AI_PROVIDER_ORDER))
        settings.setdefault("source_channels", list(SOURCE_CHANNELS))
        for provider in AI_PROVIDER_ORDER:
            pstate = ai["providers"][provider]
            pstate.setdefault("last_error", {})
            pstate.setdefault("usage", {"requests": 0, "successes": 0, "errors": 0,
                                         "prompt_tokens": 0, "completion_tokens": 0})
        # v5.3 state migration: add new control/analytics fields without removing existing news data.
        v53 = default_state()
        for key, value in v53.items():
            if key not in base or not isinstance(base.get(key), type(value)):
                base[key] = copy.deepcopy(value)
        for channel in settings.get("source_channels", SOURCE_CHANNELS):
            base["_source_stats"].setdefault(str(channel), {"fetched": 0, "new": 0, "published": 0, "rejected": 0, "errors": 0})
            base["_source_health"].setdefault(str(channel), {"fail_count": 0, "last_ok_ts": 0, "last_error_ts": 0, "last_error": ""})
        base["version"] = 5.3
        return base
    except Exception as exc:
        log.exception("خواندن state شکست خورد؛ state خالی ساخته می‌شود: %s", exc)
        return default_state()


def save_state(state):
    """Atomic write protected against simultaneous admin/main-thread saves."""
    with STATE_LOCK, STATE_IO_LOCK:
        path = os.path.abspath(STATE_FILE)
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".state.", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass


# ============================================================
# Tokenization / similarity / event detection
# ============================================================

def normalize_for_match(text):
    text = (text or "").lower()
    replacements = {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ۀ": "ه",
        "ة": "ه",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "ٱ": "ا",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^\w\s\u0600-\u06ff.-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokens(text):
    text = normalize_for_match(text)
    raw = re.findall(r"[\w\u0600-\u06ff]{2,}", text, flags=re.UNICODE)
    return {
        token for token in raw
        if token not in STOPWORDS and not token.isdigit()
    }


def token_similarity(a, b):
    ta = tokens(a)
    tb = tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def title_similarity(a, b):
    return difflib.SequenceMatcher(
        None,
        normalize_for_match(a),
        normalize_for_match(b),
    ).ratio()


def combined_similarity(a_title, a_body, b_title, b_body):
    t = title_similarity(a_title, b_title)
    j = token_similarity(
        f"{a_title} {a_body}",
        f"{b_title} {b_body}",
    )
    return max(t * 0.75 + j * 0.25, j)


def related_post(a, b):
    """
    تشخیص ارزان و محافظه‌کارانه برای گروه‌کردن پست‌های یک واقعه.
    قرار نیست صددرصد معنایی باشد؛ هدف این است که پست‌های واضحاً مرتبط را
    قبل از Gemini به یک batch بدهیم و مصرف API کم شود.
    """
    dt_a = parse_iso_or_now(a.get("published_at"))
    dt_b = parse_iso_or_now(b.get("published_at"))

    delta_minutes = abs((dt_a - dt_b).total_seconds()) / 60.0
    if delta_minutes > EVENT_CLUSTER_WINDOW_MINUTES:
        return False

    a_title = a.get("title") or a.get("text", "")[:220]
    b_title = b.get("title") or b.get("text", "")[:220]
    a_body = a.get("text", "")
    b_body = b.get("text", "")

    sim = combined_similarity(a_title, a_body, b_title, b_body)
    if sim >= 0.34:
        return True

    at = tokens(f"{a_title} {a_body}")
    bt = tokens(f"{b_title} {b_body}")

    strong_a = {
        x for x in at
        if x in {normalize_for_match(v) for v in MILITARY_EVENT_TERMS}
    }
    strong_b = {
        x for x in bt
        if x in {normalize_for_match(v) for v in MILITARY_EVENT_TERMS}
    }

    if len(at & bt) >= 3:
        return True

    if strong_a and strong_b and len(at & bt) >= 2:
        return True

    # پست‌های تکمیلی رایج:
    combined_title = normalize_for_match(f"{a_title} {b_title}")
    continuation_terms = (
        "جزئیات", "تصاویر", "ادامه", "به روزرسانی", "بروزرسانی",
        "update", "details", "new footage", "latest",
    )
    if any(x in combined_title for x in continuation_terms) and len(at & bt) >= 1:
        return True

    return False


def cluster_posts(posts):
    """
    پست‌های جدید را به گروه‌های رویدادی تقسیم می‌کند.
    هر گروه حداکثر EVENT_MAX_ITEMS_PER_CLUSTER ورودی دارد.
    """
    ordered = sorted(
        posts,
        key=lambda p: parse_iso_or_now(p.get("published_at")),
    )

    clusters = []
    for post in ordered:
        placed = False

        # ابتدا نزدیک‌ترین خوشه زمانی را بررسی کن.
        candidate_indices = list(range(max(0, len(clusters) - 12), len(clusters)))
        for idx in reversed(candidate_indices):
            cluster = clusters[idx]
            if len(cluster) >= EVENT_MAX_ITEMS_PER_CLUSTER:
                continue

            if any(related_post(post, existing) for existing in cluster):
                cluster.append(post)
                placed = True
                break

        if not placed:
            clusters.append([post])

    return clusters


# ============================================================
# Geography / priority
# ============================================================

def contains_any(text, terms):
    normalized = normalize_for_match(text)
    return any(normalize_for_match(term) in normalized for term in terms)


def infer_region(text, model_region=None):
    model_region = (model_region or "").strip().lower()
    if model_region in {"iran", "middle_east", "world", "superpower"}:
        # ایران باید اولویت خودش را نگه دارد.
        if contains_any(text, IRAN_TERMS):
            return "iran"
        return model_region

    if contains_any(text, IRAN_TERMS):
        return "iran"
    if contains_any(text, MIDDLE_EAST_TERMS):
        return "middle_east"
    if contains_any(text, SUPERPOWER_TERMS):
        return "superpower"
    return "world"


def is_military_event(text, event_type=""):
    if event_type == "military_event":
        return True
    return contains_any(text, MILITARY_EVENT_TERMS)


def is_breaking(text):
    return contains_any(text, BREAKING_TERMS)


def infer_content_bucket(text, region=""):
    """طبقه‌بندی محافظه‌کارانه برای حفظ سبد محتوایی نسخه‌های قبل."""
    text = normalize_for_match(text)

    if contains_any(text, IRAN_TERMS):
        return "iran"

    ru_terms = {
        "روسیه", "russia", "اوکراین", "ukraine",
        "moscow", "مسکو", "کی‌یف", "کیف", "kyiv"
    }
    if contains_any(text, ru_terms) and contains_any(
        text, MILITARY_EVENT_TERMS | {"ukraine", "اوکراین"}
    ):
        return "russia_ukraine"

    if contains_any(text, MIDDLE_EAST_TERMS):
        if contains_any(text, MILITARY_EVENT_TERMS):
            return "middle_east_war"
        return "middle_east_developments"

    military_terms = {
        "نظامی", "military", "سلاح", "تسلیحات", "weapon",
        "weapons", "defense", "دفاعی", "جنگنده", "fighter",
        "تانک", "tank", "پهپاد", "drone", "موشک", "missile",
        "پدافند", "air defense", "navy", "ارتش"
    }
    if contains_any(text, SUPERPOWER_ALLOWED_TERMS) and contains_any(text, military_terms):
        return "superpower_military"

    return "world"


def is_allowed_content_bucket(bucket):
    return bucket in ALLOWED_CONTENT_BUCKETS


def calculate_priority(item):
    """
    قواعد موردنظر کاربر:
    1. واقعه نظامی / حمله / جنگ بالاتر از خبر عادی
    2. خاورمیانه بالاتر از سایر جهان
    3. ایران بالاتر از غیرایرانی
    4. ابرقدرت‌ها مهم
    5. تنوع جهانی حفظ شود، بنابراین region تکراری کمی جریمه می‌گیرد.
    """
    title = item.get("title", "")
    body = item.get("body", "")
    text = f"{title}\n{body}"

    region = infer_region(text, item.get("region"))
    event_type = item.get("event_type", "routine")

    score = 0

    bucket = item.get("content_bucket") or infer_content_bucket(text, region)
    if bucket in {"middle_east_war", "russia_ukraine"}:
        score += 70
    elif bucket == "iran":
        score += 45
    elif bucket == "middle_east_developments":
        score += 30
    elif bucket == "superpower_military":
        score += 22
    else:
        score -= 100

    if is_military_event(text, event_type):
        score += 55

    if safe_bool(item.get("urgent")):
        score += 40

    if safe_bool(item.get("important")):
        score += 20

    if region == "iran":
        score += 35
    elif region == "middle_east":
        score += 25
    elif region == "superpower":
        score += 18
    else:
        score += 5

    if contains_any(text, SUPERPOWER_TERMS):
        score += 8

    # رویدادها و حملات لحظه‌ای کمی جلوتر.
    if is_breaking(text):
        score += 18

    # ارزش خبری مدل فقط بخشی از اولویت است؛ خبر کم‌امتیاز همچنان امکان انتشار
    # دارد، اما خبر 65+ در صف جلوتر می‌رود.
    news_score = min(100, max(0, safe_int(item.get("news_score"), 0)))
    score += int(news_score * 0.35)

    # راهنمای قدیمی همچنان tie-breaker سبک است.
    model_hint = min(100, max(0, safe_int(item.get("priority_hint"), 0)))
    score += int(model_hint * 0.06)

    # تازگی خبر هم در اولویت اثر می‌گذارد؛ خبر بسیار قدیمی نباید صرفاً به دلیل
    # برچسب important یک‌باره بالاتر از خبر تازه قرار بگیرد.
    published_at = item.get("published_at") or ""
    try:
        published_dt = parse_iso_or_now(published_at)
        age_minutes = max(0.0, (datetime.now(timezone.utc) - published_dt.astimezone(timezone.utc)).total_seconds() / 60.0)
        if age_minutes <= 60:
            score += 8
        elif age_minutes <= 180:
            score += 4
        elif age_minutes > 24 * 60 and not safe_bool(item.get("urgent")):
            score -= 8
    except Exception:
        pass

    # خبر جهانی صفر نمی‌شود؛ فقط نسبت به اولویت منطقه‌ای کمی عقب‌تر است.
    if region == "world" and not is_military_event(text, event_type):
        score -= 3

    return score, region


def recent_bucket_counts(state):
    recent = state.get("_last_posted_buckets", [])[-CONTENT_BUCKET_WINDOW:]
    return {bucket: recent.count(bucket) for bucket in ALLOWED_CONTENT_BUCKETS}


def bucket_deficit(state, bucket):
    recent = state.get("_last_posted_buckets", [])[-CONTENT_BUCKET_WINDOW:]
    total = len(recent)
    if total == 0:
        return CONTENT_BUCKET_TARGETS.get(bucket, 0)
    target = CONTENT_BUCKET_TARGETS.get(bucket, 0) * min(
        CONTENT_BUCKET_WINDOW, total + 1
    )
    return target - recent.count(bucket)


def apply_queue_diversity_score(state, item):
    score, region = calculate_priority(item)
    recent_regions = state.get("_last_posted_regions", [])[-3:]
    bucket = item.get("content_bucket") or infer_content_bucket(
        f"{item.get('title','')}\n{item.get('body','')}",
        region
    )
    deficit = bucket_deficit(state, bucket)
    score += int(max(-20, min(35, deficit * 10)))

    if recent_regions and region == recent_regions[-1]:
        score -= 10
    if len(recent_regions) >= 2 and all(x == region for x in recent_regions[-2:]):
        score -= 20

    # اگر خبر نظامی/ایرانی مهم باشد، تنوع نباید آن را کاملاً عقب بیندازد.
    if safe_bool(item.get("urgent")):
        score += 20

    item["region"] = region
    item["priority_score"] = score
    return score


# ============================================================
# AI provider engine — Groq primary / Gemini backup / Mistral backup
# ============================================================

class AIProviderError(RuntimeError):
    def __init__(self, provider, status_code, message, retry_after=0, fatal=False):
        self.provider = provider
        self.status_code = status_code
        self.retry_after = retry_after
        self.fatal = fatal
        super().__init__(f"{provider} HTTP {status_code}: {message}")


class AIAllProvidersUnavailable(RuntimeError):
    pass


def provider_state(state, provider):
    return state["_ai"]["providers"][provider]


def provider_key_states(state, provider):
    return state["_ai"]["keys"].setdefault(provider, {})


def initialize_ai_state(state):
    for provider in AI_PROVIDER_ORDER:
        pstate = provider_state(state, provider)
        pstate.setdefault("cooldown_until", 0)
        pstate.setdefault("last_request_ts", 0)
        pstate.setdefault("fail_count", 0)
        pstate.setdefault("last_status", 0)
        pstate.setdefault("last_error", {})
        pstate.setdefault("usage", {"requests": 0, "successes": 0, "errors": 0,
                                     "prompt_tokens": 0, "completion_tokens": 0})
        key_states = provider_key_states(state, provider)
        for key in AI_PROVIDER_KEYS[provider]:
            key_states.setdefault(
                key_id(key),
                {
                    "cooldown_until": 0,
                    "fail_count": 0,
                    "last_status": 0,
                    "last_used": 0,
                },
            )
        valid = {key_id(k) for k in AI_PROVIDER_KEYS[provider]}
        for kid in list(key_states):
            if kid not in valid:
                del key_states[kid]


def choose_available_provider_keys(state, provider):
    keys = AI_PROVIDER_KEYS[provider]
    if not keys:
        return []
    initialize_ai_state(state)
    now = now_ts()
    info = provider_key_states(state, provider)
    available = []
    for idx, key in enumerate(keys):
        kid = key_id(key)
        entry = info[kid]
        if float(entry.get("cooldown_until", 0) or 0) <= now:
            available.append((idx, key))
    if available:
        return available

    # اگر همه کلیدهای همان provider cooldown هستند، provider را هم bypass می‌کنیم؛
    # کل زنجیره نباید روی یک کلید منتظر بماند.
    return []


def mark_provider_cooldown(state, provider, seconds, status=0):
    pstate = provider_state(state, provider)
    pstate["cooldown_until"] = max(
        float(pstate.get("cooldown_until", 0) or 0),
        now_ts() + max(0, seconds),
    )
    pstate["fail_count"] = int(pstate.get("fail_count", 0)) + 1
    pstate["last_status"] = status


def mark_key_cooldown(state, provider, key, seconds, status=0):
    entry = provider_key_states(state, provider).setdefault(
        key_id(key),
        {"cooldown_until": 0, "fail_count": 0, "last_status": 0, "last_used": 0},
    )
    entry["cooldown_until"] = now_ts() + max(0, seconds)
    entry["fail_count"] = int(entry.get("fail_count", 0)) + 1
    entry["last_status"] = status
    entry["last_used"] = now_ts()


def mark_provider_success(state, provider, key):
    pstate = provider_state(state, provider)
    pstate["cooldown_until"] = 0
    pstate["fail_count"] = 0
    pstate["last_status"] = 200
    pstate["last_request_ts"] = now_ts()
    pstate["last_error"] = {}
    entry = provider_key_states(state, provider).setdefault(
        key_id(key),
        {"cooldown_until": 0, "fail_count": 0, "last_status": 0, "last_used": 0},
    )
    entry["cooldown_until"] = 0
    entry["fail_count"] = 0
    entry["last_status"] = 200
    entry["last_used"] = now_ts()


def retry_after_seconds(response, default_seconds):
    raw = response.headers.get("Retry-After")
    if raw:
        try:
            return max(1, int(float(raw)))
        except Exception:
            pass
    try:
        payload = response.json()
        raw = payload.get("retry_after") or payload.get("parameters", {}).get("retry_after")
        if raw:
            return max(1, int(float(raw)))
    except Exception:
        pass
    return default_seconds


def provider_model_name(provider):
    return AI_PROVIDER_MODELS[provider]


def groq_structured_output_schema():
    """
    Schema کوچک و محافظه‌کارانه مشترک برای Stage 1/2.
    فیلدهای قابل استنتاج مثل region/content_bucket در کد محلی ساخته می‌شوند تا
    حجم JSON اجباری پایین بماند و Structured Outputs پایدارتر شود.
    """
    item_properties = {
        "draft_id": {"type": "integer"},
        "relevant": {"type": "boolean"},
        "duplicate": {"type": "boolean"},
        "duplicate_reason": {"type": "string"},
        "is_update": {"type": "boolean"},
        "novelty_score": {"type": "integer"},
        "news_score": {"type": "integer"},
        "event_key": {"type": "string"},
        "title": {"type": "string"},
        "body": {"type": "string"},
        "post_indices": {
            "type": "array",
            "items": {"type": "integer"},
        },
        "media_post_index": {"type": "integer"},
        "verification": {"type": "string"},
        "urgent": {"type": "boolean"},
        "important": {"type": "boolean"},
        "needs_editor": {"type": "boolean"},
        "style_mode": {"type": "string"},
        "length_class": {"type": "string"},
        "entities": {
            "type": "array",
            "items": {"type": "string"},
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "raptor_news_items_v5",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": item_properties,
                            "required": list(item_properties.keys()),
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["items"],
                "additionalProperties": False,
            },
        },
    }


def provider_prompt_payload(provider, prompt, max_output_tokens):
    if provider == "groq":
        payload = {
            "model": provider_model_name(provider),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return only the JSON object required by the supplied schema. "
                        "Do not add Markdown or commentary. "
                        "Every required field must be present."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.15,
            # max_completion_tokens is the current Groq parameter.
            "max_completion_tokens": max_output_tokens,
            # GPT-OSS supports include_reasoning=false; this keeps reasoning out of
            # the structured output path and leaves the token budget for JSON itself.
            "include_reasoning": False,
            # reasoning_effort=low is supported by GPT-OSS 120B and keeps the
            # classification/rewriting call focused on the requested JSON.
            "reasoning_effort": "low",
            "response_format": groq_structured_output_schema(),
        }
        return payload

    if provider == "mistral":
        return {
            "model": provider_model_name(provider),
            "messages": [
                {
                    "role": "system",
                    "content": "Output only valid JSON. No Markdown. No extra commentary.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.18,
            "max_tokens": max_output_tokens,
            "response_format": {"type": "json_object"},
        }

    return {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.15,
            "maxOutputTokens": max_output_tokens,
        },
    }


def parse_provider_text(provider, data):
    if provider == "gemini":
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini candidate ندارد")
        parts = candidates[0].get("content", {}).get("parts", [])
        raw = "\n".join(str(p.get("text", "")) for p in parts if isinstance(p, dict)).strip()
        if not raw:
            raise ValueError("Gemini پاسخ متنی خالی برگرداند")
        return raw

    choices = data.get("choices", [])
    if not choices:
        raise ValueError(f"{AI_PROVIDER_LABELS[provider]} choices ندارد")
    message = choices[0].get("message", {}) or {}
    content = message.get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            str(x.get("text", "")) for x in content if isinstance(x, dict)
        )
    raw = str(content or "").strip()
    if not raw:
        raise ValueError(f"{AI_PROVIDER_LABELS[provider]} پاسخ متنی خالی برگرداند")
    return raw


def parse_json_response(raw, provider="AI"):
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"خروجی {provider} JSON معتبر نیست: {raw[:500]}")


def provider_headers(provider, key):
    if provider == "gemini":
        return {
            "Content-Type": "application/json",
            "X-goog-api-key": key,
        }
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }


def provider_url(provider):
    return {
        "groq": GROQ_API_URL,
        "gemini": GEMINI_API_URL,
        "mistral": MISTRAL_API_URL,
    }[provider]


def wait_for_provider_slot(state, provider):
    pstate = provider_state(state, provider)
    last_request = float(pstate.get("last_request_ts", 0) or 0)
    delay = AI_MIN_REQUEST_INTERVAL_SECONDS - (now_ts() - last_request)
    if delay > 0:
        time.sleep(delay)


def call_single_provider(state, provider, prompt, task_name):
    keys = choose_available_provider_keys(state, provider)
    if not keys:
        raise AIProviderError(
            provider,
            429,
            "همه کلیدهای این provider فعلاً cooldown هستند یا کلیدی تعریف نشده است.",
        )

    last_error = None
    attempts = min(runtime_ai_attempts(state), len(keys))

    for attempt in range(attempts):
        _, key = keys[attempt]
        wait_for_provider_slot(state, provider)
        provider_state(state, provider)["last_request_ts"] = now_ts()
        save_state(state)

        payload = provider_prompt_payload(provider, prompt, AI_MAX_OUTPUT_TOKENS)
        timeout = (15, 90) if provider != "mistral" else (15, 90)

        try:
            usage_stats = provider_state(state, provider).setdefault(
                "usage", {"requests": 0, "successes": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0}
            )
            usage_stats["requests"] = safe_int(usage_stats.get("requests"), 0) + 1
            response = requests.post(
                provider_url(provider),
                headers=provider_headers(provider, key),
                json=payload,
                timeout=timeout,
            )
            status = response.status_code

            if status == 200:
                data = response.json()
                raw = parse_provider_text(provider, data)
                parsed = parse_json_response(raw, AI_PROVIDER_LABELS[provider])
                usage_meta = data.get("usage") or data.get("usageMetadata") or {}
                usage_stats = provider_state(state, provider).setdefault(
                    "usage", {"requests": 0, "successes": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0}
                )
                usage_stats["successes"] = safe_int(usage_stats.get("successes"), 0) + 1
                usage_stats["prompt_tokens"] += safe_int(usage_meta.get("prompt_tokens", usage_meta.get("promptTokenCount", 0)), 0)
                usage_stats["completion_tokens"] += safe_int(usage_meta.get("completion_tokens", usage_meta.get("candidatesTokenCount", 0)), 0)
                mark_provider_success(state, provider, key)
                save_state(state)
                log.info(
                    "%s موفق | task=%s | model=%s | key=%s",
                    AI_PROVIDER_LABELS[provider],
                    task_name,
                    provider_model_name(provider),
                    key_id(key),
                )
                return parsed

            retry_after = retry_after_seconds(
                response,
                AI_PROVIDER_COOLDOWN_FLOOR_SECONDS,
            )

            if status == 429:
                retry_after = min(
                    AI_429_MAX_COOLDOWN_SECONDS,
                    max(AI_PROVIDER_COOLDOWN_FLOOR_SECONDS, retry_after),
                )
                mark_provider_cooldown(state, provider, retry_after, status)
                mark_key_cooldown(state, provider, key, retry_after, status)
                save_state(state)
                raise AIProviderError(
                    provider,
                    status,
                    f"429؛ cooldown {retry_after}s",
                    retry_after=retry_after,
                )

            if status in (401, 403):
                mark_key_cooldown(state, provider, key, 6 * 3600, status)
                mark_provider_cooldown(state, provider, 30, status)
                save_state(state)
                last_error = AIProviderError(provider, status, "کلید مجوز معتبر ندارد")
                continue

            if status in (400, 404):
                # این خطا غالباً از مدل/درخواست است؛ provider را چند دقیقه کنار می‌گذاریم
                # و فوراً به fallback بعدی می‌رویم تا ربات قفل نشود.
                mark_provider_cooldown(state, provider, 180, status)
                save_state(state)
                raise AIProviderError(
                    provider,
                    status,
                    truncate_text(response.text, 600),
                    retry_after=180,
                )

            if status in (408, 409, 425, 500, 502, 503, 504):
                backoff = min(
                    AI_PROVIDER_RETRY_MAX_SECONDS,
                    AI_PROVIDER_RETRY_BASE_SECONDS * (2 ** attempt),
                )
                cooldown = min(90, max(10, backoff))
                mark_key_cooldown(state, provider, key, cooldown, status)
                last_error = AIProviderError(provider, status, truncate_text(response.text, 500))
                save_state(state)
                if attempt + 1 < attempts:
                    jitter = 0.25 + (0.25 * (int(sha1_text(key_id(key))[:6], 16) % 100) / 100.0)
                    time.sleep(min(AI_PROVIDER_RETRY_MAX_SECONDS, backoff + jitter))
                    continue
                raise last_error

            raise AIProviderError(
                provider,
                status,
                truncate_text(response.text, 700),
            )

        except AIProviderError:
            raise
        except (requests.Timeout, requests.ConnectionError, requests.RequestException) as exc:
            last_error = exc
            backoff = min(
                AI_PROVIDER_RETRY_MAX_SECONDS,
                AI_PROVIDER_RETRY_BASE_SECONDS * (2 ** attempt),
            )
            mark_key_cooldown(state, provider, max(20, int(backoff)), -1)
            save_state(state)
            if attempt + 1 < attempts:
                jitter = 0.25 + (0.25 * (int(sha1_text(key_id(key))[:6], 16) % 100) / 100.0)
                time.sleep(min(AI_PROVIDER_RETRY_MAX_SECONDS, backoff + jitter))
                continue
        except Exception as exc:
            last_error = exc
            log.warning(
                "%s task=%s پاسخ نامعتبر/غیرمنتظره بود: %s",
                AI_PROVIDER_LABELS[provider], task_name, exc
            )
            mark_key_cooldown(state, provider, key, 30, -2)
            save_state(state)
            if attempt + 1 < attempts:
                continue

    if isinstance(last_error, AIProviderError):
        raise last_error
    raise AIProviderError(
        provider,
        -1,
        f"بعد از {attempts} تلاش داخلی ناموفق شد: {last_error}",
    )


def rotate_provider_order(start_provider=None, order=None):
    order = list(order or AI_PROVIDER_ORDER)
    if start_provider not in order:
        return order
    idx = order.index(start_provider)
    return [order[(idx + offset) % len(order)] for offset in range(1, len(order) + 1)]


def ai_call_json(state, prompt, task_name, preferred_order=None):
    initialize_ai_state(state)
    configured_order = list(preferred_order or runtime_provider_order(state))
    enabled = set(runtime_provider_order(state))
    order = [provider for provider in configured_order if provider in enabled]
    tried = []

    for provider in order:
        if not AI_PROVIDER_KEYS[provider]:
            continue
        pstate = provider_state(state, provider)
        if float(pstate.get("cooldown_until", 0) or 0) > now_ts():
            continue
        tried.append(provider)
        try:
            parsed = call_single_provider(state, provider, prompt, task_name)
            return parsed, provider
        except AIProviderError as exc:
            record_provider_error(state, provider, task_name, exc)
            log.warning(
                "%s برای task=%s در دسترس نبود؛ fallback بعدی فعال می‌شود | status=%s | %s",
                AI_PROVIDER_LABELS[provider], task_name, exc.status_code, sanitize_error_text(str(exc)),
            )
            continue
        except Exception as exc:
            record_system_error(state, "AI", f"{AI_PROVIDER_LABELS[provider]} / {task_name}", str(exc), provider=provider)
            log.warning(
                "%s برای task=%s خطای پیش‌بینی‌نشده داد؛ fallback بعدی | %s",
                AI_PROVIDER_LABELS[provider], task_name, sanitize_error_text(str(exc)),
            )
            continue

    if not order:
        raise AIAllProvidersUnavailable("همه ارائه‌دهندگان هوش مصنوعی از پنل غیرفعال شده‌اند یا کلیدی ندارند.")
    raise AIAllProvidersUnavailable(
        "هیچ موتور AI در این چرخه در دسترس نبود. تلاش‌شده: "
        + ", ".join(tried or ["none"])
    )


def build_style_memory_context(state):
    style = state.get("_style_memory", {})
    titles = [x for x in style.get("recent_titles", []) if x][-8:]
    openings = [x for x in style.get("recent_openings", []) if x][-6:]
    counts = style.get("entity_counts", {}) if isinstance(style.get("entity_counts"), dict) else {}
    top_entities = [k for k, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]]
    return (
        "تیترهای اخیر برای جلوگیری از ساختار تکراری:\n- "
        + "\n- ".join(titles or ["موردی ثبت نشده"])
        + "\n\nشروع جمله‌های اخیر:\n- "
        + "\n- ".join(openings or ["موردی ثبت نشده"])
        + "\n\nنام‌های پرتکرار برای یکدست‌نویسی:\n- "
        + "، ".join(top_entities or ["موردی ثبت نشده"])
    )


def build_event_memory_context(state):
    now = now_ts()
    memories = []
    for event_key, item in state.get("_event_memory", {}).items():
        ts = float(item.get("last_published_ts", 0) or 0)
        if now - ts <= EVENT_MEMORY_WINDOW_MINUTES * 60:
            memories.append((ts, event_key, item))
    memories.sort(reverse=True)
    chunks = []
    for _, event_key, item in memories[:8]:
        chunks.append(
            f"event_key={event_key}\n"
            f"title={item.get('title','')}\n"
            f"summary={truncate_text(item.get('body',''), 300)}\n"
            f"bucket={item.get('content_bucket','')}\n"
            f"last_published_ts={item.get('last_published_ts',0)}\n"
            f"updates_24h={item.get('updates_24h',0)}\n"
            f"entities={', '.join(item.get('entities', [])[:6])}"
        )
    return "\n---\n".join(chunks) if chunks else "مورد قبلی مهمی در حافظه کوتاه‌مدت وجود ندارد."


def build_analysis_content(batch_posts):
    chunks = []
    current_date = now_tehran().strftime("%Y-%m-%d")
    for index, post in enumerate(batch_posts, start=1):
        source_text = post.get("text", "")
        text = truncate_text(source_text, AI_INPUT_MAX_CHARS_PER_POST)
        media = []
        if post.get("photo"):
            media.append("photo")
        if post.get("video"):
            media.append("video")
        media_label = "، ".join(media) if media else "none"
        chunks.append(
            f"[پست {index}]\n"
            f"منبع داخلی: {post.get('source_name','')}\n"
            f"زمان انتشار منبع: {post.get('published_at','')}\n"
            f"تاریخ امروز تهران: {current_date}\n"
            f"رسانه منبع: {media_label}\n"
            f"متن خام:\n{text}"
        )
    return "\n\n---\n\n".join(chunks)


def validate_ai_items(items, max_items=EVENT_MAX_OUTPUT_ITEMS, state=None):
    if not isinstance(items, list):
        raise ValueError("AI items باید list باشد")

    validated = []
    allowed_event_types = {"military_event", "security", "defense", "geopolitics", "routine"}
    allowed_categories = {"military", "security", "defense", "geopolitics", "technology", "general"}
    allowed_verification = {"reported", "developing", "confirmed", "analysis"}
    allowed_styles = {
        "breaking", "brief", "standard", "conflict", "defense_tech",
        "iran", "important", "analysis", "geopolitics",
        "fact_check", "comparison", "context",
    }
    allowed_lengths = {"brief", "standard", "full"}

    for raw in items[:max_items]:
        if not isinstance(raw, dict):
            continue

        relevant = safe_bool(raw.get("relevant"))
        if not relevant:
            validated.append({
                "draft_id": safe_int(raw.get("draft_id"), 0),
                "relevant": False,
                "duplicate": safe_bool(raw.get("duplicate")),
                "duplicate_reason": normalize_space(str(raw.get("duplicate_reason", "")))[:400],
                "is_update": False,
                "novelty_score": 0,
                "news_score": 0,
                "event_key": "",
                "title": "",
                "body": "",
                "post_indices": [],
                "media_post_index": 0,
                "verification": "reported",
                "urgent": False,
                "important": False,
                "needs_editor": False,
                "style_mode": "brief",
                "length_class": "brief",
                "priority_hint": 0,
                "source_note": "",
                "entities": [],
                "significance": "",
                "image_query": "",
            })
            continue

        title = normalize_persian_text(str(raw.get("title", "")))
        body = normalize_persian_text(str(raw.get("body", "")))
        if not title or not body:
            if state is not None:
                record_decision(state, title or "پیش‌نویس بدون تیتر", "رد‌شده", "تیتر یا بدنه خبر خالی بود", raw.get("news_score"))
            continue

        news_score = max(0, min(100, safe_int(raw.get("news_score"), 0)))
        # خبرهای زیر حداقل ارزش خبری اصلاً وارد صف نشوند.
        if news_score < MIN_NEWS_SCORE:
            log.info("خبر به علت ارزش خبری پایین رد تحریریه‌ای شد | score=%s | title=%s", news_score, title[:120])
            reason = f"امتیاز ارزش خبری {news_score} کمتر از حداقل {MIN_NEWS_SCORE} بود؛ رد نهایی تحریریه‌ای، نه خطای فنی"
            if state is not None:
                record_decision(state, title, "رد‌شده", reason, news_score)
            validated.append({
                "draft_id": safe_int(raw.get("draft_id"), 0), "relevant": False,
                "duplicate": False, "duplicate_reason": reason, "rejection_reason": "low_news_score",
                "news_score": news_score, "title": title[:240], "body": "", "event_key": "",
                "post_indices": [], "media_post_index": 0, "verification": "reported",
                "urgent": False, "important": False, "needs_editor": False,
                "style_mode": "brief", "length_class": "brief", "entities": [],
            })
            continue

        event_type = str(raw.get("event_type", "routine")).strip()
        if event_type not in allowed_event_types:
            event_type = "routine"

        category = str(raw.get("category", "general")).strip().lower()
        if category not in allowed_categories:
            category = "general"

        verification = str(raw.get("verification", "reported")).strip().lower()
        if verification not in allowed_verification:
            verification = "reported"

        style_mode = str(raw.get("style_mode", "standard")).strip().lower()
        if style_mode not in allowed_styles:
            style_mode = "standard"

        length_class = str(raw.get("length_class", "standard")).strip().lower()
        if length_class not in allowed_lengths:
            length_class = "standard"

        region = str(raw.get("region", "world")).strip().lower()
        if region not in {"iran", "middle_east", "world", "superpower"}:
            region = infer_region(f"{title}\n{body}")

        bucket = str(raw.get("content_bucket", "")).strip().lower()
        if bucket not in ALLOWED_CONTENT_BUCKETS:
            bucket = infer_content_bucket(f"{title}\n{body}", region)

        raw_indices = raw.get("post_indices", [])
        if not isinstance(raw_indices, list):
            raw_indices = []
        post_indices = []
        for value in raw_indices:
            idx = safe_int(value, 0)
            if idx > 0 and idx not in post_indices:
                post_indices.append(idx)
        post_indices = post_indices[:AI_BATCH_MAX_POSTS]

        media_post_index = safe_int(raw.get("media_post_index"), 0)
        if media_post_index < 0 or media_post_index > AI_BATCH_MAX_POSTS:
            media_post_index = 0

        entities = raw.get("entities", [])
        if not isinstance(entities, list):
            entities = []
        entities = [
            normalize_persian_text(str(x))[:100]
            for x in entities
            if str(x).strip()
        ][:10]

        event_key = normalize_event_key(raw.get("event_key", ""))
        if not event_key:
            event_key = event_key_or_signature(title, body, "")

        novelty = max(0, min(100, safe_int(raw.get("novelty_score"), 0)))

        validated.append({
            "draft_id": safe_int(raw.get("draft_id"), 0),
            "relevant": True,
            "duplicate": safe_bool(raw.get("duplicate")),
            "duplicate_reason": normalize_space(str(raw.get("duplicate_reason", "")))[:400],
            "is_update": safe_bool(raw.get("is_update")),
            "novelty_score": novelty,
            "news_score": news_score,
            "event_key": event_key,
            "title": title[:240],
            "body": body[:6500],
            "significance": "",
            "image_query": "",
            "post_indices": post_indices,
            "media_post_index": media_post_index,
            "content_bucket": bucket,
            "region": region,
            "event_type": event_type,
            "category": category,
            "verification": verification,
            "urgent": safe_bool(raw.get("urgent")),
            "important": safe_bool(raw.get("important")) or news_score >= 80,
            "needs_editor": (
                safe_bool(raw.get("needs_editor"))
                or news_score < PREFERRED_NEWS_SCORE
                or (verification == "developing" and news_score < 80)
            ),
            "style_mode": style_mode,
            "length_class": length_class,
            "priority_hint": min(100, max(0, news_score)),
            "source_note": "",
            "entities": entities,
        })
    return validated


def analyze_and_rewrite_safe(state, batch_posts, source_name):
    if not batch_posts:
        return [], None

    content = build_analysis_content(batch_posts)
    prompt = REWRITE_PROMPT.replace("{content}", content)
    prompt = prompt.replace("{current_date}", now_tehran().strftime("%Y-%m-%d"))
    prompt = prompt.replace("{event_memory}", build_event_memory_context(state))
    prompt = prompt.replace("{style_memory}", build_style_memory_context(state))

    parsed, provider = ai_call_json(
        state,
        prompt,
        "analysis",
        preferred_order=runtime_provider_order(state),
    )
    items = parsed.get("items", []) if isinstance(parsed, dict) else []
    if not items and isinstance(parsed, dict) and "title" in parsed:
        items = [parsed]
    return validate_ai_items(items, state=state), provider


def build_editor_prompt(state, drafts, source_posts):
    source_chunks = []
    for idx, post in enumerate(source_posts, start=1):
        source_chunks.append(
            f"[ورودی {idx}]\n"
            f"زمان: {post.get('published_at','')}\n"
            f"متن: {truncate_text(post.get('text',''), 1400)}"
        )
    draft_chunks = []
    for draft_id, draft in enumerate(drafts, start=1):
        draft_chunks.append(
            f"[پیش‌نویس {draft_id}]\n"
            + json.dumps({
                "draft_id": draft_id,
                "relevant": True,
                "event_key": draft.get("event_key", ""),
                "title": draft.get("title", ""),
                "body": draft.get("body", ""),
                "post_indices": draft.get("post_indices", []),
                "media_post_index": draft.get("media_post_index", 0),
                "content_bucket": draft.get("content_bucket", ""),
                "region": draft.get("region", ""),
                "news_score": draft.get("news_score", 0),
                "event_type": draft.get("event_type", ""),
                "category": draft.get("category", ""),
                "verification": draft.get("verification", ""),
                "urgent": draft.get("urgent", False),
                "important": draft.get("important", False),
                "style_mode": draft.get("style_mode", "standard"),
                "length_class": draft.get("length_class", "standard"),
                "priority_hint": draft.get("priority_hint", 0),
                "source_note": draft.get("source_note", ""),
                "entities": draft.get("entities", []),
            }, ensure_ascii=False)
        )

    return (
        EDITOR_PROMPT
        + "\n\nورودی‌های خام برای کنترل واقعیت:\n"
        + "\n---\n".join(source_chunks)
        + "\n\n---\nپیش‌نویس‌ها:\n"
        + "\n---\n".join(draft_chunks)
        + "\n\nحافظه سبک کانال:\n"
        + build_style_memory_context(state)
    )


def final_editor(state, drafts, source_posts, first_provider):
    if not EDITOR_ENABLED or not drafts:
        return drafts, first_provider

    drafts = drafts[:min(AI_EDITOR_MAX_ITEMS, EVENT_MAX_OUTPUT_ITEMS)]
    prompt = build_editor_prompt(state, drafts, source_posts)
    order = rotate_provider_order(first_provider, runtime_provider_order(state))
    try:
        parsed, provider = ai_call_json(
            state,
            prompt,
            "editor",
            preferred_order=order,
        )
        raw_items = parsed.get("items", []) if isinstance(parsed, dict) else []
        edited = validate_ai_items(raw_items, max_items=len(drafts), state=state)
        if not edited:
            raise ValueError("editor خروجی قابل استفاده نداد")

        # editor نباید اطلاعات ساختاری stage1 را از بین ببرد. بر اساس draft_id یا ترتیب merge می‌کنیم.
        merged = []
        for idx, original in enumerate(drafts, start=1):
            candidate = None
            for item in edited:
                if safe_int(item.get("draft_id"), 0) == idx:
                    candidate = item
                    break
            if candidate is None and idx <= len(edited):
                candidate = edited[idx - 1]
            if candidate is None:
                # فقط نبود خروجی برای این draft را با نسخه اصلی جبران می‌کنیم؛
                # veto صریح ویراستار نباید با fallback بی‌اثر شود.
                merged.append(original)
                continue
            if not candidate.get("relevant", False):
                continue

            merged_item = dict(original)
            for field in (
                "title", "body", "event_key", "verification",
                "style_mode", "length_class", "entities", "needs_editor",
                "priority_hint", "urgent", "important", "news_score",
                "media_post_index",
            ):
                if field in candidate and candidate.get(field) not in (None, ""):
                    merged_item[field] = candidate[field]
            merged_item["event_key"] = normalize_event_key(merged_item.get("event_key")) or original.get("event_key")
            merged_item["post_indices"] = original.get("post_indices", [])
            merged_item["content_bucket"] = original.get("content_bucket")
            merged_item["region"] = original.get("region")
            merged_item["event_type"] = original.get("event_type")
            merged_item["category"] = original.get("category")
            merged_item["is_update"] = original.get("is_update", False)
            merged_item["novelty_score"] = original.get("novelty_score", 0)
            merged.append(merged_item)

        return merged, provider
    except Exception as exc:
        log.warning(
            "Final editor در دسترس نبود؛ از پیش‌نویس مرحله اول استفاده می‌کنیم: %s",
            exc,
        )
        return drafts, first_provider

# ============================================================
# Telegram public channel scraping
# ============================================================

NOISE_PATTERNS = (
    "join our channel",
    "subscribe to our channel",
    "подписывайтесь",
    "подписаться",
    "عضو شوید",
    "به کانال ما بپیوندید",
    "تبلیغات",
)


def looks_like_low_value_post(post):
    text = normalize_space(post.get("text", ""))
    if not text and not post.get("photo") and not post.get("video"):
        return True
    if len(text) < 12 and not post.get("photo") and not post.get("video"):
        return True

    lowered = normalize_for_match(text)
    if any(normalize_for_match(pattern) in lowered for pattern in NOISE_PATTERNS):
        if not contains_any(text, MILITARY_EVENT_TERMS | BREAKING_TERMS):
            return True

    if text and re.fullmatch(r"(https?://\S+|@\w+)(\s+https?://\S+|\s+@\w+)*", text):
        return True
    return False


def filter_posts(posts):
    return [post for post in posts if not looks_like_low_value_post(post)]


def extract_telegram_posts_from_html(channel, html_text):
    if BeautifulSoup is None:
        raise RuntimeError("beautifulsoup4 نصب نیست.")

    soup = BeautifulSoup(html_text, "html.parser")
    nodes = soup.select("[data-post]")
    posts = []

    seen = set()

    for node in nodes:
        data_post = node.get("data-post", "")
        expected_prefix = f"{channel}/"
        if not data_post.startswith(expected_prefix):
            continue

        message_id = data_post[len(expected_prefix):]
        if not message_id or message_id in seen:
            continue
        seen.add(message_id)

        text_node = node.select_one(".tgme_widget_message_text")
        text = text_node.get_text("\n", strip=True) if text_node else ""

        time_node = node.select_one("time")
        published_at = time_node.get("datetime") if time_node else ""

        link_node = node.select_one(".tgme_widget_message_date")
        source_url = None
        if link_node is not None:
            anchor = link_node.find("a")
            if anchor and anchor.get("href"):
                source_url = anchor["href"]

        photo_url = None
        photo_wrap = node.select_one(".tgme_widget_message_photo_wrap")
        if photo_wrap:
            style = photo_wrap.get("style", "")
            m = re.search(
                r"background-image:url\(['\"]?([^'\")]+)",
                style,
                flags=re.IGNORECASE,
            )
            if m:
                photo_url = html.unescape(m.group(1))

        video_url = None
        video_node = node.select_one("video")
        if video_node and video_node.get("src"):
            video_url = html.unescape(video_node["src"])

        # پست‌های مدیا بدون متن هم نگه داشته می‌شوند.
        if text or photo_url or video_url:
            posts.append(
                {
                    "uid": f"tg:{channel}:{message_id}",
                    "_source_key": f"tg:{channel}",
                    "text": text,
                    "title": text.split("\n", 1)[0][:220] if text else "",
                    "photo": photo_url,
                    "video": video_url,
                    "source_name": f"کانال {channel}",
                    "source_url": source_url or f"https://t.me/{channel}/{message_id}",
                    "published_at": published_at or datetime.now(timezone.utc).isoformat(),
                }
            )

    return filter_posts(posts)


def fetch_channel_posts(channel, state=None):
    url = f"https://t.me/s/{channel}"
    try:
        resp = requests.get(
            url,
            timeout=(10, 30),
            headers=HTTP_HEADERS,
        )
        resp.raise_for_status()
        posts = extract_telegram_posts_from_html(channel, resp.text)
        if state is not None:
            with STATE_LOCK:
                runtime = state.setdefault("_runtime", {})
                runtime["last_source_success_ts"] = now_ts()
                runtime["last_source_error"] = ""
                health = state.setdefault("_source_health", {}).setdefault(channel, {"fail_count": 0, "last_ok_ts": 0, "last_error_ts": 0, "last_error": ""})
                health.update({"fail_count": 0, "last_ok_ts": now_ts(), "last_error": ""})
        return posts
    except Exception as exc:
        log.warning("خطا در گرفتن کانال %s: %s", channel, sanitize_error_text(exc))
        if state is not None:
            with STATE_LOCK:
                state.setdefault("_runtime", {})["last_source_error"] = sanitize_error_text(exc)
                health = state.setdefault("_source_health", {}).setdefault(channel, {"fail_count": 0, "last_ok_ts": 0, "last_error_ts": 0, "last_error": ""})
                health["fail_count"] = safe_int(health.get("fail_count"), 0) + 1
                health["last_error_ts"] = now_ts()
                health["last_error"] = sanitize_error_text(exc)[:220]
                stat = state.setdefault("_source_stats", {}).setdefault(channel, {"fetched": 0, "new": 0, "published": 0, "rejected": 0, "errors": 0})
                stat["errors"] = safe_int(stat.get("errors"), 0) + 1
                record_system_error(state, "Source", f"@{channel}", exc)
                save_state(state)
        return []


# ============================================================
# Image search / image acquisition
# ============================================================

def find_wikimedia_image(query):
    """
    fallback بدون API key:
    Wikimedia Commons MediaSearch
    """
    if not query:
        return None

    try:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": 8,
            "prop": "imageinfo",
            "iiprop": "url|mime|size",
            "format": "json",
        }

        resp = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params=params,
            timeout=(10, 25),
            headers=HTTP_HEADERS,
        )
        resp.raise_for_status()

        data = resp.json()
        pages = list((data.get("query") or {}).get("pages", {}).values())

        # فقط تصاویر رایج را برگردان.
        allowed = {"image/jpeg", "image/png", "image/webp"}

        candidates = []
        for page in pages:
            infos = page.get("imageinfo") or []
            if not infos:
                continue
            info = infos[0]
            url = info.get("thumburl") or info.get("url")
            mime = info.get("mime")
            width = safe_int(info.get("width"), 0)

            if url and (mime in allowed or not mime):
                candidates.append((width, url))

        candidates.sort(reverse=True)

        if candidates:
            return candidates[0][1]

    except Exception as exc:
        log.info("Wikimedia image search failed: %s", exc)

    return None


def image_query_from_item(item):
    query = normalize_space(item.get("image_query", ""))
    if query:
        return query

    title = item.get("title", "")
    # query ساده برای fallback.
    return " ".join(title.split()[:6])


def choose_image_url(item, original_posts):
    """
    پیش‌فرض v5 فقط رسانه‌ای را قبول می‌کند که از خود منابع ورودی آمده باشد.
    جست‌وجوی تصویر عمومی به‌صورت opt-in باقی می‌ماند و پیش‌فرض خاموش است.
    """
    for post in original_posts:
        if post.get("photo"):
            return post["photo"]

    for post in original_posts:
        if post.get("article_image"):
            return post["article_image"]

    if not ALLOW_GENERIC_IMAGE_FALLBACK:
        return None

    query = image_query_from_item(item)
    return find_wikimedia_image(query)


def select_result_media(result, cluster, linked_posts):
    """
    اگر مدل یک media_post_index مطمئن داده باشد، رسانه فقط از همان ورودی گرفته می‌شود.
    در غیر این صورت fallback محافظه‌کارانه روی رسانه‌های linked_posts انجام می‌شود.
    """
    preferred = safe_int(result.get("media_post_index"), 0)
    if 1 <= preferred <= len(cluster):
        preferred_post = cluster[preferred - 1]
        photo = preferred_post.get("photo")
        video = preferred_post.get("video")
        source_url = preferred_post.get("source_url", "")
        source_urls = [source_url] if source_url else []
        if photo or video:
            return photo, video, source_url, source_urls

    return combine_original_media(linked_posts)


# ============================================================
# Media streaming
# ============================================================

def stream_download_to_temp(url, max_bytes, timeout, suffix):
    if not url:
        raise ValueError("URL رسانه خالی است.")

    response = requests.get(
        url,
        stream=True,
        timeout=(10, timeout),
        headers=HTTP_HEADERS,
        allow_redirects=True,
    )
    response.raise_for_status()

    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            content_length_int = int(content_length)
        except (TypeError, ValueError):
            content_length_int = None
        if content_length_int is not None and content_length_int > max_bytes:
            raise ValueError(
                f"رسانه از حد مجاز بزرگ‌تر است: {content_length_int} bytes"
            )

    fd, path = tempfile.mkstemp(prefix="media-", suffix=suffix)
    os.close(fd)

    written = 0

    try:
        with open(path, "wb") as out:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                written += len(chunk)
                if written > max_bytes:
                    raise ValueError(
                        f"رسانه از حد مجاز بزرگ‌تر است: {written} bytes"
                    )
                out.write(chunk)

        return path, written
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


# ============================================================
# Telegram API
# ============================================================

class TelegramAPIError(RuntimeError):
    def __init__(self, status_code, description, retry_after=0):
        super().__init__(f"Telegram HTTP {status_code}: {description}")
        self.status_code = status_code
        self.description = description
        self.retry_after = retry_after
        # 5xx/timeout-like HTTP responses can happen after Telegram accepted a send.
        self.delivery_uncertain = (safe_int(status_code, 0) >= 500 or safe_int(status_code, 0) == 408)


class TelegramTransportError(RuntimeError):
    """Network failure after a Telegram send was attempted; result may be unknown."""


def telegram_request(method, data=None, files=None, timeout=30):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"

    try:
        response = requests.post(
            url,
            data=data,
            files=files,
            timeout=timeout,
        )
    except Exception as exc:
        raise TelegramTransportError(sanitize_error_text(exc)) from exc

    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if response.ok and (not isinstance(payload, dict) or "ok" not in payload):
        raise TelegramTransportError("Telegram returned a malformed success response; delivery status is uncertain.")

    if response.status_code == 429:
        retry_after = (
            payload.get("parameters", {}).get("retry_after")
            or response.headers.get("Retry-After")
            or 30
        )
        raise TelegramAPIError(
            response.status_code,
            payload.get("description", response.text[:500]),
            int(retry_after),
        )

    if not response.ok or payload.get("ok") is False:
        raise TelegramAPIError(
            response.status_code,
            payload.get("description", response.text[:500]),
        )

    return payload


def _utf16_len(value):
    return len(str(value or "").encode("utf-16-le")) // 2


def send_text_to_telegram(text, bold_prefix_len=0):
    """ارسال قطعه‌بندی‌شده؛ تیتر پیام اول با MessageEntity به‌صورت Bold نمایش داده می‌شود."""
    text = (text or "").strip()
    if not text:
        return None
    MAX_TEXT_CHARS = 4000
    chunks = []
    remaining = text
    while len(remaining) > MAX_TEXT_CHARS:
        part = truncate_text(remaining, MAX_TEXT_CHARS)
        cut_len = len(part.rstrip("…"))
        if cut_len <= 0 or cut_len >= len(remaining):
            cut_len = MAX_TEXT_CHARS
            part = remaining[:cut_len]
        chunks.append(part.rstrip())
        remaining = remaining[cut_len:].lstrip()
    if remaining:
        chunks.append(remaining)
    result = None
    all_message_ids = []
    for index, chunk in enumerate(chunks):
        payload = {"chat_id": TARGET_CHAT_ID, "text": chunk, "disable_web_page_preview": False}
        if index == 0 and safe_int(bold_prefix_len, 0) > 0:
            payload["entities"] = json.dumps([{"type": "bold", "offset": 0, "length": min(_utf16_len(chunk[:bold_prefix_len]), _utf16_len(chunk))}])
        try:
            result = telegram_request("sendMessage", data=payload, timeout=30)
        except Exception as exc:
            try:
                setattr(exc, "partial_message_ids", all_message_ids)
            except Exception:
                pass
            raise
        mid = telegram_result_message_id(result)
        if mid:
            all_message_ids.append(mid)
    if isinstance(result, dict):
        result["_all_message_ids"] = all_message_ids
    return result

def build_media_caption(title, body, max_chars=1024):
    """
    نسخه 3.5:
    متن خبر و رسانه همیشه در همان یک پیام Telegram قرار می‌گیرند.
    چون caption رسانه سقف 1024 کاراکتر دارد، فقط بخش بدنه خبر
    در صورت نیاز کوتاه می‌شود؛ پیام متنی دومی ساخته نمی‌شود.
    """
    title = normalize_space(title)
    body = (body or "").strip()
    footer = FOOTER

    # ابتدا برای تیتر فضای کافی نگه می‌داریم.
    title_room = max(100, max_chars - len(footer) - 20)
    if len(title) > title_room:
        title = truncate_text(title, title_room)

    separators = 6  # فاصله‌های بین title/body/footer
    body_room = max_chars - len(title) - len(footer) - separators

    if body_room <= 0:
        return f"{title}\n\n{footer}"[:max_chars]

    body_part = truncate_text(body, body_room)
    caption = f"{title}\n\n{body_part}\n\n{footer}"

    # اگر به دلیل نحوه برش چند کاراکتر اضافه شد، فقط body را کوتاه‌تر می‌کنیم.
    if len(caption) > max_chars:
        extra = len(caption) - max_chars
        body_room = max(1, body_room - extra)
        body_part = truncate_text(body, body_room)
        caption = f"{title}\n\n{body_part}\n\n{footer}"

    return caption[:max_chars]


def caption_bold_entities(caption):
    first_line = str(caption or "").split("\n", 1)[0]
    if not first_line:
        return []
    return [{"type": "bold", "offset": 0, "length": _utf16_len(first_line)}]


def send_photo_to_telegram(title, body, photo_url):
    caption = build_media_caption(title, body, 1024)
    temp_path = None
    try:
        temp_path, _ = stream_download_to_temp(
            photo_url,
            max_bytes=MAX_IMAGE_BYTES,
            timeout=45,
            suffix=".jpg",
        )

        with open(temp_path, "rb") as photo_file:
            return telegram_request(
                "sendPhoto",
                data={
                    "chat_id": TARGET_CHAT_ID,
                    "caption": caption,
                    "caption_entities": json.dumps(caption_bold_entities(caption)),
                },
                files={
                    "photo": (
                        "photo.jpg",
                        photo_file,
                        "image/jpeg",
                    )
                },
                timeout=90,
            )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass



def send_photo_file_id_to_telegram(title, body, photo_file_id):
    """ارسال عکس دریافت‌شده از خود Telegram با file_id؛ بدون دانلود/آپلود مجدد."""
    return telegram_request(
        "sendPhoto",
        data={
            "chat_id": TARGET_CHAT_ID,
            "photo": photo_file_id,
            "caption": build_media_caption(title, body, 1024),
            "caption_entities": json.dumps(caption_bold_entities(build_media_caption(title, body, 1024))),
        },
        timeout=30,
    )


def send_video_to_telegram(title, body, video_url):
    caption = build_media_caption(title, body, 1024)
    temp_path = None
    try:
        temp_path, _ = stream_download_to_temp(
            video_url,
            max_bytes=MAX_VIDEO_BYTES,
            timeout=120,
            suffix=".mp4",
        )

        with open(temp_path, "rb") as video_file:
            return telegram_request(
                "sendVideo",
                data={
                    "chat_id": TARGET_CHAT_ID,
                    "caption": caption,
                    "caption_entities": json.dumps(caption_bold_entities(caption)),
                    "supports_streaming": "true",
                },
                files={
                    "video": (
                        "video.mp4",
                        video_file,
                        "video/mp4",
                    )
                },
                timeout=180,
            )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def strip_source_channel_attribution(text):
    """
    نام/یوزرنیم کانال‌های منبع نباید وارد پست عمومی شوند.
    این پاک‌سازی فقط attribution به کانال‌های منبع را هدف می‌گیرد و
    تلاش می‌کند نام سازمان‌ها، کشورها و منابع رسمی داخل متن خبر را دست‌نخورده نگه دارد.
    """
    text = normalize_persian_text(text)
    if not text:
        return ""

    patterns = [
        # «به نقل/گزارش کانال @name» و حالت‌های مشابه
        r"(?i)\b(?:به\s+)?(?:نقل|گزارش)\s+از\s+(?:کانال|چنل)\s+[@#]?[A-Za-z0-9_\-]+",
        r"(?i)\bبه\s+نقل\s+از\s+(?:کانال|چنل)\s+[^،.!؟\n]+",
        r"(?i)\b(?:به\s+گفته|بر\s+اساس\s+گزارش)\s+(?:کانال|چنل)\s+[@#]?[A-Za-z0-9_\-]+",
        r"(?i)\b(?:کانال|چنل)\s+[@#][A-Za-z0-9_\-]+\s+(?:گزارش\s+داد|اعلام\s+کرد|نوشت|مدعی\s+شد)",
        r"(?i)\b(?:کانال|چنل)\s+[@#][A-Za-z0-9_\-]+",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text)

    # نام‌های منبع فعلی ربات را نیز اگر به شکل plain text داخل attribution آمده باشند، حذف کن.
    for channel in SOURCE_CHANNELS:
        if not channel:
            continue
        text = re.sub(
            rf"(?i)(?<![A-Za-z0-9_])(?:کانال|چنل)\s+@?{re.escape(channel)}\b",
            "",
            text,
        )
        text = re.sub(
            rf"(?i)(?<![A-Za-z0-9_])@{re.escape(channel)}\b",
            "",
            text,
        )

    # فاصله/نشانه‌گذاری باقی‌مانده از حذف attribution را مرتب کن.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([،؛؟!])", r"\1", text)
    text = re.sub(r"([،؛])\s*(?=[،؛])", r"\1 ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(" \n،؛")


def format_public_title(item):
    return normalize_persian_text(
        strip_source_channel_attribution(item.get("title", ""))
    )


def format_public_body(item):
    """خروجی عمومی فقط خبر است؛ هیچ تحلیل/📌 یا عبارت قالبیِ تأیید به‌صورت خودکار تزریق نمی‌شود."""
    body = strip_source_channel_attribution(item.get("body", ""))
    return normalize_persian_text(body)


def build_text_only_message(item):
    title = format_public_title(item)
    body = format_public_body(item)
    parts = []
    if title:
        parts.append(title)
    if body:
        parts.append(body)
    parts.append(FOOTER)
    return "\n\n".join(parts)


def is_recoverable_media_error(exc):
    """خطاهای قطعی رسانه که یعنی Telegram خود فایل را نپذیرفته است."""
    if not isinstance(exc, TelegramAPIError) or exc.status_code != 400:
        return False
    desc = normalize_for_match(exc.description or "")
    media_terms = {
        "photo_invalid_dimensions",
        "wrong file identifier",
        "bad request photo",
        "photo",
        "video",
        "file is too big",
        "wrong type of the web page content",
    }
    return any(normalize_for_match(term) in desc for term in media_terms)


def dispatch_item(item, state=None):
    if is_test_mode(state):
        log.info(
            "TEST_MODE فعال است؛ انتشار واقعی انجام نشد | title=%s | media=%s",
            format_public_title(item),
            "video" if item.get("video") else ("photo" if item.get("photo") else "none"),
        )
        return {"test_mode": True}

    title = format_public_title(item)
    body = format_public_body(item)
    photo = item.get("photo")
    video = item.get("video")

    # اگر رسانه خراب/نامعتبر باشد، همان خبر بدون رسانه منتشر می‌شود.
    # 429 عمداً fallback نمی‌شود؛ چون مشکل rate limit است نه فایل.
    if video:
        try:
            result = send_video_to_telegram(title, body, video)
            item["_sent_media_type"] = "video"
            return result
        except TelegramAPIError as exc:
            if exc.status_code == 429 or getattr(exc, "delivery_uncertain", False):
                raise
            if is_recoverable_media_error(exc):
                log.warning("ویدیو توسط Telegram رد شد؛ به عکس/متن fallback می‌کنیم: %s", exc)
            else:
                raise
        except Exception as exc:
            if isinstance(exc, TelegramTransportError):
                raise
            log.warning("دریافت/ارسال ویدیو شکست خورد؛ به عکس/متن fallback می‌کنیم: %s", exc)

    if item.get("photo_file_id"):
        try:
            result = send_photo_file_id_to_telegram(title, body, item["photo_file_id"])
            item["_sent_media_type"] = "photo"
            return result
        except TelegramAPIError as exc:
            if exc.status_code == 429 or getattr(exc, "delivery_uncertain", False):
                raise
            log.warning("ارسال عکس file_id شکست خورد؛ fallback فعال می‌شود: %s", sanitize_error_text(exc))
        except Exception as exc:
            if isinstance(exc, TelegramTransportError):
                raise
            log.warning("ارسال عکس file_id شکست خورد؛ fallback فعال می‌شود: %s", sanitize_error_text(exc))

    if photo:
        try:
            result = send_photo_to_telegram(title, body, photo)
            item["_sent_media_type"] = "photo"
            return result
        except TelegramAPIError as exc:
            if exc.status_code == 429 or getattr(exc, "delivery_uncertain", False):
                raise
            if is_recoverable_media_error(exc):
                log.warning("عکس توسط Telegram رد شد؛ به متن fallback می‌کنیم: %s", exc)
            else:
                raise
        except Exception as exc:
            if isinstance(exc, TelegramTransportError):
                raise
            log.warning("دریافت/ارسال عکس شکست خورد؛ به متن fallback می‌کنیم: %s", exc)

    # در صورت شکست رسانه، متن کامل خبر منتشر می‌شود؛ ادامه‌ی جداگانه‌ای برای رسانه نداریم.
    item["_sent_media_type"] = "text"
    public_title = format_public_title(item)
    return send_text_to_telegram(build_text_only_message(item), bold_prefix_len=len(public_title))


# ============================================================
# Dedup / event memory
# ============================================================


def purge_recent_memories(state):
    cutoff = now_ts() - DEDUP_WINDOW_MINUTES * 60
    state["_recent_titles"] = [
        x for x in state.get("_recent_titles", [])
        if x.get("ts", 0) >= cutoff
    ][-500:]

    state["_recent_events"] = [
        x for x in state.get("_recent_events", [])
        if x.get("ts", 0) >= cutoff
    ][-500:]

    event_cutoff = now_ts() - EVENT_MEMORY_WINDOW_MINUTES * 60
    memory = state.get("_event_memory", {})
    if isinstance(memory, dict):
        state["_event_memory"] = {
            k: v for k, v in memory.items()
            if float(v.get("last_published_ts", 0) or 0) >= event_cutoff
        }


def purge_recent_inputs(state):
    cutoff = now_ts() - AI_RECENT_INPUT_WINDOW_MINUTES * 60
    state["_recent_inputs"] = [
        x for x in state.get("_recent_inputs", [])
        if x.get("ts", 0) >= cutoff
    ][-800:]


def _is_input_duplicate_unlocked(state, post):
    """فقط near-exact raw duplicates را قبل از AI حذف می‌کند؛ updateهای واقعی رد نمی‌شوند."""
    purge_recent_inputs(state)
    text = normalize_space(post.get("text", ""))
    if len(text) < 40:
        return False

    current_hash = sha1_text(normalize_for_match(text))
    current_tokens = tokens(text)
    for recent in state.get("_recent_inputs", []):
        if recent.get("hash") == current_hash:
            return True
        old_text = recent.get("text", "")
        if not old_text:
            continue
        sim = combined_similarity(text, "", old_text, "")
        if sim >= AI_LOCAL_DUP_THRESHOLD:
            return True
        old_tokens = set(recent.get("tokens", []))
        if current_tokens and old_tokens:
            overlap = len(current_tokens & old_tokens) / max(1, len(current_tokens | old_tokens))
            if overlap >= 0.75 and min(len(current_tokens), len(old_tokens)) >= 6:
                return True
    return False


def is_input_duplicate(state, post):
    """Serialize duplicate-memory reads and cleanup with other state mutations."""
    with STATE_LOCK:
        return _is_input_duplicate_unlocked(state, post)


def remember_input_posts(state, posts):
    purge_recent_inputs(state)
    for post in posts:
        text = normalize_space(post.get("text", ""))
        if len(text) < 20:
            continue
        state["_recent_inputs"].append({
            "text": text[:3500],
            "hash": sha1_text(normalize_for_match(text)),
            "tokens": list(tokens(text))[:120],
            "ts": now_ts(),
        })
    state["_recent_inputs"] = state["_recent_inputs"][-800:]


def entity_overlap_count(a, b):
    def normalize_entities(value):
        if not isinstance(value, list):
            return set()
        return {
            normalize_for_match(str(x)).strip()
            for x in value
            if str(x).strip()
        }

    ea = normalize_entities(a)
    eb = normalize_entities(b)
    return len(ea & eb)


def update_limit_reached(existing):
    last_ts = float(existing.get("last_published_ts", 0) or 0)
    updates = int(existing.get("updates_24h", 0) or 0)
    if not last_ts:
        return False
    if now_ts() - last_ts > 24 * 3600:
        return False
    return updates >= EVENT_MAX_UPDATES_PER_24H


def compare_event_relation(state, item, existing_items):
    """returns: 'duplicate', 'update', or 'new'.

    This is deliberately stronger than the v3.8 text-only gate: exact event keys,
    semantic signatures, and shared entity anchors can all connect differently
    worded reports of the same event.
    """
    title = item.get("title", "")
    body = item.get("body", "")
    event_key = normalize_event_key(item.get("event_key", ""))
    signature = event_signature(f"{title} {body}")
    is_update = safe_bool(item.get("is_update"))
    novelty = safe_int(item.get("novelty_score"), 0)
    bucket = item.get("content_bucket", "")
    event_type = item.get("event_type", "")
    entities = item.get("entities", [])
    incoming_uids = {str(x) for x in item.get("source_uids", []) if str(x).strip()}

    for existing in existing_items:
        old_uids = {str(x) for x in existing.get("source_uids", []) if str(x).strip()}
        if incoming_uids and old_uids and incoming_uids & old_uids:
            if update_limit_reached(existing):
                return "duplicate"
            if is_update and novelty >= EVENT_UPDATE_MIN_NOVELTY:
                return "update"
            return "duplicate"
        old_title = existing.get("title", "")
        old_body = existing.get("body", "")
        old_key = normalize_event_key(existing.get("event_key", ""))
        old_signature = existing.get("event_signature", "") or event_signature(f"{old_title} {old_body}")
        old_bucket = existing.get("content_bucket", "")
        old_event_type = existing.get("event_type", "")

        exact_key = bool(event_key and old_key and event_key == old_key)
        full_sim = combined_similarity(title, body, old_title, old_body)
        title_sim = title_similarity(title, old_title)
        sig_sim = token_similarity(signature, old_signature)
        shared_entities = entity_overlap_count(entities, existing.get("entities", []))
        same_bucket = bool(bucket and old_bucket and bucket == old_bucket)
        same_event_type = bool(event_type and old_event_type and event_type == old_event_type)

        related = False
        if exact_key:
            related = True
        elif full_sim >= DEDUP_SIMILARITY_THRESHOLD:
            related = True
        elif sig_sim >= EVENT_SIMILARITY_THRESHOLD:
            related = True
        elif shared_entities >= 2 and same_bucket and (same_event_type or title_sim >= 0.52):
            related = True
        elif shared_entities >= 3 and same_bucket:
            related = True

        if not related:
            continue

        # Once the allowed daily update budget is used, new reports of that same
        # event are treated as duplicate rather than endlessly refreshing the queue.
        if update_limit_reached(existing):
            return "duplicate"

        if is_update and novelty >= EVENT_UPDATE_MIN_NOVELTY:
            return "update"
        return "duplicate"

    return "new"


def recent_memory_items(state):
    purge_recent_memories(state)
    out = []
    for key, value in state.get("_event_memory", {}).items():
        item = dict(value)
        item["event_key"] = key
        out.append(item)
    return out


def is_duplicate(state, title, body, event_key="", is_update=False, novelty_score=0):
    item = {
        "title": title,
        "body": body,
        "event_key": event_key,
        "is_update": is_update,
        "novelty_score": novelty_score,
    }
    relation = compare_event_relation(state, item, recent_memory_items(state))
    return relation == "duplicate"


def update_style_memory(state, item):
    style = state.setdefault("_style_memory", {
        "recent_titles": [],
        "recent_openings": [],
        "entity_counts": {},
    })
    title = normalize_persian_text(item.get("title", ""))
    body = normalize_persian_text(item.get("body", ""))
    if title:
        style.setdefault("recent_titles", []).append(title)
        style["recent_titles"] = style["recent_titles"][-30:]
    if body:
        first_para = body.split("\n\n", 1)[0]
        words = first_para.split()
        opening = " ".join(words[:10])
        if opening:
            style.setdefault("recent_openings", []).append(opening)
            style["recent_openings"] = style["recent_openings"][-20:]

    counts = style.setdefault("entity_counts", {})
    for entity in item.get("entities", []) or []:
        entity = normalize_persian_text(str(entity)).strip()
        if not entity:
            continue
        counts[entity] = int(counts.get(entity, 0)) + 1
    if len(counts) > 300:
        counts = dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:200])
        style["entity_counts"] = counts


def remember_item(state, item):
    purge_recent_memories(state)
    title = normalize_persian_text(item.get("title", ""))
    body = normalize_persian_text(item.get("body", ""))
    event_key = normalize_event_key(item.get("event_key", "")) or event_key_or_signature(title, body)
    signature = event_signature(f"{title} {body}")
    now = now_ts()

    recent_memory = state.setdefault("_event_memory", {})
    old = recent_memory.get(event_key, {})
    updates_24h = int(old.get("updates_24h", 0))
    if old and now - float(old.get("last_published_ts", 0) or 0) <= 24 * 3600:
        if safe_bool(item.get("is_update")):
            updates_24h += 1
    else:
        updates_24h = 0

    source_uids = item.get("source_uids", [])
    if not isinstance(source_uids, list):
        source_uids = []

    recent_memory[event_key] = {
        "title": title,
        "body": body,
        "event_signature": signature,
        "region": item.get("region", "world"),
        "content_bucket": item.get("content_bucket", ""),
        "event_type": item.get("event_type", ""),
        "last_published_ts": now,
        "updates_24h": min(EVENT_MAX_UPDATES_PER_24H + 1, updates_24h),
        "entities": item.get("entities", [])[:10],
        "source_uids": source_uids[:20],
    }

    state["_recent_titles"].append({
        "title": title,
        "body": body,
        "event_key": event_key,
        "ts": now,
    })
    state["_recent_events"].append({
        "tokens": list(tokens(f"{title} {body}"))[:100],
        "event_key": event_key,
        "region": item.get("region", "world"),
        "ts": now,
    })
    state["_recent_titles"] = state["_recent_titles"][-500:]
    state["_recent_events"] = state["_recent_events"][-500:]
    update_style_memory(state, item)


def clear_retry_uids(state, posts):
    retry = state.setdefault("_retry_uids", {})
    for post in posts:
        uid = post.get("uid")
        if uid:
            retry.pop(uid, None)


def mark_retry_uids(state, posts, minutes=None):
    retry = state.setdefault("_retry_uids", {})
    base_minutes = max(3, int(minutes or AI_FAILURE_RETRY_MINUTES))
    now = now_ts()

    for post in posts:
        uid = post.get("uid")
        if not uid:
            continue

        previous = retry.get(uid)
        if isinstance(previous, dict):
            attempts = int(previous.get("attempts", 0) or 0) + 1
        else:
            # سازگاری با state قدیمی که فقط timestamp ذخیره می‌کرد.
            attempts = 1

        if attempts >= AI_RETRY_MAX_ATTEMPTS:
            delay_minutes = AI_RETRY_QUARANTINE_MINUTES
            retry[uid] = {
                "until": now + delay_minutes * 60,
                "attempts": attempts,
                "quarantined": True,
            }
            log.warning(
                "UID پس از %s تلاش ناموفق وارد قرنطینه %s دقیقه‌ای شد: %s",
                attempts, delay_minutes, uid,
            )
            continue

        delay_minutes = min(
            AI_RETRY_MAX_DELAY_MINUTES,
            base_minutes * (2 ** (attempts - 1)),
        )
        retry[uid] = {
            "until": now + delay_minutes * 60,
            "attempts": attempts,
            "quarantined": False,
        }


def _is_retry_blocked_unlocked(state, uid):
    if not uid:
        return False
    retry = state.get("_retry_uids", {})
    entry = retry.get(uid)

    if isinstance(entry, dict):
        until = float(entry.get("until", 0) or 0)
        if until <= now_ts():
            # تلاش‌های عادی باید حفظ شوند تا backoff واقعاً نمایی باشد؛
            # فقط پس از پایان quarantine چرخه retry از نو شروع می‌شود.
            if entry.get("quarantined"):
                retry.pop(uid, None)
            return False
        return True

    # سازگاری با نسخه قبلی: مقدار قدیمی فقط timestamp بود.
    try:
        until = float(entry or 0)
    except (TypeError, ValueError):
        retry.pop(uid, None)
        return False
    if until <= now_ts():
        retry.pop(uid, None)
        return False
    return True


def is_retry_blocked(state, uid):
    with STATE_LOCK:
        return _is_retry_blocked_unlocked(state, uid)


# ============================================================
# Queue
# ============================================================

def item_fingerprint(item):
    base = normalize_for_match(
        f"{item.get('title','')} {item.get('body','')[:600]}"
    )
    return sha1_text(base)[:20]


def queue_relation(state, item):
    queue = state.get("_pending_queue", [])
    if not queue:
        return "new", None
    relation = compare_event_relation(state, item, queue)
    if relation == "update":
        # همان event موجود در صف را برای update تازه پیدا کن.
        for idx, existing in enumerate(queue):
            key_match = (
                normalize_event_key(item.get("event_key", ""))
                and normalize_event_key(item.get("event_key", ""))
                == normalize_event_key(existing.get("event_key", ""))
            )
            sig_match = token_similarity(
                event_signature(f"{item.get('title','')} {item.get('body','')}"),
                event_signature(f"{existing.get('title','')} {existing.get('body','')}"),
            ) >= EVENT_SIMILARITY_THRESHOLD
            if key_match or sig_match:
                return "update", idx
        return "update", None
    if relation == "duplicate":
        return "duplicate", None
    return "new", None


def enqueue_item(state, item):
    queue = state.setdefault("_pending_queue", [])
    item = dict(item)
    item["event_key"] = normalize_event_key(item.get("event_key", "")) or event_key_or_signature(
        item.get("title", ""), item.get("body", "")
    )
    item.setdefault("queued_at", now_ts())
    item.setdefault("admin_id", sha1_text(f"{item.get('queued_at')}|{item.get('title','')}|{item.get('event_key','')}")[:12])
    item.setdefault("important", False)
    item.setdefault("urgent", False)
    item.setdefault("delivery_status", "queued")
    item.setdefault("manual_priority", 0)
    item.setdefault("region", "world")
    item.setdefault("entities", [])
    item["fingerprint"] = item_fingerprint(item)

    # exact fingerprint duplicate
    if any(x.get("fingerprint") == item["fingerprint"] for x in queue):
        return False

    relation, index = queue_relation(state, item)
    if relation == "duplicate":
        return False

    if relation == "update" and index is not None:
        # update جدید جایگزین نسخه قبلی در صف می‌شود؛ ترتیب زمانی صف حفظ می‌شود.
        old = queue[index]
        item["queued_at"] = old.get("queued_at", now_ts())
        queue[index] = item
    else:
        queue.append(item)

    apply_queue_diversity_score(state, item)
    queue.sort(
        key=lambda x: (
            bool(x.get("urgent")),
            x.get("priority_score", 0),
            x.get("queued_at", 0),
        ),
        reverse=True,
    )

    state["_pending_queue"] = queue[:500]
    save_state(state)
    return True


def compute_publish_spacing_minutes(now_dt, state=None):
    settings = admin_settings(state) if state is not None else {}
    start_hour = safe_int(settings.get("active_start_hour", 8), 8)
    afternoon_hour = safe_int(settings.get("afternoon_start_hour", 14), 14)
    end_hour = safe_int(settings.get("active_end_hour", 24), 24)
    if not hour_is_active(now_dt.hour, start_hour, end_hour):
        return None
    if now_dt.hour < afternoon_hour and hour_is_active(now_dt.hour, start_hour, afternoon_hour):
        return max(5, safe_int(settings.get("morning_spacing_minutes", MORNING_POST_SPACING_MINUTES), MORNING_POST_SPACING_MINUTES))
    return max(5, safe_int(settings.get("afternoon_spacing_minutes", AFTERNOON_POST_SPACING_MINUTES), AFTERNOON_POST_SPACING_MINUTES))


def choose_next_queue_item(state):
    queue = state.get("_pending_queue", [])
    if not queue:
        return None, None

    # پست Pinشده در نوبت بعدی انتخاب می‌شود؛ زمان‌بندی آن عمداً به اکنون برمی‌گردد.
    pinned_id = str(state.get("_priority_pin_id", "") or "")
    if pinned_id:
        for pin_index, pin_item in enumerate(queue):
            eligible_status = pin_item.get("delivery_status", "queued") not in {"sending", "uncertain", "partial", "sent"}
            eligible_time = not safe_float(pin_item.get("scheduled_at", 0), 0) or safe_float(pin_item.get("scheduled_at", 0), 0) <= now_ts()
            if (str(pin_item.get("admin_id", "")) == pinned_id and eligible_status and eligible_time
                    and not safe_bool(pin_item.get("manual_approval_required"))
                    and safe_int(pin_item.get("news_score"), 0) >= MIN_NEWS_SCORE):
                return pin_index, dict(pin_item)

    # امتیاز هر بار بر اساس وضعیت تنوع دوباره محاسبه می‌شود.
    candidates = []
    for index, item in enumerate(queue):
        scheduled_at = safe_float(item.get("scheduled_at", 0), 0)
        if scheduled_at and scheduled_at > now_ts():
            continue
        if item.get("delivery_status") in {"sending", "uncertain", "partial", "sent"}:
            continue
        if safe_bool(item.get("manual_approval_required")):
            continue
        item = dict(item)
        score = apply_queue_diversity_score(state, item)
        score += safe_int(item.get("manual_priority", 0), 0)

        news_score = safe_int(item.get("news_score"), 0)
        if news_score < MIN_NEWS_SCORE:
            continue

        # خبر با ارزش خبری بهتر، کمی اولویت بیشتر می‌گیرد؛ اما ساختار کلی
        # region/bucket/urgent همچنان در score اصلی دخیل است.
        score += int(news_score * 0.25)

        queued_at = item.get("queued_at", now_ts())
        age_hours = min(24, max(0, (now_ts() - queued_at) / 3600.0))
        score += min(12, int(age_hours))

        candidates.append((score, index, item))

    if not candidates:
        return None, None

    preferred = [
        item for item in candidates
        if safe_int(item[2].get("news_score"), 0) >= PREFERRED_NEWS_SCORE
    ]
    pool = preferred or candidates

    pool.sort(
        key=lambda x: (
            x[0],
            bool(x[2].get("urgent", False)),
            safe_int(x[2].get("news_score"), 0),
        ),
        reverse=True,
    )
    _, index, chosen = pool[0]
    return index, chosen


def mark_posted_region(state, item):
    region = item.get("region", "world")
    state["_last_posted_region"] = region
    recent = state.setdefault("_last_posted_regions", [])
    recent.append(region)
    state["_last_posted_regions"] = recent[-5:]

    bucket = item.get("content_bucket")
    if bucket:
        bucket_recent = state.setdefault("_last_posted_buckets", [])
        bucket_recent.append(bucket)
        state["_last_posted_buckets"] = bucket_recent[-CONTENT_BUCKET_WINDOW:]


def _process_queue_unlocked(state):
    settings = admin_settings(state)
    pause_until = safe_float(state.get("_paused_until", 0), 0)
    if pause_until and now_ts() >= pause_until:
        state["_paused_until"] = 0
        save_state(state)
    if state.get("_pause_until_important"):
        has_important = any(
            (safe_bool(x.get("urgent")) or safe_bool(x.get("important")))
            and x.get("delivery_status", "queued") not in {"sending", "uncertain", "partial", "sent"}
            and not safe_bool(x.get("manual_approval_required"))
            and (not safe_float(x.get("scheduled_at", 0), 0) or safe_float(x.get("scheduled_at", 0), 0) <= now_ts())
            and safe_int(x.get("news_score"), 0) >= MIN_NEWS_SCORE
            for x in state.get("_pending_queue", [])
        )
        if has_important:
            state["_pause_until_important"] = False
            save_state(state)
        else:
            return
    elif safe_float(state.get("_paused_until", 0), 0) > now_ts():
        return
    if not settings.get("auto_publish_enabled", True) or settings.get("emergency_stop", False):
        return
    if is_test_mode(state):
        return
    ensure_queue_ids(state)
    queue = state.get("_pending_queue", [])
    if not queue:
        return

    now_dt = now_tehran()
    if is_quiet_hour(now_dt, state):
        return

    daily_limit = max(0, safe_int(settings.get("daily_post_limit", 0), 0))
    if daily_limit and published_count_today(state) >= daily_limit:
        return

    spacing = compute_publish_spacing_minutes(now_dt, state)
    if spacing is None:
        return

    last_release = state.get("_last_queue_release_ts", 0)
    if last_release:
        elapsed_minutes = (now_ts() - last_release) / 60.0
        if elapsed_minutes < spacing:
            return

    index, item = choose_next_queue_item(state)
    if item is None:
        return

    # آخرین سد تکراری: هم حافظه منتشرشده و هم update logic بررسی می‌شود.
    relation = compare_event_relation(state, item, recent_memory_items(state))
    if relation == "duplicate":
        log.warning("یک آیتم تکراری در آخرین مرحله از صف حذف شد: %s", item.get("title"))
        record_decision(state, item.get("title"), "از صف حذف‌شده", "در بررسی نهایی پیش از انتشار، تکراری تشخیص داده شد", item.get("news_score"))
        queue.pop(index)
        state["_pending_queue"] = queue
        save_state(state)
        return

    # Write-ahead intent: after a restart an uncertain send will not be auto-sent twice.
    queue_item = next((x for x in queue if str(x.get("admin_id", "")) == str(item.get("admin_id", ""))), None)
    if queue_item is not None:
        queue_item["delivery_status"] = "sending"
        queue_item["delivery_started_ts"] = now_ts()
        queue_item["delivery_attempts"] = safe_int(queue_item.get("delivery_attempts", 0), 0) + 1
        queue_item["_delivery_message_ids"] = []
        save_state(state)
    try:
        with STATE_LOCK:
            dispatch_result = dispatch_item(item, state=state)
        if isinstance(dispatch_result, dict) and dispatch_result.get("test_mode"):
            if queue_item is not None:
                queue_item["delivery_status"] = "queued"
                save_state(state)
            return
    except TelegramAPIError as exc:
        if queue_item is not None:
            partial_ids = getattr(exc, "partial_message_ids", [])
            queue_item["_delivery_message_ids"] = partial_ids
            if partial_ids:
                queue_item["delivery_status"] = "partial"
                queue_item["delivery_error"] = "ارسال چندبخشی ناقص؛ از انتشار خودکار مجدد جلوگیری شد"
            elif getattr(exc, "delivery_uncertain", False):
                queue_item["delivery_status"] = "uncertain"
                queue_item["delivery_uncertain_ts"] = now_ts()
                queue_item["delivery_error"] = sanitize_error_text(exc)[:240]
            else:
                # 4xx/429 normally means the send was rejected, so retry is not ambiguous.
                queue_item["delivery_status"] = "queued"
        record_system_error(state, "Telegram", "انتشار خودکار صف", exc, status_code=exc.status_code)
        if exc.status_code == 429:
            save_state(state)
            log.warning(
                "Telegram rate limit در صف؛ %s ثانیه بعد دوباره تلاش می‌شود.",
                exc.retry_after,
            )
            return
        log.error("خطای Telegram در انتشار صف: %s", sanitize_error_text(exc))
        save_state(state)
        return
    except Exception as exc:
        if queue_item is not None:
            partial_ids = getattr(exc, "partial_message_ids", [])
            queue_item["_delivery_message_ids"] = partial_ids
            queue_item["delivery_status"] = "uncertain" if not partial_ids else "partial"
            queue_item["delivery_uncertain_ts"] = now_ts()
            queue_item["delivery_error"] = sanitize_error_text(exc)[:240]
        record_system_error(state, "Telegram", "انتشار خودکار صف", exc)
        log.error("خطا در انتشار آیتم صف؛ برای جلوگیری از انتشار تکراری وضعیت بررسی دستی ثبت شد: %s", sanitize_error_text(exc))
        save_state(state)
        return

    queue.pop(index)
    state["_pending_queue"] = queue
    if str(state.get("_priority_pin_id", "")) == str(item.get("admin_id", "")):
        state["_priority_pin_id"] = ""
    state["_last_queue_release_ts"] = now_ts()
    mark_posted_region(state, item)
    remember_item(state, item)  # فقط پس از انتشار موفق، نه صرفاً enqueue.
    record_published_item(state, item, dispatch_result, manual=False)
    save_state(state)

    log.info(
        "پست منتشر شد | provider_pipeline=%s | region=%s | score=%s | queue=%s | spacing=%s دقیقه",
        ">".join(AI_PROVIDER_LABELS[x] for x in runtime_provider_order(state)),
        item.get("region"),
        item.get("priority_score"),
        len(queue),
        spacing,
    )


def process_queue(state):
    with STATE_LOCK:
        return _process_queue_unlocked(state)


# ============================================================
# Source grouping / processing
# ============================================================

def combine_original_media(cluster):
    photo = next((p.get("photo") for p in cluster if p.get("photo")), None)
    video = next((p.get("video") for p in cluster if p.get("video")), None)
    source_url = next((p.get("source_url") for p in cluster if p.get("source_url")), "")
    source_urls = [p.get("source_url") for p in cluster if p.get("source_url")]
    return photo, video, source_url, source_urls


def _process_cluster_unlocked(state, source_key, cluster):
    if not cluster:
        return

    source_names = sorted({
        p.get("source_name", source_key)
        for p in cluster
        if p.get("source_name")
    })
    source_name = "، ".join(source_names) if source_names else source_key

    # Stage 1: استخراج رویداد/خبر و نگارش اولیه با زنجیره سه‌موتوره.
    drafts, first_provider = analyze_and_rewrite_safe(state, cluster, source_name)
    if not drafts:
        for post in cluster:
            record_decision(state, post.get("title") or post.get("text", "")[:140], "رد‌شده", "خروجی قابل استفاده از AI دریافت نشد")
        raise ValueError("هیچ draft قابل انتشار از موتور اصلی/کمکی‌ها برنگشت.")

    for draft in drafts:
        title_for_log = draft.get("title") or "ورودی بدون تیتر"
        if not draft.get("relevant"):
            record_decision(state, title_for_log, "رد‌شده", draft.get("duplicate_reason") or "AI خبر را مرتبط/قابل انتشار ندانست", draft.get("news_score"))
        elif safe_bool(draft.get("duplicate")):
            record_decision(state, title_for_log, "رد‌شده", draft.get("duplicate_reason") or "تشخیص تکراری توسط AI", draft.get("news_score"))
    drafts = [x for x in drafts if x.get("relevant")]
    if not drafts:
        save_state(state)
        return

    # Stage 2: ویراستار نهایی به‌صورت هوشمند و شرطی اجرا می‌شود تا ظرفیت رایگان
    # سه سرویس بی‌دلیل مصرف نشود. خبرهای چندمنبعی، update، مهم/فوری و تحلیل‌ها
    # به‌طور خودکار نیازمند بررسی دوم می‌شوند.
    related_inputs = any(
        related_post(cluster[i], cluster[j])
        for i in range(len(cluster))
        for j in range(i)
    ) if len(cluster) > 1 else False
    needs_editor = any(
        safe_bool(x.get("needs_editor"))
        or safe_bool(x.get("urgent"))
        or safe_bool(x.get("is_update"))
        or x.get("style_mode") in {
            "analysis", "geopolitics", "fact_check", "comparison",
            "defense_tech",
        }
        or x.get("length_class") == "full"
        for x in drafts
    ) or related_inputs

    if EDITOR_ENABLED and needs_editor:
        edited_items, editor_provider = final_editor(
            state,
            drafts,
            cluster,
            first_provider,
        )
    else:
        edited_items, editor_provider = drafts, first_provider

    successful_interpretation = False
    shared_photo, shared_video, first_source_url, source_urls = combine_original_media(cluster)

    for result in edited_items[:EVENT_MAX_OUTPUT_ITEMS]:
        if not result.get("relevant"):
            continue
        if safe_bool(result.get("duplicate")):
            log.info(
                "AI خروجی را duplicate تشخیص داد و منتشر نشد: %s",
                result.get("title", "")
            )
            successful_interpretation = True
            continue

        linked_indices = [
            idx for idx in result.get("post_indices", [])
            if 1 <= safe_int(idx, 0) <= len(cluster)
        ]
        linked_posts = [cluster[idx - 1] for idx in linked_indices] if linked_indices else cluster
        result_photo, result_video, result_source_url, result_source_urls = select_result_media(
            result, cluster, linked_posts
        )

        title = normalize_persian_text(result.get("title", ""))
        body = normalize_persian_text(result.get("body", ""))
        if not title or not body:
            continue

        item = {
            "title": title,
            "body": body,
            "significance": normalize_persian_text(result.get("significance", "")),
            "image_query": result.get("image_query", ""),
            "urgent": safe_bool(result.get("urgent")),
            "important": safe_bool(result.get("important")),
            "region": infer_region(f"{title}\n{body}", result.get("region")),
            "content_bucket": result.get("content_bucket") or infer_content_bucket(
                f"{title}\n{body}", result.get("region", "")
            ),
            "event_type": result.get("event_type", "routine"),
            "category": result.get("category", "general"),
            "verification": result.get("verification", "reported"),
            "style_mode": result.get("style_mode", "standard"),
            "length_class": result.get("length_class", "standard"),
            "priority_hint": safe_int(result.get("priority_hint"), 0),
            "news_score": max(0, min(100, safe_int(result.get("news_score"), 0))),
            "source_note": "",
            "event_key": normalize_event_key(result.get("event_key", "")) or event_key_or_signature(title, body),
            "event_signature": event_signature(f"{title} {body}"),
            "is_update": safe_bool(result.get("is_update")),
            "needs_editor": safe_bool(result.get("needs_editor")),
            "source_uids": [p.get("uid") for p in linked_posts if p.get("uid")],
            "source_channels": sorted({str(p.get("_source_key", "")).replace("tg:", "") for p in linked_posts if p.get("_source_key")}),
            "classification_model": next((p.get("classification_model") for p in linked_posts if p.get("classification_model")), "disabled"),
            "ai_provider_stage1": first_provider or "unknown",
            "ai_provider_editor": (editor_provider or "not_used") if needs_editor and EDITOR_ENABLED else "not_used",
            "ai_provider": (f"{first_provider or 'unknown'} + editor:{editor_provider}" if needs_editor and EDITOR_ENABLED else (first_provider or "unknown")),
            "novelty_score": safe_int(result.get("novelty_score"), 0),
            "entities": result.get("entities", [])[:10],
            "source_url": result_source_url or first_source_url,
            "source_urls": (result_source_urls or source_urls)[:10],
            "photo": result_photo or shared_photo,
            "video": result_video or shared_video,
            "queued_at": now_ts(),
        }

        priority_terms = [str(x).lower() for x in state.get("_priority_keywords", []) if str(x).strip()]
        matched_priority_terms = [term for term in priority_terms if term in (title + "\n" + body).lower()]
        if matched_priority_terms:
            item["manual_priority"] = min(5000, 100 * len(matched_priority_terms))
            item["matched_priority_keywords"] = matched_priority_terms[:10]
        item["delivery_status"] = "queued"
        apply_auto_approve_rules(state, item)

        if not is_allowed_content_bucket(item["content_bucket"]):
            log.info("خبر خارج از سبد محتوایی حذف شد: %s", title)
            record_decision(state, title, "رد‌شده", "خارج از سبد محتوایی مجاز", item["news_score"])
            successful_interpretation = True
            continue

        relation = compare_event_relation(
            state,
            item,
            recent_memory_items(state),
        )

        # update واقعی فقط در صورت novelty مناسب و سقف روزانه انجام می‌شود.
        if relation == "duplicate":
            log.info("خروجی تکراری/بدون اطلاعات تازه رد شد: %s", title)
            record_decision(state, title, "رد‌شده", "با حافظه رویدادها/خبرهای منتشرشده تکراری تشخیص داده شد", item["news_score"])
            successful_interpretation = True
            continue

        if relation == "update" and item["novelty_score"] < EVENT_UPDATE_MIN_NOVELTY:
            log.info("update با novelty پایین رد شد: %s", title)
            record_decision(state, title, "رد‌شده", "به‌روزرسانی اطلاعات تازه کافی نداشت", item["news_score"])
            successful_interpretation = True
            continue

        # اگر رسانه منبع نیست، fallback فعلی حفظ می‌شود.
        if not item["photo"] and not item["video"]:
            item["photo"] = choose_image_url(item, linked_posts)

        # قبل از enqueue یک بار صف را بررسی می‌کنیم؛ remember_item دیگر اینجا اجرا نمی‌شود.
        if enqueue_item(state, item):
            record_decision(state, title, "وارد صف", "از فیلترهای ارزش خبری، سبد محتوایی و تکراری عبور کرد", item["news_score"])
            successful_interpretation = True
        else:
            log.info("آیتم به علت duplicate/update صفی کنار گذاشته شد: %s", title)
            record_decision(state, title, "رد‌شده", "تکراری/به‌روزرسانی مشابه مورد موجود در صف", item["news_score"])
            successful_interpretation = True

    if not successful_interpretation:
        raise ValueError("هیچ خروجی معتبر از cluster تولید نشد.")



def process_cluster(state, source_key, cluster):
    with STATE_LOCK:
        return _process_cluster_unlocked(state, source_key, cluster)


def process_source(state, source_key, posts):
    """سازگاری با ساختار قبلی؛ پردازش واقعی در process_once چندمنبعی انجام می‌شود."""
    if not posts:
        return
    processed = set(state.get(source_key, []))
    new_posts = [p for p in posts if p.get("uid") not in processed and not is_retry_blocked(state, p.get("uid"))]
    if not new_posts:
        return
    clusters = cluster_posts(new_posts)
    for cluster in clusters:
        uids = [p.get("uid") for p in cluster]
        try:
            process_cluster(state, source_key, cluster)
            current = state.setdefault(source_key, [])
            for uid in uids:
                if uid and uid not in current:
                    current.append(uid)
            state[source_key] = current[-500:]
            clear_retry_uids(state, cluster)
            save_state(state)
        except Exception as exc:
            mark_retry_uids(state, cluster)
            save_state(state)
            log.error("cluster شکست خورد؛ retry delayed source=%s error=%s", source_key, exc)


# ============================================================
# Scheduled messages / cleanup
# ============================================================

def purge_unimportant_queue(state, today_str):
    if state.get("_last_queue_purge_date") == today_str:
        return

    queue = state.get("_pending_queue", [])
    before = len(queue)

    # خبرهای مهم و urgent باقی می‌مانند؛ باقی‌مانده‌های معمولی شب حذف می‌شوند.
    state["_pending_queue"] = [
        item for item in queue
        if safe_bool(item.get("important")) or safe_bool(item.get("urgent"))
    ]
    state["_last_queue_purge_date"] = today_str

    removed = before - len(state["_pending_queue"])
    if removed:
        log.info(
            "%s خبر عادیِ باقی‌مانده در نیمه‌شب حذف شد.",
            removed,
        )

    save_state(state)


def _check_scheduled_messages_unlocked(state):
    now = now_tehran()
    today_str = now.strftime("%Y-%m-%d")
    settings = admin_settings(state)
    scheduled_posts_enabled = (
        settings.get("auto_publish_enabled", True)
        and not settings.get("emergency_stop", False)
        and not is_test_mode(state)
    )
    start_hour = safe_int(settings.get("active_start_hour", MORNING_HOUR), MORNING_HOUR)
    configured_end = safe_int(settings.get("active_end_hour", 24), 24)
    end_message_hour = configured_end % 24  # پایان 24 یعنی ساعت صفر روز بعد

    # پیام‌های آغاز/پایان روز نیز از برنامه زمانی پنل پیروی می‌کنند.
    if scheduled_posts_enabled and now.hour == start_hour and state.get("_last_morning_date") != today_str:
        try:
            send_text_to_telegram(MORNING_MESSAGE)
            state["_last_morning_date"] = today_str
            save_state(state)
            log.info("پیام آغاز بازه فعالیت ارسال شد.")
        except Exception as exc:
            log.error("ارسال پیام آغاز بازه فعالیت شکست خورد: %s", sanitize_error_text(exc))

    if scheduled_posts_enabled and now.hour == end_message_hour:
        # اگر فعالیت دقیقاً از ساعت صفر تا پایان 24 تنظیم شده باشد، پیام آغاز و پایان هم‌زمان نشوند.
        if end_message_hour != start_hour and state.get("_last_night_date") != today_str:
            try:
                send_text_to_telegram(NIGHT_MESSAGE)
                state["_last_night_date"] = today_str
                save_state(state)
                log.info("پیام پایان بازه فعالیت ارسال شد.")
            except Exception as exc:
                log.error("ارسال پیام پایان بازه فعالیت شکست خورد: %s", sanitize_error_text(exc))
        purge_unimportant_queue(state, today_str)


def check_scheduled_messages(state):
    with STATE_LOCK:
        return _check_scheduled_messages_unlocked(state)


# ============================================================
# Polling
# ============================================================

def build_gemini_batches(clusters):
    """نام قدیمی حفظ شده تا سازگاری کد بماند؛ داخلش batching سه‌موتوره انجام می‌شود."""
    batches = []
    current = []
    current_size = 0

    for cluster in clusters:
        if not cluster:
            continue
        cluster_size = len(cluster)
        if current and current_size + cluster_size > AI_BATCH_MAX_POSTS:
            batches.append(current)
            current = []
            current_size = 0

        if cluster_size > AI_BATCH_MAX_POSTS:
            for start_idx in range(0, cluster_size, AI_BATCH_MAX_POSTS):
                part = cluster[start_idx:start_idx + AI_BATCH_MAX_POSTS]
                if current:
                    batches.append(current)
                    current = []
                    current_size = 0
                batches.append(part)
            continue

        current.extend(cluster)
        current_size += cluster_size

    if current:
        batches.append(current)
    return batches



def log_runtime_health(state):
    """لاگ سبک دوره‌ای برای مشاهده صف و وضعیت موتورهای AI در Railway."""
    now = now_ts()
    last = float(state.get("_last_health_log_ts", 0) or 0)
    if last and now - last < 10 * 60:
        return

    state["_last_health_log_ts"] = now
    queue_len = len(state.get("_pending_queue", []))
    provider_parts = []
    for provider in AI_PROVIDER_ORDER:
        pstate = provider_state(state, provider)
        cooldown = max(0, int(float(pstate.get("cooldown_until", 0) or 0) - now))
        provider_parts.append(
            f"{AI_PROVIDER_LABELS[provider]}:cooldown={cooldown}s"
        )

    log.info(
        "health | queue=%s | providers=%s | state=%s",
        queue_len,
        " ".join(provider_parts),
        STATE_FILE,
    )


def process_once(state):
    """
    چرخه v5: گردآوری چندمنبعی، پردازش محدودشده AI و انتشار قابل‌کنترل.
    1) جمع‌آوری از همه منابع
    2) حذف near-exact duplicate خام
    3) خوشه‌بندی و batching چندمنبعی
    4) Stage 1 با Groq > Gemini > Mistral
    5) Stage 2 با provider بعدی برای توزیع بار، و fallback به draft در صورت شکست editor
    6) dedup سه‌لایه + event memory + update merge
    7) صف و زمان‌بندی ثابت
    """
    all_new_posts = []
    with STATE_LOCK:
        runtime = state.setdefault("_runtime", {})
        runtime["last_source_check_ts"] = now_ts()
        runtime["last_cycle_error"] = ""
        save_state(state)

    for channel in runtime_source_channels(state):
        source_key = f"tg:{channel}"
        posts = fetch_channel_posts(channel, state=state)
        with STATE_LOCK:
            source_stats = state.setdefault("_source_stats", {}).setdefault(channel, {"scanned": 0, "fetched": 0, "new": 0, "published": 0, "rejected": 0, "errors": 0})
            source_stats["scanned"] = safe_int(source_stats.get("scanned"), 0) + len(posts)
            save_state(state)
        if not posts:
            continue

        processed = set(state.get(source_key, []))
        if not state.get("_source_initialized", {}).get(source_key):
            with STATE_LOCK:
                state.setdefault("_source_initialized", {})[source_key] = True
                state[source_key] = [p["uid"] for p in posts[-500:]]
                save_state(state)
            log.info(
                "منبع %s برای اولین بار ثبت شد؛ پست‌های موجود قدیمی پردازش نشدند.",
                source_key,
            )
            continue

        for post in posts:
            uid = post.get("uid")
            if uid in processed:
                continue
            if is_retry_blocked(state, uid):
                continue
            post["_source_key"] = source_key
            with STATE_LOCK:
                source_stats = state.setdefault("_source_stats", {}).setdefault(channel, {"scanned": 0, "fetched": 0, "new": 0, "published": 0, "rejected": 0, "errors": 0})
                source_stats["fetched"] = safe_int(source_stats.get("fetched"), 0) + 1
            if is_blocked_or_silenced(state, post.get("text", "")):
                reason = block_or_silence_reason(state, post.get("text", ""))
                with STATE_LOCK:
                    record_decision(state, post.get("title") or post.get("text", "")[:140], "رد‌شده", reason)
                    current = state.setdefault(source_key, [])
                    if uid not in current: current.append(uid)
                    state[source_key] = current[-500:]
                    source_stats["rejected"] = safe_int(source_stats.get("rejected"), 0) + 1
                    save_state(state)
                continue
            all_new_posts.append(post)
            with STATE_LOCK:
                source_stats["new"] = safe_int(source_stats.get("new"), 0) + 1

    if all_new_posts:
        all_new_posts = cheap_classify_posts(state, all_new_posts)
        filtered_new_posts = []
        locally_skipped = 0
        for post in all_new_posts:
            with STATE_LOCK:
                duplicate_input = is_input_duplicate(state, post)
                if duplicate_input:
                    locally_skipped += 1
                    source_key = post.get("_source_key")
                    uid = post.get("uid")
                    if source_key and uid:
                        current = state.setdefault(source_key, [])
                        if uid not in current:
                            current.append(uid)
                        state[source_key] = current[-500:]
                    save_state(state)
            if duplicate_input:
                continue
            filtered_new_posts.append(post)

        if locally_skipped:
            log.info("%s پست near-exact duplicate قبل از AI حذف شد.", locally_skipped)

        clusters = cluster_posts(filtered_new_posts)
        ai_batches = build_gemini_batches(clusters)
        log.info(
            "%s پست جدید -> %s cluster -> %s batch چندمنبعی | AI=%s",
            len(filtered_new_posts), len(clusters), len(ai_batches),
            ">".join(AI_PROVIDER_LABELS[x] for x in runtime_provider_order(state)) or "بدون ارائه‌دهنده فعال"
        )

        for batch in ai_batches:
            try:
                process_cluster(state, "multi-source", batch)
                with STATE_LOCK:
                    remember_input_posts(state, batch)
                    for post in batch:
                        source_key = post.get("_source_key")
                        uid = post.get("uid")
                        if not source_key or not uid:
                            continue
                        current = state.setdefault(source_key, [])
                        if uid not in current:
                            current.append(uid)
                        state[source_key] = current[-500:]
                    clear_retry_uids(state, batch)
                    save_state(state)
                log.info(
                    "batch با %s پست پردازش شد؛ sources=%s",
                    len(batch),
                    len({p.get("_source_key") for p in batch}),
                )

            except AIAllProvidersUnavailable as exc:
                with STATE_LOCK:
                    mark_retry_uids(state, batch, minutes=AI_FAILURE_RETRY_MINUTES)
                    save_state(state)
                log.warning(
                    "همه موتورهای AI موقتاً در دسترس نبودند؛ batch برای retry نگه داشته شد: %s",
                    exc,
                )
            except Exception as exc:
                with STATE_LOCK:
                    mark_retry_uids(state, batch)
                    save_state(state)
                log.error(
                    "batch شکست خورد و UIDها تا retry بعدی مصرف‌شده تلقی نشدند: %s",
                    exc,
                )

    with STATE_LOCK:
        state.setdefault("_runtime", {})["last_source_count"] = len(all_new_posts)
    check_scheduled_messages(state)
    process_queue(state)
    with STATE_LOCK:
        purge_recent_memories(state)
        purge_recent_inputs(state)
        log_runtime_health(state)
        state.setdefault("_runtime", {})["last_cycle_ok_ts"] = now_ts()
        save_state(state)
    # تماس شبکه‌ای هشدارها بیرون از قفل state انجام می‌شود تا پنل بی‌جهت منتظر نماند.
    check_v53_alerts(state)
    maybe_send_weekly_summary(state)


# ============================================================
# v5 runtime controls, diagnostics and Telegram admin panel
# ============================================================


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def admin_settings(state):
    if state is None:
        return {}
    fresh = default_state()["_admin_settings"]
    settings = state.setdefault("_admin_settings", {})
    for key, value in fresh.items():
        settings.setdefault(key, value)
    if not isinstance(settings.get("ai_enabled"), dict):
        settings["ai_enabled"] = {provider: True for provider in AI_PROVIDER_ORDER}
    if not isinstance(settings.get("ai_order"), list):
        settings["ai_order"] = list(AI_PROVIDER_ORDER)
    if not isinstance(settings.get("source_channels"), list):
        settings["source_channels"] = list(SOURCE_CHANNELS)
    return settings


def runtime_source_channels(state):
    configured = admin_settings(state).get("source_channels", SOURCE_CHANNELS) if state is not None else SOURCE_CHANNELS
    disabled = set(state.get("_disabled_sources", [])) if state is not None else set()
    return [str(x).strip().lstrip("@") for x in configured if str(x).strip() and str(x).strip().lstrip("@") not in disabled]


def runtime_provider_order(state):
    settings = admin_settings(state)
    order = settings.get("ai_order") or list(AI_PROVIDER_ORDER)
    order = [x for x in order if x in AI_PROVIDER_ORDER]
    order.extend(x for x in AI_PROVIDER_ORDER if x not in order)
    enabled = settings.get("ai_enabled", {})
    return [x for x in order if enabled.get(x, True)]


def runtime_ai_attempts(state):
    return max(1, min(2, safe_int(admin_settings(state).get("ai_attempts", AI_MAX_ATTEMPTS_PER_PROVIDER), AI_MAX_ATTEMPTS_PER_PROVIDER)))


def is_test_mode(state=None):
    # Railway TEST_MODE=true همیشه قفل ایمنی بالادستی است؛ پنل نمی‌تواند آن را خاموش کند.
    if TEST_MODE:
        return True
    if state is not None:
        override = admin_settings(state).get("test_mode_override")
        if isinstance(override, bool):
            return override
    return False


def hour_is_active(hour, start_hour, end_hour):
    start_hour = max(0, min(23, safe_int(start_hour, 8)))
    end_hour = max(0, min(24, safe_int(end_hour, 24)))
    hour = safe_int(hour, 0) % 24
    if end_hour == 24 and start_hour < 24:
        return hour >= start_hour
    if start_hour < end_hour:
        return start_hour <= hour < end_hour
    if start_hour > end_hour:
        return hour >= start_hour or hour < end_hour
    return False


def is_quiet_hour(now_dt, state=None):
    settings = admin_settings(state) if state is not None else {}
    return not hour_is_active(
        now_dt.hour,
        settings.get("active_start_hour", 8),
        settings.get("active_end_hour", 24),
    )


def sanitize_error_text(value):
    text = str(value or "")
    text = re.sub(r"(?i)(api[_ -]?key|token|authorization)\s*[:=]\s*[^\s,;]+", r"\1=[REDACTED]", text)
    text = re.sub(r"\b\d{5,}:[A-Za-z0-9_-]{20,}\b", "[BOT_TOKEN_REDACTED]", text)
    # Avoid leaking accidental raw API-key-shaped strings into the admin panel.
    text = re.sub(r"\bAIza[0-9A-Za-z_-]{20,}\b", "[API_KEY_REDACTED]", text)
    text = re.sub(r"\bgsk_[A-Za-z0-9_-]{12,}\b", "[API_KEY_REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{16,}\b", "[API_KEY_REDACTED]", text)
    for secret in [BOT_TOKEN] + [key for keys in AI_PROVIDER_KEYS.values() for key in keys]:
        if secret and len(secret) >= 8:
            text = text.replace(secret, "[SECRET_REDACTED]")
    return truncate_text(text.replace("\x00", ""), 700)


def classify_provider_error(status_code, detail=""):
    if status_code in (401, 403):
        return "احتمال مشکل کلید یا مجوز"
    if status_code == 429:
        return "محدودیت درخواست / سهمیه"
    if status_code in (400, 404):
        return "احتمال مدل یا قالب درخواست"
    if status_code >= 500:
        return "اختلال سرویس ارائه‌دهنده"
    if status_code in (-1, -2):
        return "شبکه، پاسخ نامعتبر یا timeout"
    if "timeout" in str(detail).lower() or "connection" in str(detail).lower():
        return "شبکه یا timeout"
    return "خطای ارائه‌دهنده"


def record_system_error(state, area, context, detail, provider=None, status_code=0):
    entry = {
        "ts": now_ts(),
        "area": str(area)[:40],
        "context": str(context)[:120],
        "provider": provider or "",
        "status_code": safe_int(status_code, 0),
        "category": classify_provider_error(safe_int(status_code, 0), detail) if area == "AI" else "خطای اجرایی",
        "detail": sanitize_error_text(detail),
    }
    errors = state.setdefault("_system_errors", [])
    errors.append(entry)
    state["_system_errors"] = errors[-MAX_ADMIN_LOGS:]


def record_provider_error(state, provider, task_name, exc):
    status = safe_int(getattr(exc, "status_code", 0), 0)
    detail = sanitize_error_text(str(exc))
    pstate = provider_state(state, provider)
    pstate["last_error"] = {
        "ts": now_ts(), "task": str(task_name)[:60], "status_code": status,
        "category": classify_provider_error(status, detail), "detail": detail,
    }
    pstate["last_status"] = status
    usage = pstate.setdefault("usage", {"requests": 0, "successes": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0})
    usage["errors"] = safe_int(usage.get("errors"), 0) + 1
    record_system_error(state, "AI", f"{AI_PROVIDER_LABELS[provider]} / {task_name}", detail, provider, status)
    save_state(state)


def record_admin_activity(state, user_id, action, detail=""):
    entries = state.setdefault("_admin_activity", [])
    entries.append({
        "ts": now_ts(), "user_id": safe_int(user_id, 0),
        "role": admin_role(state, user_id) or "unknown",
        "action": str(action)[:80], "detail": sanitize_error_text(detail)[:220],
    })
    state["_admin_activity"] = entries[-MAX_ADMIN_LOGS:]


def record_decision(state, title, outcome, reason, score=None):
    entries = state.setdefault("_recent_decisions", [])
    entries.append({
        "ts": now_ts(), "title": normalize_space(str(title or "بدون تیتر"))[:180],
        "outcome": str(outcome)[:30], "reason": sanitize_error_text(reason)[:220],
        "score": safe_int(score, -1) if score is not None else -1,
    })
    state["_recent_decisions"] = entries[-MAX_ADMIN_LOGS:]


def published_count_today(state):
    today = now_tehran().date()
    count = 0
    for item in state.get("_published_history", []):
        timestamp = safe_float(item.get("ts"), 0)
        if timestamp and datetime.fromtimestamp(timestamp, TEHRAN_TZ or timezone.utc).date() == today:
            count += 1
    return count


def telegram_result_message_id(result):
    if not isinstance(result, dict):
        return 0
    message = result.get("result", {})
    return safe_int(message.get("message_id"), 0) if isinstance(message, dict) else 0


def record_published_item(state, item, result, manual=False):
    if isinstance(result, dict) and result.get("test_mode"):
        return None
    record = {
        "history_id": sha1_text(f"{now_ts()}|{item.get('title','')}|{telegram_result_message_id(result)}")[:12],
        "ts": now_ts(), "title": str(item.get("title", ""))[:240],
        "body": str(item.get("body", ""))[:6500],
        "message_id": telegram_result_message_id(result),
        "message_ids": (result.get("_all_message_ids", []) if isinstance(result, dict) else []),
        "chat_id": TARGET_CHAT_ID,
        "media_type": item.get("_sent_media_type") or ("video" if item.get("video") else ("photo" if item.get("photo") or item.get("photo_file_id") else "text")),
        "photo": item.get("photo"), "photo_file_id": item.get("photo_file_id"),
        "video": item.get("video"),
        "manual": bool(manual), "deleted": False,
        "news_score": safe_int(item.get("news_score"), 0),
        "event_key": str(item.get("event_key", ""))[:140],
        "ai_provider": str(item.get("ai_provider", item.get("_ai_provider", "unknown")))[:80],
        "ai_provider_stage1": str(item.get("ai_provider_stage1", "unknown"))[:40],
        "ai_provider_editor": str(item.get("ai_provider_editor", "not_used"))[:40],
        "classification_model": str(item.get("classification_model", "disabled"))[:80],
        "verification": str(item.get("verification", "reported"))[:40],
        "source_channels": list(item.get("source_channels", []))[:20],
        "source_urls": list(item.get("source_urls", []))[:20],
        "content_bucket": str(item.get("content_bucket", "world"))[:60],
        "region": str(item.get("region", "world"))[:40],
    }
    history = state.setdefault("_published_history", [])
    history.append(record)
    state["_published_history"] = history[-MAX_ADMIN_HISTORY:]
    for source in record.get("source_channels", []):
        stat = state.setdefault("_source_stats", {}).setdefault(source, {"fetched": 0, "new": 0, "published": 0, "rejected": 0, "errors": 0})
        stat["published"] = safe_int(stat.get("published"), 0) + 1
    runtime = state.setdefault("_runtime", {})
    runtime["last_posted_ts"] = record["ts"]
    return record


def admin_role(state, user_id):
    uid = safe_int(user_id, 0)
    if OWNER_TELEGRAM_ID and uid == OWNER_TELEGRAM_ID:
        return "owner"
    entry = state.get("_admin_users", {}).get(str(uid), {})
    role = entry.get("role") if isinstance(entry, dict) else None
    return role if role in {"manager", "observer"} else None


def admin_can(role, area):
    permissions = {
        "owner": {"stats", "settings", "news", "queue", "ai", "admins", "sources", "logs", "tools", "decisions"},
        "manager": {"stats", "settings", "news", "queue", "ai", "sources", "logs", "tools", "decisions"},
        "observer": {"stats", "ai_view", "logs", "decisions", "history_view", "diagnose"},
    }
    return area in permissions.get(role, set())


def private_admin_user(update):
    callback = update.get("callback_query")
    if callback:
        user = callback.get("from", {})
        message = callback.get("message", {})
        chat = message.get("chat", {})
        return user, chat if chat.get("type") == "private" else {}
    message = update.get("message", {})
    user = message.get("from", {})
    chat = message.get("chat", {})
    return user, chat if chat.get("type") == "private" else {}


def admin_button(text, data):
    return {"text": str(text)[:60], "callback_data": str(data)[:64]}


def admin_keyboard(rows):
    return {"inline_keyboard": rows}


def admin_render(chat_id, text, rows=None, edit_message_id=None):
    text = str(text or "")
    if len(text) > 3900:
        text = truncate_text(text, 3900)
    keyboard = admin_keyboard(rows or [[admin_button("🏠 منوی اصلی", "menu")]])
    if edit_message_id:
        try:
            telegram_request("editMessageText", data={
                "chat_id": chat_id, "message_id": edit_message_id,
                "text": text, "reply_markup": json.dumps(keyboard, ensure_ascii=False),
                "disable_web_page_preview": "true",
            }, timeout=20)
            return
        except TelegramAPIError as exc:
            if "message is not modified" in str(exc).lower():
                return
            log.debug("ویرایش پیام پنل ممکن نشد؛ پیام جدید ساخته می‌شود: %s", sanitize_error_text(str(exc)))
        except Exception:
            pass
    try:
        telegram_request("sendMessage", data={
            "chat_id": chat_id, "text": text,
            "reply_markup": json.dumps(keyboard, ensure_ascii=False),
            "disable_web_page_preview": "true",
        }, timeout=20)
    except Exception as exc:
        log.warning("ارسال پنل مدیریت شکست خورد: %s", sanitize_error_text(exc))


def admin_answer_callback(callback_id, text=""):
    try:
        telegram_request("answerCallbackQuery", data={"callback_query_id": callback_id, "text": str(text)[:180], "show_alert": "false"}, timeout=10)
    except Exception:
        pass


def admin_show_menu(chat_id, message_id=None, role="observer"):
    if role == "observer":
        rows = [
            [admin_button("📊 آمار و وضعیت ربات", "stats")],
            [admin_button("🤖 وضعیت هوش مصنوعی", "ai")],
            [admin_button("🗃 تاریخچه انتشار", "history")],
            [admin_button("📝 گزارش فعالیت‌ها", "logs")],
            [admin_button("🧾 دلایل انتخاب/رد خبر", "decisions")],
            [admin_button("🩺 عیب‌یابی", "diagnose")],
        ]
    else:
        rows = [
            [admin_button("📊 آمار و وضعیت ربات", "stats")],
            [admin_button("⚙️ تنظیمات ربات", "settings")],
            [admin_button("📰 مدیریت اخبار", "news")],
            [admin_button("📥 مدیریت صف انتشار", "queue")],
            [admin_button("🤖 مدیریت هوش مصنوعی", "ai")],
        ]
        if role == "owner":
            rows.append([admin_button("👥 مدیریت ادمین‌ها", "admins")])
        rows.extend([
            [admin_button("📡 مدیریت منابع", "sources")],
            [admin_button("📝 گزارش فعالیت‌ها", "logs")],
            [admin_button("🛠 ابزارهای پیشرفته", "tools")],
        ])
    if role != "observer":
        rows.append([admin_button("🧰 قابلیت‌های V5.3", "v53")])
    admin_render(chat_id, f"🦅 پنل مدیریت رپتور — نسخه {BOT_VERSION}\nیک بخش را انتخاب کن:", rows, message_id)


def render_stats(state, chat_id, message_id=None):
    rt = state.setdefault("_runtime", {})
    uptime = max(0, int(now_ts() - safe_float(rt.get("started_ts"), now_ts())))
    hours, remainder = divmod(uptime, 3600)
    minutes = remainder // 60
    last_source = rt.get("last_source_check_ts")
    last_post = rt.get("last_posted_ts")
    settings = admin_settings(state)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(STATE_FILE)) or ".", exist_ok=True)
        storage_ok = os.access(os.path.dirname(os.path.abspath(STATE_FILE)) or ".", os.W_OK)
    except Exception:
        storage_ok = False
    p_lines = []
    for provider in AI_PROVIDER_ORDER:
        ps = provider_state(state, provider)
        configured = len(AI_PROVIDER_KEYS[provider])
        disabled = not settings.get("ai_enabled", {}).get(provider, True)
        cooldown = max(0, int(safe_float(ps.get("cooldown_until"), 0) - now_ts()))
        status = "غیرفعال از پنل" if disabled else ("بدون کلید" if not configured else (f"محدود/منتظر {cooldown}ث" if cooldown else "آماده/آخرین وضعیت " + str(ps.get("last_status", "—"))))
        p_lines.append(f"• {AI_PROVIDER_LABELS[provider]}: {status} | کلیدهای تنظیم‌شده: {configured}")
    source_error_line = f"آخرین خطای دریافت: {sanitize_error_text(rt.get('last_source_error'))}\n" if rt.get("last_source_error") else ""
    text = (
        f"📊 وضعیت رپتور — آخرین به‌روزرسانی: {now_tehran().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"وضعیت چرخه: {'خطای اخیر دارد' if rt.get('last_cycle_error') else 'فعال'}\n"
        f"مدت از آخرین راه‌اندازی: {hours} ساعت و {minutes} دقیقه\n"
        f"آخرین بررسی منابع: {datetime.fromtimestamp(last_source, TEHRAN_TZ or timezone.utc).strftime('%Y-%m-%d %H:%M') if last_source else 'هنوز ثبت نشده'}\n"
        f"آخرین دریافت موفق: {datetime.fromtimestamp(safe_float(rt.get('last_source_success_ts'), 0), TEHRAN_TZ or timezone.utc).strftime('%Y-%m-%d %H:%M') if safe_float(rt.get('last_source_success_ts'), 0) else 'هنوز ثبت نشده'}\n"
        f"پست‌های تازه در آخرین چرخه: {safe_int(rt.get('last_source_count'), 0)}\n"
        f"{source_error_line}"
        f"آخرین انتشار: {datetime.fromtimestamp(last_post, TEHRAN_TZ or timezone.utc).strftime('%Y-%m-%d %H:%M') if last_post else 'هنوز ثبت نشده'}\n"
        f"صف انتشار: {len(state.get('_pending_queue', []))} خبر\n"
        f"توقف زمان‌دار: {('تا ' + datetime.fromtimestamp(safe_float(state.get('_paused_until'), 0), TEHRAN_TZ or timezone.utc).strftime('%m-%d %H:%M')) if safe_float(state.get('_paused_until'), 0) > now_ts() else ('تا خبر مهم' if state.get('_pause_until_important') else 'ندارد')}\n"
        f"انتشار خودکار: {'روشن' if settings.get('auto_publish_enabled', True) else 'خاموش'}\n"
        f"توقف اضطراری: {'فعال' if settings.get('emergency_stop') else 'غیرفعال'}\n"
        f"حالت آزمایشی: {'روشن' if is_test_mode(state) else 'خاموش'}\n"
        f"ذخیره‌سازی: {'قابل‌نوشتن' if storage_ok else 'بررسی شود'}\n\n"
        "وضعیت ارائه‌دهندگان:\n" + "\n".join(p_lines)
    )
    rows = [[admin_button("🔄 تازه‌سازی", "stats")], [admin_button("🩺 عیب‌یابی", "diagnose")], [admin_button("🏠 منوی اصلی", "menu")]]
    admin_render(chat_id, text, rows, message_id)


def render_settings(state, chat_id, message_id=None):
    st = admin_settings(state)
    schedule = f"فعال: ساعت {st.get('active_start_hour', 8)} تا {st.get('active_end_hour', 24)}؛ شروع بازه دوم: {st.get('afternoon_start_hour', 14)}"
    text = (
        "⚙️ تنظیمات ربات\n\n"
        f"انتشار خودکار: {'روشن' if st.get('auto_publish_enabled', True) else 'خاموش'}\n"
        f"توقف اضطراری: {'فعال' if st.get('emergency_stop') else 'غیرفعال'}\n"
        f"فاصله بازه اول: {st.get('morning_spacing_minutes', 60)} دقیقه\n"
        f"فاصله بازه دوم: {st.get('afternoon_spacing_minutes', 45)} دقیقه\n"
        f"{schedule}\n"
        f"سقف روزانه: {st.get('daily_post_limit', 0) or 'بدون سقف'}\n\n"
        "برای فاصله زمانی عدد دقیقه، برای ساعت‌های فعالیت سه عدد با ویرگول وارد کن: شروع,شروع‌بازه‌دوم,پایان (مثال: 8,14,24). پایان می‌تواند 24 باشد."
    )
    rows = [
        [admin_button("🔁 تغییر انتشار خودکار", "setting|auto")],
        [admin_button("🚨 توقف اضطراری", "setting|emergency")],
        [admin_button("⏱ فاصله بازه اول", "setprompt|morning_spacing")],
        [admin_button("⏱ فاصله بازه دوم", "setprompt|afternoon_spacing")],
        [admin_button("🕒 ساعات فعالیت", "setprompt|active_hours")],
        [admin_button("📈 سقف انتشار روزانه", "setprompt|daily_limit")],
        [admin_button("♻️ بازنشانی تنظیمات", "confirm|resetsettings|1")],
        [admin_button("⬅️ منوی اصلی", "menu")],
    ]
    admin_render(chat_id, text, rows, message_id)


def ensure_queue_ids(state):
    changed = False
    for item in state.get("_pending_queue", []):
        if not item.get("admin_id"):
            item["admin_id"] = sha1_text(f"{item.get('queued_at', 0)}|{item.get('title', '')}|{item.get('event_key', '')}")[:12]
            changed = True
    if changed:
        save_state(state)


def find_queue_item(state, item_id):
    ensure_queue_ids(state)
    for item in state.get("_pending_queue", []):
        if str(item.get("admin_id", "")) == str(item_id):
            return item
    return None


def render_queue(state, chat_id, message_id=None, page=0):
    ensure_queue_ids(state)
    queue = state.get("_pending_queue", [])
    start = max(0, page) * ADMIN_PAGE_SIZE
    subset = queue[start:start + ADMIN_PAGE_SIZE]
    rows = []
    for item in subset:
        title = normalize_space(item.get("title", "بدون تیتر"))[:42]
        score = safe_int(item.get("news_score"), 0)
        rows.append([admin_button(f"{title} · {score}", f"q|{item.get('admin_id')}")])
    nav = []
    if page > 0:
        nav.append(admin_button("◀️ قبلی", f"queuepage|{page-1}"))
    if start + ADMIN_PAGE_SIZE < len(queue):
        nav.append(admin_button("بعدی ▶️", f"queuepage|{page+1}"))
    if nav:
        rows.append(nav)
    rows.extend([
        [admin_button("🧹 حذف موارد تکراری واضح", "dedupqueue")],
        [admin_button("📰 تاریخچه منتشرشده", "history")],
        [admin_button("⬅️ منوی اصلی", "menu")],
    ])
    text = f"📥 صف انتشار — {len(queue)} خبر\nآخرین بررسی پنل: {now_tehran().strftime('%Y-%m-%d %H:%M:%S')}\nصفحه {page+1}\nبرای پیش‌نمایش روی خبر بزن."
    if not queue:
        text += "\n\nصف خالی است."
    admin_render(chat_id, text, rows, message_id)


def render_queue_item(state, chat_id, item_id, message_id=None):
    item = find_queue_item(state, item_id)
    if not item:
        admin_render(chat_id, "این خبر دیگر در صف وجود ندارد.", [[admin_button("⬅️ بازگشت به صف", "queue")]], message_id)
        return
    scheduled = safe_float(item.get("scheduled_at", 0), 0)
    scheduled_text = datetime.fromtimestamp(scheduled, TEHRAN_TZ or timezone.utc).strftime('%Y-%m-%d %H:%M') if scheduled else "به‌محض فراهم‌شدن زمان انتشار"
    body = format_public_body(item)
    text = (
        "📰 پیش‌نمایش خبر صف\n\n"
        f"تیتر: {item.get('title', '')}\n\n"
        f"{body}\n\n"
        f"امتیاز: {safe_int(item.get('news_score'), 0)} | منطقه: {item.get('region', '—')}\n"
        f"منبع: {', '.join(item.get('source_channels', [])) or item.get('source_name', 'ثبت نشده')}\n"
        f"هوش مصنوعی: {item.get('ai_provider', item.get('ai_provider_stage1', 'نامشخص'))}\n"
        f"فیلتر اولیه: {item.get('classification_model', 'غیرفعال')}\n"
        f"اعتبار: {item.get('verification', 'reported')} | وضعیت تحویل: {item.get('delivery_status', 'queued')}\n"
        f"تأیید: {'نیازمند تأیید دستی' if item.get('manual_approval_required') else ('تأیید خودکار' if item.get('auto_approved_by_rule') else 'طبق سیاست جاری')}\n"
        f"رسانه: {'ویدیو' if item.get('video') else ('عکس' if item.get('photo') or item.get('photo_file_id') else 'بدون رسانه')}\n"
        f"زمان برنامه‌ریزی‌شده: {scheduled_text}"
    )
    if item.get("delivery_status") in {"uncertain", "partial"}:
        ids = item.get("_delivery_message_ids", [])
        text += ("\n\n⚠️ نتیجه انتشار نامشخص/ناقص است؛ ربات آن را خودکار دوباره نمی‌فرستد. "
                 f"شناسه‌های دریافت‌شده: {', '.join(str(x) for x in ids) or 'ثبت نشده'}"
                 f"\nآخرین خطا: {item.get('delivery_error', 'ثبت نشده')}")
    rows = [
        [admin_button("🚀 انتشار فوری", f"confirm|qpublish|{item_id}")],
        [admin_button("✏️ ویرایش متن", f"qedit|{item_id}"), admin_button("🖼 تغییر عکس/رسانه", f"qimg|{item_id}")],
        [admin_button("⏰ تأخیر دقیقه‌ای", f"qdelay|{item_id}"), admin_button("🗓 ساعت دقیق", f"qexact|{item_id}")],
        [admin_button("⬆️ بالای صف", f"qbump|{item_id}|top"), admin_button("⬇️ ته صف", f"qbump|{item_id}|bottom")],
        [admin_button("📌 انتشار بعدی", f"qpin|{item_id}"), admin_button("🔄 بازنویسی", f"qrewrite|{item_id}")],
        [admin_button("🔗 مشاهده منابع", f"qsources|{item_id}")],
        [admin_button("🔎 بازبینی ادعا با AI", f"qverify|{item_id}")],
        ([admin_button("✅ این خبر منتشر شده", f"qresolve|{item_id}|sent"), admin_button("⚠️ اجازه تلاش مجدد", f"qresolve|{item_id}|retry")] if item.get("delivery_status") in {"uncertain", "partial"} else []),
        ([admin_button("✅ تأیید برای انتشار", f"qapprove|{item_id}")] if item.get("manual_approval_required") else []),
        [admin_button("🗑 رد و حذف خبر", f"confirm|qdelete|{item_id}")],
        [admin_button("⬅️ بازگشت به صف", "queue")],
    ]
    rows = [row for row in rows if row]
    admin_render(chat_id, text, rows, message_id)


def render_history(state, chat_id, message_id=None, page=0):
    history = [x for x in state.get("_published_history", []) if not x.get("deleted")]
    history = list(reversed(history))
    start = max(0, page) * ADMIN_PAGE_SIZE
    subset = history[start:start + ADMIN_PAGE_SIZE]
    rows = []
    for item in subset:
        title = normalize_space(item.get("title", "بدون تیتر"))[:42]
        rows.append([admin_button(title, f"h|{item.get('history_id')}")])
    nav = []
    if page > 0:
        nav.append(admin_button("◀️ قبلی", f"historypage|{page-1}"))
    if start + ADMIN_PAGE_SIZE < len(history):
        nav.append(admin_button("بعدی ▶️", f"historypage|{page+1}"))
    if nav:
        rows.append(nav)
    rows.append([admin_button("⬅️ بازگشت", "news")])
    text = f"🗃 تاریخچه انتشار — {len(history)} پست ثبت‌شده\nآخرین بررسی پنل: {now_tehran().strftime('%Y-%m-%d %H:%M:%S')}\nصفحه {page+1}"
    if not history:
        text += "\nهنوز تاریخچه‌ای ذخیره نشده است."
    admin_render(chat_id, text, rows, message_id)


def find_history_item(state, history_id):
    for item in state.get("_published_history", []):
        if str(item.get("history_id", "")) == str(history_id):
            return item
    return None


def render_history_item(state, chat_id, history_id, message_id=None):
    item = find_history_item(state, history_id)
    if not item or item.get("deleted"):
        admin_render(chat_id, "این رکورد در تاریخچه فعال نیست.", [[admin_button("⬅️ تاریخچه", "history")]], message_id)
        return
    ts = safe_float(item.get("ts"), 0)
    stamp = datetime.fromtimestamp(ts, TEHRAN_TZ or timezone.utc).strftime('%Y-%m-%d %H:%M') if ts else "نامشخص"
    source_urls = item.get("source_urls", []) if isinstance(item.get("source_urls"), list) else []
    source_text = "\n".join(source_urls[:8]) or "ثبت نشده"
    text = (f"🗃 پست منتشرشده\n{stamp}\n\n{item.get('title','')}\n\n{item.get('body','')}"
            f"\n\nشناسه پیام تلگرام: {item.get('message_id') or 'ثبت نشده'}"
            f"\nشناسه‌های قطعه‌ای: {', '.join(str(x) for x in item.get('message_ids', []) if x) or '—'}"
            f"\nهوش مصنوعی مرحله اصلی: {item.get('ai_provider_stage1','unknown')}"
            f"\nویراستار AI: {item.get('ai_provider_editor','not_used')}"
            f"\nفیلتر اولیه: {item.get('classification_model','disabled')}"
            f"\nوضعیت اعتبار: {item.get('verification','reported')}"
            f"\nمنابع: {', '.join(item.get('source_channels', [])) or 'نامشخص'}"
            f"\nپیوندهای منبع:\n{source_text}")
    role = admin_role(state, chat_id)
    if role == "observer":
        rows = [[admin_button("⬅️ تاریخچه", "history")]]
    else:
        rows = [
            [admin_button("✏️ اصلاح پست کانال", f"hedit|{history_id}")],
        ]
        if item.get("media_type") in {"photo", "video"}:
            rows.append([admin_button("🖼 تغییر عکس پست", f"himg|{history_id}")])
        rows.extend([
            [admin_button("🗑 حذف پست از کانال", f"confirm|hdelete|{history_id}")],
            [admin_button("⬅️ تاریخچه", "history")],
        ])
    admin_render(chat_id, text, rows, message_id)


def render_ai_menu(state, chat_id, message_id=None):
    settings = admin_settings(state)
    order = settings.get("ai_order", list(AI_PROVIDER_ORDER))
    primary = order[0] if order else "نامشخص"
    rows = []
    for provider in AI_PROVIDER_ORDER:
        enabled = settings.get("ai_enabled", {}).get(provider, True)
        rows.append([admin_button(f"{'🟢' if enabled else '⚪'} {AI_PROVIDER_LABELS[provider]}{' ⭐ اصلی' if provider == primary else ''}", f"aiprovider|{provider}")])
    rows.extend([
        [admin_button("🔢 تعداد تلاش مجدد", "setprompt|ai_attempts")],
        [admin_button("🧪 آزمایش ترجمه بدون انتشار", "aitest")],
        [admin_button("⬅️ منوی اصلی", "menu")],
    ])
    admin_render(chat_id, "🤖 مدیریت هوش مصنوعی\n\nترتیب فعلی: " + " → ".join(AI_PROVIDER_LABELS[x] for x in order) + f"\nحالت آزمایشی: {'روشن' if is_test_mode(state) else 'خاموش'}\n\nبرای جزئیات، فعال/غیرفعال‌کردن یا تعیین سرویس اصلی، یک ارائه‌دهنده را انتخاب کن.", rows, message_id)


def render_ai_provider(state, chat_id, provider, message_id=None):
    if provider not in AI_PROVIDER_ORDER:
        render_ai_menu(state, chat_id, message_id)
        return
    ps = provider_state(state, provider)
    usage = ps.setdefault("usage", {})
    error = ps.get("last_error", {})
    enabled = admin_settings(state).get("ai_enabled", {}).get(provider, True)
    keys = AI_PROVIDER_KEYS[provider]
    last_use = safe_float(ps.get("last_request_ts"), 0)
    cooldown = max(0, int(safe_float(ps.get("cooldown_until"), 0) - now_ts()))
    text = (
        f"🤖 {AI_PROVIDER_LABELS[provider]}\n\n"
        f"وضعیت پنل: {'فعال' if enabled else 'غیرفعال'}\n"
        f"مدل: {provider_model_name(provider)}\n"
        f"کلیدهای تعریف‌شده: {len(keys)} (مقدار کلیدها نمایش داده نمی‌شود)\n"
        f"آخرین وضعیت HTTP: {ps.get('last_status', '—')}\n"
        f"Cooldown: {cooldown} ثانیه\n"
        f"آخرین استفاده: {datetime.fromtimestamp(last_use, TEHRAN_TZ or timezone.utc).strftime('%Y-%m-%d %H:%M:%S') if last_use else 'هنوز استفاده نشده'}\n"
        f"درخواست‌های ثبت‌شده: {safe_int(usage.get('requests'), 0)} | موفق: {safe_int(usage.get('successes'), 0)} | خطا: {safe_int(usage.get('errors'), 0)}\n"
        f"توکن‌های ورودی/خروجی در صورت گزارش API: {safe_int(usage.get('prompt_tokens'), 0)} / {safe_int(usage.get('completion_tokens'), 0)}"
    )
    if error:
        text += f"\n\nآخرین خطا: {error.get('category', 'خطا')} | HTTP {error.get('status_code', 0)}\n{sanitize_error_text(error.get('detail', ''))}"
    rows = [
        [admin_button("⏯ فعال/غیرفعال", f"aienable|{provider}")],
        [admin_button("⭐ انتخاب به‌عنوان سرویس اصلی", f"aiprimary|{provider}")],
        [admin_button("⬅️ بازگشت به AI", "ai")],
    ]
    admin_render(chat_id, text, rows, message_id)


def render_sources(state, chat_id, message_id=None):
    configured = [str(x).strip().lstrip("@") for x in admin_settings(state).get("source_channels", SOURCE_CHANNELS) if str(x).strip()]
    disabled = set(state.get("_disabled_sources", []))
    rows = []
    for source in configured:
        rows.append([admin_button(("⚪ فعال‌سازی " if source in disabled else "🟢 غیرفعال‌سازی ") + "@" + source[:32], f"sourcetoggle|{source}")])
        rows.append([admin_button("➖ حذف @" + source[:36], f"confirm|sourcedel|{source}")])
    rows.extend([[admin_button("➕ افزودن کانال تلگرام", "sourceadd")], [admin_button("📈 عملکرد منابع", "sourcestats")], [admin_button("⬅️ منوی اصلی", "menu")]])
    text = "📡 منابع تلگرام\n\n" + ("\n".join(("⚪ " if x in disabled else "🟢 ") + "@" + x for x in configured) if configured else "هیچ منبعی تنظیم نشده است") + "\n\nغیرفعال‌سازی موقت، منبع و تاریخچه آن را حذف نمی‌کند."
    admin_render(chat_id, text, rows, message_id)


def render_admins(state, chat_id, message_id=None):
    if admin_role(state, chat_id) != "owner":
        admin_render(chat_id, "فقط مالک اصلی اجازه مدیریت ادمین‌ها را دارد.", [[admin_button("🏠 منوی اصلی", "menu")]], message_id)
        return
    lines = [f"مالک اصلی: {OWNER_TELEGRAM_ID} (قابل حذف یا تنزل نیست)"]
    rows = []
    for uid, entry in sorted(state.get("_admin_users", {}).items(), key=lambda pair: int(pair[0]) if str(pair[0]).isdigit() else 0):
        role = entry.get("role", "observer") if isinstance(entry, dict) else "observer"
        role_name = "مدیر" if role == "manager" else "ناظر"
        lines.append(f"• {uid}: {role_name}")
        rows.append([admin_button(f"حذف {uid} ({role_name})", f"confirm|admin_del|{uid}")])
    rows.extend([[admin_button("➕ افزودن ادمین", "adminadd")], [admin_button("⬅️ منوی اصلی", "menu")]])
    admin_render(chat_id, "👥 مدیریت ادمین‌ها\n\n" + "\n".join(lines), rows, message_id)


def render_logs(state, chat_id, message_id=None):
    errors = list(reversed(state.get("_system_errors", [])[-12:]))
    activities = list(reversed(state.get("_admin_activity", [])[-12:]))
    text_parts = ["📝 گزارش فعالیت‌ها"]
    text_parts.append("\nخطاهای اخیر:")
    if errors:
        for item in errors:
            when = datetime.fromtimestamp(safe_float(item.get("ts"), 0), TEHRAN_TZ or timezone.utc).strftime('%m-%d %H:%M')
            text_parts.append(f"• {when} | {item.get('area')} | {item.get('context')} | {item.get('category')} | {item.get('detail', '')[:180]}")
    else:
        text_parts.append("• خطایی ثبت نشده است.")
    text_parts.append("\nآخرین اقدامات مدیریتی:")
    if activities:
        for item in activities:
            when = datetime.fromtimestamp(safe_float(item.get("ts"), 0), TEHRAN_TZ or timezone.utc).strftime('%m-%d %H:%M')
            text_parts.append(f"• {when} | {item.get('role')} {item.get('user_id')} | {item.get('action')} | {item.get('detail', '')}")
    else:
        text_parts.append("• هنوز اقدامی ثبت نشده است.")
    rows = [[admin_button("🔄 تازه‌سازی", "logs")], [admin_button("🧾 چرا خبر انتخاب/رد شد؟", "decisions")], [admin_button("⬅️ منوی اصلی", "menu")]]
    admin_render(chat_id, "\n".join(text_parts), rows, message_id)


def render_decisions(state, chat_id, message_id=None):
    decisions = list(reversed(state.get("_recent_decisions", [])[-18:]))
    lines = ["🧾 تصمیم‌های تحریریه — این لاگ از داده‌های پردازش‌شده ربات تشکیل می‌شود:"]
    for item in decisions:
        when = datetime.fromtimestamp(safe_float(item.get("ts"), 0), TEHRAN_TZ or timezone.utc).strftime('%m-%d %H:%M')
        score = item.get("score", -1)
        score_text = f" | امتیاز {score}" if safe_int(score, -1) >= 0 else ""
        lines.append(f"• {when} | {item.get('outcome')}{score_text}\n  {item.get('title', '')}\n  علت: {item.get('reason', '')}")
    if not decisions:
        lines.append("\nهنوز تصمیمی ثبت نشده است؛ با چرخه‌های بعدی این فهرست پر می‌شود.")
    admin_render(chat_id, "\n".join(lines), [[admin_button("⬅️ گزارش‌ها", "logs")], [admin_button("🏠 منوی اصلی", "menu")]], message_id)


def render_tools(state, chat_id, message_id=None):
    rows = [
        [admin_button("🩺 بررسی و عیب‌یابی", "diagnose")],
        [admin_button("🧪 تغییر حالت آزمایشی", "testmode")],
        [admin_button("🚨 توقف اضطراری انتشار", "setting|emergency")],
        [admin_button("🧹 بررسی تکراری‌های صف", "dedupqueue")],
        [admin_button("⬅️ منوی اصلی", "menu")],
    ]
    admin_render(chat_id, f"🛠 ابزارهای پیشرفته\n\nحالت آزمایشی: {'روشن' if is_test_mode(state) else 'خاموش'}\nانتشار خودکار: {'روشن' if admin_settings(state).get('auto_publish_enabled', True) else 'خاموش'}\nتوقف اضطراری: {'فعال' if admin_settings(state).get('emergency_stop') else 'غیرفعال'}", rows, message_id)


def diagnose_system(state):
    results = []
    try:
        path = os.path.abspath(STATE_FILE)
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        probe = os.path.join(parent, ".raptor-write-test")
        with open(probe, "w", encoding="utf-8") as file:
            file.write("ok")
        os.unlink(probe)
        results.append("✅ فایل وضعیت: قابل‌نوشتن")
    except Exception as exc:
        results.append("❌ فایل وضعیت: " + sanitize_error_text(exc))
    try:
        payload = telegram_request("getChat", data={"chat_id": TARGET_CHAT_ID}, timeout=20)
        chat_title = payload.get("result", {}).get("title", TARGET_CHAT_ID)
        results.append(f"✅ Telegram Bot API و دسترسی به کانال: پاسخ گرفت ({chat_title})")
    except Exception as exc:
        results.append("❌ Telegram/کانال: " + sanitize_error_text(exc))
    sources = runtime_source_channels(state)
    if not sources:
        results.append("❌ دریافت منابع: هیچ کانالی تعریف نشده")
    else:
        try:
            test_posts = fetch_channel_posts(sources[0], state=state)
            results.append(f"✅ دریافت منبع @{sources[0]}: {len(test_posts)} پست خوانده شد")
        except Exception as exc:
            results.append(f"❌ دریافت منبع @{sources[0]}: {sanitize_error_text(exc)}")
    for provider in AI_PROVIDER_ORDER:
        ps = provider_state(state, provider)
        if not admin_settings(state).get("ai_enabled", {}).get(provider, True):
            results.append(f"⚪ {AI_PROVIDER_LABELS[provider]}: از پنل غیرفعال است")
        elif not AI_PROVIDER_KEYS[provider]:
            results.append(f"⚪ {AI_PROVIDER_LABELS[provider]}: کلیدی تعریف نشده؛ بررسی نشد")
        elif safe_float(ps.get("cooldown_until"), 0) > now_ts():
            results.append(f"⚠️ {AI_PROVIDER_LABELS[provider]}: در cooldown؛ {int(ps['cooldown_until'] - now_ts())} ثانیه باقی مانده")
        elif ps.get("last_status") in (401, 403, 429) and ps.get("last_error"):
            results.append(f"⚠️ {AI_PROVIDER_LABELS[provider]}: {ps['last_error'].get('category', 'خطا')}")
        else:
            results.append(f"ℹ️ {AI_PROVIDER_LABELS[provider]}: کلید دارد؛ سلامت واقعی در درخواست بعدی مشخص می‌شود")
    return "🩺 نتیجه عیب‌یابی\n\n" + "\n".join(results)


def deduplicate_queue_admin(state):
    queue = state.get("_pending_queue", [])
    keep = []
    removed = 0
    # این بررسی فقط با درخواست مدیر اجرا می‌شود؛ چرخه عادی ربات را سنگین نمی‌کند.
    for item in queue:
        duplicate_index = None
        for idx, old in enumerate(keep):
            if item_fingerprint(item) == item_fingerprint(old):
                duplicate_index = idx
                break
            similarity = combined_similarity(
                item.get("title", ""), item.get("body", ""), old.get("title", ""), old.get("body", "")
            )
            if similarity >= 0.94 and normalize_event_key(item.get("event_key", "")) == normalize_event_key(old.get("event_key", "")):
                duplicate_index = idx
                break
        if duplicate_index is None:
            keep.append(item)
        else:
            removed += 1
            old = keep[duplicate_index]
            if safe_int(item.get("news_score"), 0) > safe_int(old.get("news_score"), 0):
                keep[duplicate_index] = item
    state["_pending_queue"] = keep
    save_state(state)
    return removed


def edit_published_message(item):
    message_id = safe_int(item.get("message_id"), 0)
    if not message_id:
        raise RuntimeError("شناسه پیام تلگرام ذخیره نشده است؛ این رکورد از نسخه جدید یا پیام قابل‌ویرایش نیست.")
    if item.get("media_type") in {"photo", "video"}:
        text = build_media_caption(item.get("title", ""), item.get("body", ""), 1024)
        method = "editMessageCaption"
        data = {"chat_id": item.get("chat_id", TARGET_CHAT_ID), "message_id": message_id, "caption": text,
                "caption_entities": json.dumps(caption_bold_entities(text))}
    else:
        message_text = build_text_only_message(item)
        title_len = _utf16_len(format_public_title(item))
        data = {"chat_id": item.get("chat_id", TARGET_CHAT_ID), "message_id": message_id,
                "text": message_text,
                "entities": json.dumps([{"type": "bold", "offset": 0, "length": title_len}]) if title_len else "[]",
                "disable_web_page_preview": "false"}
        method = "editMessageText"
    return telegram_request(method, data=data, timeout=25)



def edit_published_media(item):
    message_id = safe_int(item.get("message_id"), 0)
    if not message_id:
        raise RuntimeError("شناسه پیام تلگرام ثبت نشده است.")
    media = item.get("photo_file_id") or item.get("photo")
    media_type = "photo" if media else ""
    if not media:
        raise RuntimeError("برای این رکورد URL یا file_id عکس ذخیره نشده است؛ می‌توانی پست را حذف کنی و خبر را دوباره منتشر کنی.")
    caption = build_media_caption(item.get("title", ""), item.get("body", ""), 1024)
    input_media = {"type": media_type, "media": media, "caption": caption,
                   "caption_entities": caption_bold_entities(caption)}
    return telegram_request("editMessageMedia", data={
        "chat_id": item.get("chat_id", TARGET_CHAT_ID),
        "message_id": message_id,
        "media": json.dumps(input_media, ensure_ascii=False),
    }, timeout=30)


def delete_published_message(item):
    message_id = safe_int(item.get("message_id"), 0)
    if not message_id:
        raise RuntimeError("شناسه پیام تلگرام ثبت نشده است؛ حذف از کانال ممکن نیست.")
    return telegram_request("deleteMessage", data={"chat_id": item.get("chat_id", TARGET_CHAT_ID), "message_id": message_id}, timeout=20)


def rewrite_queue_item_with_style(state, user_id, chat_id, item_id, style_code, message_id=None):
    item = find_queue_item(state, item_id)
    if not item:
        admin_render(chat_id, "خبر در صف پیدا نشد.", [[admin_button("صف", "queue")]], message_id)
        return
    styles = {
        "short": "خبر را کوتاه‌تر کن؛ نام‌ها، اعداد، تاریخ‌ها و ادعاهای اصلی حفظ شوند.",
        "formal": "خبر را رسمی‌تر، روشن‌تر و دقیق‌تر بنویس؛ بدون افزودن واقعیت.",
        "context": "فقط زمینه‌ای را روشن‌تر کن که از متن موجود پشتیبانی می‌شود؛ هیچ واقعیت جدیدی نساز.",
        "headline": "فقط تیتر را بهتر، روشن‌تر و دقیق‌تر کن؛ بدنه باید بدون تغییر معنایی باقی بماند.",
    }
    style_prompt = styles.get(style_code)
    if not style_prompt:
        admin_render(chat_id, "سبک بازنویسی ناشناخته است.", [[admin_button("بازگشت به خبر", f"q|{item_id}")]], message_id)
        return
    try:
        prompt = ("فقط JSON معتبر با title و body برگردان. خبر فارسی را دقیق و بدون جعل بازنویسی کن. "
                  + style_prompt + "\nتیتر: " + item.get("title", "") + "\nمتن: " + item.get("body", ""))
        parsed, provider = ai_call_json(state, prompt, "admin_rewrite")
        title = normalize_persian_text(str(parsed.get("title", "") if isinstance(parsed, dict) else ""))
        body = normalize_persian_text(str(parsed.get("body", "") if isinstance(parsed, dict) else ""))
        if not title or not body:
            raise ValueError("خروجی بازنویسی قابل استفاده نبود؛ متن تغییری نکرد.")
        item.setdefault("versions", []).append({"ts": now_ts(), "title": item.get("title", ""), "body": item.get("body", ""), "provider": provider or "unknown"})
        item["versions"] = item["versions"][-20:]
        item["title"], item["body"] = title[:240], body[:6500]
        item["ai_provider_editor"] = provider or "unknown"
        item["ai_provider"] = f"{item.get('ai_provider_stage1', 'unknown')} + admin_rewrite:{provider or 'unknown'}"
        item["fingerprint"] = item_fingerprint(item)
        save_state(state)
        record_admin_activity(state, user_id, "بازنویسی سبک AI خبر", f"{item.get('title','')} / {style_code} / {provider}")
        render_queue_item(state, chat_id, item_id, message_id)
    except Exception as exc:
        admin_render(chat_id, "بازنویسی انجام نشد: " + sanitize_error_text(exc), [[admin_button("بازگشت به خبر", f"q|{item_id}")]], message_id)


def review_queue_item_with_ai(state, user_id, chat_id, item_id, message_id=None):
    """AI-only consistency review; never misrepresents URL strings as independently verified sources."""
    item = find_queue_item(state, item_id)
    if not item:
        admin_render(chat_id, "خبر در صف پیدا نشد.", [[admin_button("صف انتشار", "queue")]], message_id)
        return
    source_urls = item.get("source_urls", []) if isinstance(item.get("source_urls"), list) else []
    prompt = (
        "نقش: ویراستار کنترل کیفیت خبر فارسی. فقط مواد داده‌شده را بررسی کن؛ جست‌وجوی وب در اختیار تو نیست "
        "و صرف دیدن URL به معنی خواندن منبع نیست. هیچ ادعایی را به‌عنوان حقیقت مستقل تأییدشده اعلام نکن. "
        "سازگاری داخلی، عبارت‌های قطعی بدون شواهد، عدد/تاریخ نیازمند منبع، و محدودیت‌های متن را مشخص کن. "
        "JSON معتبر با فیلدهای status (یکی از reported, needs_independent_source, internal_inconsistency, supplied_text_supports), "
        "explanation و claims_to_check (آرایه کوتاه) بده. اگر شواهد کافی نداری، reported یا needs_independent_source را انتخاب کن.\n"
        + json.dumps({"title": item.get("title", ""), "body": item.get("body", ""),
                      "verification_current": item.get("verification", "reported"),
                      "source_channels": item.get("source_channels", []), "source_urls": source_urls}, ensure_ascii=False)
    )
    try:
        parsed, provider = ai_call_json(state, prompt, "admin_claim_review")
        if not isinstance(parsed, dict):
            raise ValueError("خروجی بررسی باید JSON باشد")
        allowed = {"reported", "needs_independent_source", "internal_inconsistency", "supplied_text_supports"}
        status = str(parsed.get("status", "reported"))
        if status not in allowed:
            status = "reported"
        claims = parsed.get("claims_to_check", [])
        if not isinstance(claims, list):
            claims = []
        review = {"ts": now_ts(), "provider": provider or "unknown", "status": status,
                  "explanation": normalize_persian_text(str(parsed.get("explanation", "")))[:1200],
                  "claims_to_check": [normalize_persian_text(str(x))[:240] for x in claims[:8]]}
        item["verification_review"] = review
        save_state(state)
        record_admin_activity(state, user_id, "بازبینی AI ادعاهای خبر", f"{item.get('title','')} / {status} / {provider}")
        labels = {"reported": "گزارش‌شده؛ تأیید مستقل انجام نشد", "needs_independent_source": "نیازمند منبع مستقل",
                  "internal_inconsistency": "احتمال ناسازگاری داخلی", "supplied_text_supports": "متن فعلی از ادعا پشتیبانی می‌کند؛ نه تأیید بیرونی"}
        lines = ["🔎 بازبینی ادعا با هوش مصنوعی", "", f"وضعیت: {labels[status]}", f"مدل: {provider or 'نامشخص'}",
                 f"زمان: {now_tehran().strftime('%Y-%m-%d %H:%M')}", "", review["explanation"] or "توضیحی برنگشت.", "", "ادعاهای نیازمند بررسی:"]
        lines.extend(["• " + x for x in review["claims_to_check"]] or ["• مورد مشخصی گزارش نشد."])
        lines.extend(["", "محدودیت: این بررسی با متن/URLهای ثبت‌شده انجام شد و به‌خودی‌خود راستی‌آزمایی مستقل وب نیست."])
        admin_render(chat_id, "\n".join(lines), [[admin_button("بازگشت به خبر", f"q|{item_id}")], [admin_button("مشاهده منابع", f"qsources|{item_id}")]], message_id)
    except Exception as exc:
        admin_render(chat_id, "بررسی AI انجام نشد: " + sanitize_error_text(exc), [[admin_button("بازگشت به خبر", f"q|{item_id}")]], message_id)


def publish_queue_item_manually(state, item_id):
    if admin_settings(state).get("emergency_stop", False):
        raise RuntimeError("توقف اضطراری فعال است؛ ابتدا آن را از تنظیمات غیرفعال کن.")
    item = find_queue_item(state, item_id)
    if not item:
        raise RuntimeError("خبر موردنظر دیگر در صف نیست.")
    if item.get("delivery_status") in {"sending", "uncertain", "partial", "sent"}:
        raise RuntimeError("وضعیت ارسال این خبر نامشخص/ناقص است؛ ابتدا در صفحه خبر از گزینه حل وضعیت استفاده کن تا احتمال انتشار تکراری کاهش یابد.")
    if is_test_mode(state):
        dispatch_item(item, state=state)
        return False, "حالت آزمایشی روشن است؛ هیچ پستی در کانال منتشر نشد و خبر در صف ماند."
    # انتشار فوری مدیر یک اقدام صریح است؛ نتیجه هر تلاش پیش از تماس با Telegram ذخیره می‌شود.
    item["delivery_status"] = "sending"
    item["delivery_started_ts"] = now_ts()
    item["delivery_attempts"] = safe_int(item.get("delivery_attempts", 0), 0) + 1
    item["manual_approval_required"] = False
    save_state(state)
    try:
        result = dispatch_item(item, state=state)
    except Exception as exc:
        item["delivery_status"] = "uncertain" if not getattr(exc, "partial_message_ids", []) else "partial"
        item["_delivery_message_ids"] = getattr(exc, "partial_message_ids", [])
        item["delivery_error"] = sanitize_error_text(exc)[:240]
        item["delivery_uncertain_ts"] = now_ts()
        save_state(state)
        raise
    if isinstance(result, dict) and result.get("test_mode"):
        item["delivery_status"] = "queued"
        save_state(state)
        return False, "حالت آزمایشی روشن است؛ خبری منتشر نشد."
    queue = state.get("_pending_queue", [])
    state["_pending_queue"] = [x for x in queue if x.get("admin_id") != item_id]
    if str(state.get("_priority_pin_id", "")) == str(item_id):
        state["_priority_pin_id"] = ""
    mark_posted_region(state, item)
    remember_item(state, item)
    record_published_item(state, item, result, manual=True)
    state["_last_queue_release_ts"] = now_ts()
    save_state(state)
    return True, "خبر با موفقیت منتشر شد."

def set_admin_session(state, user_id, session):
    state.setdefault("_admin_sessions", {})[str(user_id)] = session
    save_state(state)


def clear_admin_session(state, user_id):
    state.setdefault("_admin_sessions", {}).pop(str(user_id), None)
    save_state(state)


def ask_admin_input(state, user_id, chat_id, session, prompt, callback_message_id=None):
    set_admin_session(state, user_id, session)
    admin_render(chat_id, prompt + "\n\nبرای لغو /cancel بفرست.", [[admin_button("لغو", "menu")]], callback_message_id)


def handle_admin_text(state, user_id, chat_id, text):
    text = str(text or "").strip()
    if text == "/cancel":
        clear_admin_session(state, user_id)
        admin_show_menu(chat_id, role=admin_role(state, user_id))
        return
    session = state.get("_admin_sessions", {}).get(str(user_id))
    if not session:
        if text.startswith("/admin"):
            admin_show_menu(chat_id, role=admin_role(state, user_id))
        elif text.startswith("/find"):
            bits = text.split(maxsplit=1)
            if len(bits) != 2 or not bits[1].strip().isdigit():
                admin_render(chat_id, "قالب دستور: /find 12345", [[admin_button("V5.3", "v53")]])
            else:
                render_find_message(state, chat_id, bits[1].strip())
        elif text.startswith("/exportcsv"):
            send_history_csv(state, chat_id)
        return
    mode = session.get("type")
    try:
        if mode == "edit_text":
            item = find_queue_item(state, session.get("item_id"))
            if not item:
                raise RuntimeError("خبر از صف حذف شده است.")
            parts = text.split("\n", 1)
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                raise ValueError("متن را در دو بخش بفرست: خط اول تیتر، خط‌های بعدی بدنه خبر.")
            item["title"] = normalize_persian_text(parts[0].strip())[:240]
            item["body"] = normalize_persian_text(parts[1].strip())[:6500]
            item["fingerprint"] = item_fingerprint(item)
            save_state(state)
            record_admin_activity(state, user_id, "ویرایش خبر صف", item["title"])
            clear_admin_session(state, user_id)
            admin_render(chat_id, "✅ متن خبر اصلاح شد.", [[admin_button("پیش‌نمایش خبر", f"q|{item.get('admin_id')}")], [admin_button("صف انتشار", "queue")]])
        elif mode == "edit_history_image":
            item = find_history_item(state, session.get("history_id"))
            if not item or item.get("deleted"):
                raise RuntimeError("پست از تاریخچه فعال خارج شده است.")
            if not re.match(r"^https?://", text, flags=re.I):
                raise ValueError("برای تغییر عکس پست منتشرشده، URL معتبر عکس با http/https بفرست یا خود عکس را به‌صورت فایل در تلگرام ارسال کن.")
            old = (item.get("photo"), item.get("photo_file_id"), item.get("media_type"))
            item["photo"] = text
            item["photo_file_id"] = None
            item["media_type"] = "photo"
            try:
                edit_published_media(item)
            except Exception:
                item["photo"], item["photo_file_id"], item["media_type"] = old
                raise
            save_state(state)
            record_admin_activity(state, user_id, "تغییر عکس پست منتشرشده", item.get("title", ""))
            clear_admin_session(state, user_id)
            admin_render(chat_id, "✅ عکس پست در کانال تغییر کرد.", [[admin_button("تاریخچه", "history")], [admin_button("منوی اصلی", "menu")]])
        elif mode == "edit_history_text":
            item = find_history_item(state, session.get("history_id"))
            if not item or item.get("deleted"):
                raise RuntimeError("پست از تاریخچه فعال خارج شده است.")
            parts = text.split("\n", 1)
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                raise ValueError("خط اول تیتر و خط‌های بعدی بدنه خبر باشد.")
            old_title, old_body = item.get("title", ""), item.get("body", "")
            item["title"] = normalize_persian_text(parts[0].strip())[:240]
            item["body"] = normalize_persian_text(parts[1].strip())[:6500]
            try:
                edit_published_message(item)
            except Exception:
                item["title"], item["body"] = old_title, old_body
                raise
            save_state(state)
            record_admin_activity(state, user_id, "اصلاح پست منتشرشده", item["title"])
            clear_admin_session(state, user_id)
            admin_render(chat_id, "✅ پست منتشرشده در کانال اصلاح شد.", [[admin_button("مشاهده پست", f"h|{item.get('history_id')}")], [admin_button("تاریخچه", "history")]])
        elif mode == "edit_image":
            item = find_queue_item(state, session.get("item_id"))
            if not item:
                raise RuntimeError("خبر از صف حذف شده است.")
            if text in {"-", "حذف", "پاک کردن"}:
                item["photo"] = None
                item["photo_file_id"] = None
                item["video"] = None
            elif re.match(r"^https?://", text, flags=re.I):
                item["photo"] = text
                item["photo_file_id"] = None
                item["video"] = None
            else:
                raise ValueError("یک URL معتبر با http/https بفرست یا برای حذف رسانه، - ارسال کن.")
            save_state(state)
            record_admin_activity(state, user_id, "تغییر رسانه خبر", item.get("title", ""))
            clear_admin_session(state, user_id)
            admin_render(chat_id, "✅ رسانه خبر به‌روزرسانی شد.", [[admin_button("پیش‌نمایش خبر", f"q|{item.get('admin_id')}")], [admin_button("صف انتشار", "queue")]])
        elif mode == "schedule_exact":
            item = find_queue_item(state, session.get("item_id"))
            if not item:
                raise RuntimeError("خبر از صف حذف شده است.")
            if item.get("delivery_status") in {"sending", "uncertain", "partial", "sent"}:
                raise RuntimeError("ابتدا وضعیت انتشار نامشخص/ناقص را از صفحه خبر حل کن؛ زمان‌بندی به‌تنهایی نباید ارسال را دوباره فعال کند.")
            try:
                dt = datetime.strptime(text, "%Y-%m-%d %H:%M")
                dt = dt.replace(tzinfo=TEHRAN_TZ or timezone.utc)
            except ValueError:
                raise ValueError("قالب معتبر نیست. نمونه: 2026-10-20 21:30")
            if dt.timestamp() <= now_ts():
                raise ValueError("زمان انتشار باید در آینده باشد.")
            item["scheduled_at"] = dt.timestamp()
            item["delivery_status"] = "queued"
            save_state(state); record_admin_activity(state, user_id, "زمان‌بندی دقیق", f"{item.get('title','')} — {text}")
            clear_admin_session(state, user_id)
            admin_render(chat_id, f"✅ زمان دقیق ذخیره شد: {dt.strftime('%Y-%m-%d %H:%M')}", [[admin_button("پیش‌نمایش خبر", f"q|{item.get('admin_id')}")], [admin_button("صف", "queue")]])
        elif mode == "block_term":
            term = normalize_space(text).lower()
            if len(term) < 2: raise ValueError("واژه باید حداقل دو نویسه داشته باشد.")
            terms = state.setdefault("_blocklist_terms", [])
            if term not in terms: terms.append(term)
            state["_blocklist_terms"] = terms[-300:]
            save_state(state); clear_admin_session(state, user_id); render_terms(state, chat_id)
        elif mode == "priority_term":
            term = normalize_space(text).lower()
            if len(term) < 2: raise ValueError("واژه باید حداقل دو نویسه داشته باشد.")
            terms = state.setdefault("_priority_keywords", [])
            if term not in terms: terms.append(term)
            state["_priority_keywords"] = terms[-300:]
            save_state(state); clear_admin_session(state, user_id); render_terms(state, chat_id)
        elif mode == "topic_silence":
            parts_in = [x.strip() for x in text.rsplit(",", 1)]
            if len(parts_in) != 2: raise ValueError("قالب: عبارت,ساعت")
            term, hours_text = parts_in
            hours = safe_int(hours_text, 0)
            if not term or not 1 <= hours <= 720: raise ValueError("مدت باید بین ۱ تا ۷۲۰ ساعت باشد.")
            silences = state.setdefault("_topic_silences", [])
            silences.append({"term": term, "created_ts": now_ts(), "until_ts": now_ts() + hours * 3600})
            state["_topic_silences"] = silences[-200:]
            save_state(state); clear_admin_session(state, user_id); render_terms(state, chat_id)
        elif mode == "auto_rule_add":
            groups = []
            for group_text in re.split(r"\s+OR\s+", text, flags=re.I):
                group = {"min_score": None, "source": "", "bucket": "", "region": ""}
                for condition in [x.strip() for x in group_text.split(";") if x.strip()]:
                    low = condition.lower()
                    if low.startswith("score>="):
                        score = safe_int(condition.split(">=", 1)[1].strip(), -1)
                        if not 0 <= score <= 100: raise ValueError("امتیاز باید بین ۰ تا ۱۰۰ باشد.")
                        group["min_score"] = score
                    elif low.startswith("source="):
                        group["source"] = condition.split("=", 1)[1].strip().lstrip("@")
                    elif low.startswith("bucket="):
                        group["bucket"] = condition.split("=", 1)[1].strip()
                    elif low.startswith("region="):
                        group["region"] = condition.split("=", 1)[1].strip()
                    else: raise ValueError(f"شرط نامعتبر: {condition}")
                if group["min_score"] is None: raise ValueError("در هر گروه OR شرط score>=... الزامی است.")
                groups.append(group)
            rule = {"id": sha1_text(f"{now_ts()}|{text}")[:8], "enabled": True, "created_ts": now_ts(), "groups": groups}
            if len(groups) == 1: rule.update(groups[0])
            rules = state.setdefault("_auto_approve_rules", []); rules.append(rule); state["_auto_approve_rules"] = rules[-100:]
            save_state(state); clear_admin_session(state, user_id); record_admin_activity(state, user_id, "افزودن قانون تأیید خودکار", text)
            render_auto_rules(state, chat_id)
        elif mode == "history_search":
            filters = {}
            free_terms = []
            for token in text.split():
                if "=" in token and token.split("=", 1)[0].lower() in {"source", "bucket", "date", "region", "ai", "id"}:
                    key, value = token.split("=", 1); filters[key.lower()] = normalize_for_match(value)
                else:
                    free_terms.append(token)
            query = normalize_for_match(" ".join(free_terms))
            matches = []
            for rec in reversed(state.get("_published_history", [])):
                stamp = datetime.fromtimestamp(safe_float(rec.get("ts"), 0), TEHRAN_TZ or timezone.utc).strftime("%Y-%m-%d") if safe_float(rec.get("ts"), 0) else ""
                if filters.get("source") and filters["source"] not in normalize_for_match(" ".join(rec.get("source_channels", []))): continue
                if filters.get("bucket") and filters["bucket"] not in normalize_for_match(rec.get("content_bucket", "")): continue
                if filters.get("date") and filters["date"] not in normalize_for_match(stamp): continue
                if filters.get("region") and filters["region"] not in normalize_for_match(rec.get("region", "")): continue
                if filters.get("ai") and filters["ai"] not in normalize_for_match(rec.get("ai_provider", "")): continue
                if filters.get("id") and filters["id"] not in normalize_for_match(" ".join([str(rec.get("message_id", "")), " ".join(str(x) for x in rec.get("message_ids", []))])): continue
                haystack = normalize_for_match(" ".join([str(rec.get("title", "")), str(rec.get("body", "")), str(rec.get("ai_provider", "")), " ".join(rec.get("source_channels", [])), str(rec.get("content_bucket", "")), str(rec.get("message_id", ""))]))
                if not query or query in haystack or filters:
                    matches.append(rec)
            clear_admin_session(state, user_id)
            render_history_search_results(state, chat_id, text, matches[:20])
        elif mode == "rewrite_queue":
            item = find_queue_item(state, session.get("item_id"))
            if not item: raise RuntimeError("خبر از صف حذف شده است.")
            style_prompt = {
                "کوتاه‌تر": "خبر را کوتاه‌تر کن و اطلاعات قطعی را حفظ کن.",
                "رسمی‌تر": "خبر را رسمی‌تر و دقیق‌تر بازنویسی کن.",
                "با زمینه بیشتر": "فقط زمینه‌ای را اضافه کن که از متن موجود پشتیبانی می‌شود؛ واقعیت جدید نساز.",
                "تیتر بهتر": "فقط تیتر را به شکل روشن‌تر و دقیق‌تر اصلاح کن؛ بدنه را دست‌نخورده نگه دار.",
            }.get(text.strip(), text.strip())
            if len(style_prompt) < 3: raise ValueError("یک سبک معتبر وارد کن.")
            parsed, provider = ai_call_json(state, "فقط یک JSON با title و body برگردان. نام‌ها، عددها و ادعاها را تغییر نده. دستور: " + style_prompt + "\nتیتر: " + item.get("title", "") + "\nمتن: " + item.get("body", ""), "admin_rewrite")
            new_title = normalize_persian_text(str(parsed.get("title", "") if isinstance(parsed, dict) else ""))
            new_body = normalize_persian_text(str(parsed.get("body", "") if isinstance(parsed, dict) else ""))
            if not new_title or not new_body: raise ValueError("خروجی بازنویسی قابل استفاده نبود؛ متن تغییری نکرد.")
            item.setdefault("versions", []).append({"ts": now_ts(), "title": item.get("title", ""), "body": item.get("body", ""), "provider": provider})
            item["versions"] = item["versions"][-20:]
            item["title"], item["body"] = new_title[:240], new_body[:6500]
            item["ai_provider_editor"] = provider or "unknown"
            item["ai_provider"] = f"{item.get('ai_provider_stage1', 'unknown')} + admin_rewrite:{provider or 'unknown'}"
            item["fingerprint"] = item_fingerprint(item)
            save_state(state); clear_admin_session(state, user_id); record_admin_activity(state, user_id, "بازنویسی AI خبر", f"{item.get('title','')} / {provider}")
            admin_render(chat_id, f"✅ خبر بازنویسی شد با {provider or 'AI نامشخص'}.", [[admin_button("پیش‌نمایش", f"q|{item.get('admin_id')}")], [admin_button("صف", "queue")]])
        elif mode == "schedule_delay":
            delay = safe_int(text, -1)
            if delay < 0 or delay > 10080:
                raise ValueError("تأخیر باید عددی بین ۰ تا ۱۰۰۸۰ دقیقه (۷ روز) باشد.")
            item = find_queue_item(state, session.get("item_id"))
            if not item:
                raise RuntimeError("خبر از صف حذف شده است.")
            item["scheduled_at"] = now_ts() + delay * 60 if delay else 0
            save_state(state)
            record_admin_activity(state, user_id, "تغییر زمان خبر", f"{item.get('title','')}؛ تأخیر {delay} دقیقه")
            clear_admin_session(state, user_id)
            admin_render(chat_id, "✅ زمان انتشار ذخیره شد.", [[admin_button("پیش‌نمایش خبر", f"q|{item.get('admin_id')}")], [admin_button("صف انتشار", "queue")]])
        elif mode == "setting_value":
            key = session.get("key")
            settings = admin_settings(state)
            if key in {"morning_spacing", "afternoon_spacing"}:
                value = safe_int(text, -1)
                if not 5 <= value <= 1440:
                    raise ValueError("فاصله باید بین ۵ تا ۱۴۴۰ دقیقه باشد.")
                settings["morning_spacing_minutes" if key == "morning_spacing" else "afternoon_spacing_minutes"] = value
            elif key == "active_hours":
                values = [safe_int(x.strip(), -1) for x in text.split(",")]
                if len(values) != 3:
                    raise ValueError("سه عدد با ویرگول بفرست؛ نمونه: 8,14,24")
                start, afternoon, end = values
                if not (0 <= start <= 23 and 0 <= afternoon <= 23 and 1 <= end <= 24 and start != end and start < afternoon < end):
                    raise ValueError("مقادیر نامعتبرند. برای نمونه: 8,14,24 یا 6,12,22")
                settings["active_start_hour"], settings["afternoon_start_hour"], settings["active_end_hour"] = start, afternoon, end
            elif key == "daily_limit":
                value = safe_int(text, -1)
                if not 0 <= value <= 500:
                    raise ValueError("سقف روزانه باید بین ۰ تا ۵۰۰ باشد؛ صفر یعنی بدون سقف.")
                settings["daily_post_limit"] = value
            elif key == "ai_attempts":
                value = safe_int(text, -1)
                if value not in (1, 2):
                    raise ValueError("تعداد تلاش فقط ۱ یا ۲ می‌تواند باشد.")
                settings["ai_attempts"] = value
            else:
                raise ValueError("نوع تنظیم ناشناخته است.")
            save_state(state)
            record_admin_activity(state, user_id, "تغییر تنظیمات", f"{key}={text}")
            clear_admin_session(state, user_id)
            admin_render(chat_id, "✅ تنظیم ذخیره شد.", [[admin_button("تنظیمات ربات", "settings")], [admin_button("هوش مصنوعی", "ai")]])
        elif mode == "add_admin_id":
            new_id = safe_int(text, 0)
            if new_id <= 0:
                raise ValueError("شناسه باید یک عدد صحیح مثبت باشد.")
            if new_id == OWNER_TELEGRAM_ID:
                raise ValueError("مالک اصلی از قبل دسترسی کامل دارد.")
            set_admin_session(state, user_id, {"type": "add_admin_role", "pending_id": new_id})
            admin_render(chat_id, f"برای شناسه {new_id} سطح دسترسی را انتخاب کن:", [[admin_button("مدیر", f"adminrole|{new_id}|manager")], [admin_button("ناظر", f"adminrole|{new_id}|observer")], [admin_button("لغو", "admins")]])
        elif mode == "source_add":
            source = text.strip().lstrip("@").replace("https://t.me/", "").strip("/")
            if not re.fullmatch(r"[A-Za-z0-9_]{5,32}", source):
                raise ValueError("نام کاربری کانال نامعتبر است؛ فقط حروف انگلیسی، عدد و زیرخط.")
            sources = runtime_source_channels(state)
            if source in sources:
                raise ValueError("این منبع از قبل ثبت شده است.")
            sources.append(source)
            admin_settings(state)["source_channels"] = sources
            save_state(state)
            record_admin_activity(state, user_id, "افزودن منبع", source)
            clear_admin_session(state, user_id)
            render_sources(state, chat_id)
        elif mode == "ai_test":
            if len(text) < 15:
                raise ValueError("متن خبر برای آزمایش کوتاه است؛ متن کامل‌تر بفرست.")
            clear_admin_session(state, user_id)
            admin_render(chat_id, "🧪 در حال آزمایش AI… این کار خبر را در صف نمی‌گذارد و در کانال منتشر نمی‌کند.")
            test_post = {
                "source_name": "پنل مدیریت", "published_at": now_tehran().isoformat(),
                "text": text[:AI_INPUT_MAX_CHARS_PER_POST], "title": text[:180],
                "uid": "admin-test-" + sha1_text(text)[:12], "photo": None, "video": None,
                "source_url": "", "_source_key": "admin-test",
            }
            drafts, provider = analyze_and_rewrite_safe(state, [test_post], "پنل مدیریت")
            viable = [x for x in drafts if x.get("relevant")]
            if viable and EDITOR_ENABLED and any(safe_bool(x.get("needs_editor")) for x in viable):
                viable, _ = final_editor(state, viable, [test_post], provider)
            if viable:
                result_lines = [f"🧪 نتیجه آزمایش با {AI_PROVIDER_LABELS.get(provider, provider or 'AI')}:"]
                for draft in viable[:3]:
                    result_lines.append(f"\n{draft.get('title','')}\n\n{draft.get('body','')}\n\nامتیاز: {draft.get('news_score', 0)} | وضعیت: {draft.get('verification','reported')}")
                admin_render(chat_id, "\n".join(result_lines), [[admin_button("آزمایش دوباره", "aitest")], [admin_button("منوی AI", "ai")]])
            else:
                admin_render(chat_id, "AI از این ورودی پیش‌نویس قابل انتشار تولید نکرد یا همه موارد زیر حداقل امتیاز بودند.", [[admin_button("آزمایش دوباره", "aitest")], [admin_button("منوی AI", "ai")]])
            record_admin_activity(state, user_id, "آزمایش هوش مصنوعی", f"provider={provider}")
        else:
            clear_admin_session(state, user_id)
            admin_render(chat_id, "جلسه ورودی ناشناخته بود و لغو شد.", [[admin_button("منوی اصلی", "menu")]])
    except Exception as exc:
        admin_render(chat_id, "⚠️ " + sanitize_error_text(exc), [[admin_button("بازگشت به منوی اصلی", "menu")]])


def admin_confirm_action(state, user_id, chat_id, action, object_id, message_id=None):
    role = admin_role(state, user_id)
    if action in {"admin_del", "admin_add"} and role != "owner":
        admin_render(chat_id, "فقط مالک اصلی اجازه این کار را دارد.", [[admin_button("منوی اصلی", "menu")]], message_id)
        return
    if action == "resetsettings":
        admin_settings(state).update({
            "auto_publish_enabled": True, "emergency_stop": False,
            "morning_spacing_minutes": MORNING_POST_SPACING_MINUTES,
            "afternoon_spacing_minutes": AFTERNOON_POST_SPACING_MINUTES,
            "active_start_hour": 8, "afternoon_start_hour": 14, "active_end_hour": 24,
            "daily_post_limit": 0,
        })
        save_state(state)
        record_admin_activity(state, user_id, "بازنشانی تنظیمات", "مقادیر پیش‌فرض بازگردانده شدند")
        render_settings(state, chat_id, message_id)
    elif action == "emergency":
        st = admin_settings(state)
        st["emergency_stop"] = not st.get("emergency_stop", False)
        save_state(state)
        record_admin_activity(state, user_id, "توقف اضطراری", str(st["emergency_stop"]))
        render_settings(state, chat_id, message_id)
    elif action == "qpublish":
        try:
            success, result = publish_queue_item_manually(state, object_id)
            record_admin_activity(state, user_id, "انتشار فوری", f"{object_id}: {result}")
            admin_render(chat_id, ("✅ " if success else "ℹ️ ") + result, [[admin_button("پیش‌نمایش خبر", f"q|{object_id}")], [admin_button("صف انتشار", "queue")], [admin_button("منوی اصلی", "menu")]], message_id)
        except Exception as exc:
            record_system_error(state, "Telegram", "انتشار فوری از پنل", exc)
            admin_render(chat_id, "❌ انتشار انجام نشد: " + sanitize_error_text(exc), [[admin_button("بازگشت به خبر", f"q|{object_id}")], [admin_button("صف", "queue")]], message_id)
    elif action == "qdelete":
        item = find_queue_item(state, object_id)
        if item:
            state["_pending_queue"] = [x for x in state.get("_pending_queue", []) if x.get("admin_id") != object_id]
            record_decision(state, item.get("title"), "رد‌شده از پنل", "حذف دستی توسط مدیر", item.get("news_score"))
            record_admin_activity(state, user_id, "حذف خبر از صف", item.get("title", ""))
            save_state(state)
        render_queue(state, chat_id, message_id)
    elif action == "hdelete":
        item = find_history_item(state, object_id)
        if item:
            try:
                delete_published_message(item)
                item["deleted"] = True
                item["deleted_ts"] = now_ts()
                record_admin_activity(state, user_id, "حذف پست کانال", item.get("title", ""))
                save_state(state)
                admin_render(chat_id, "✅ پست از کانال حذف شد.", [[admin_button("تاریخچه", "history")], [admin_button("منوی اصلی", "menu")]], message_id)
            except Exception as exc:
                record_system_error(state, "Telegram", "حذف پست منتشرشده", exc)
                admin_render(chat_id, "❌ حذف پست ممکن نشد: " + sanitize_error_text(exc), [[admin_button("تاریخچه", "history")]], message_id)
        else:
            render_history(state, chat_id, message_id)
    elif action == "admin_del":
        if str(object_id) == str(OWNER_TELEGRAM_ID):
            admin_render(chat_id, "مالک اصلی را نمی‌توان حذف یا تنزل داد.", [[admin_button("مدیریت ادمین‌ها", "admins")]], message_id)
            return
        previous = state.get("_admin_users", {}).pop(str(object_id), None)
        if previous:
            record_admin_activity(state, user_id, "حذف ادمین", str(object_id))
        save_state(state)
        render_admins(state, chat_id, message_id)
    elif action == "sourcedel":
        sources = runtime_source_channels(state)
        source = str(object_id)
        if source in sources and len(sources) > 1:
            sources.remove(source)
            admin_settings(state)["source_channels"] = sources
            record_admin_activity(state, user_id, "حذف منبع", source)
            save_state(state)
            render_sources(state, chat_id, message_id)
        else:
            admin_render(chat_id, "منبع حذف نشد؛ دست‌کم یک منبع باید باقی بماند.", [[admin_button("منابع", "sources")]], message_id)


def _resolve_uncertain_as_sent(state, user_id, chat_id, item, item_id, ids, message_id=None):
    synthetic_result = {"result": {"message_id": (ids[0] if ids else safe_int(item.get("message_id"), 0))}, "_all_message_ids": ids}
    queue = state.get("_pending_queue", [])
    state["_pending_queue"] = [x for x in queue if str(x.get("admin_id", "")) != str(item_id)]
    if str(state.get("_priority_pin_id", "")) == str(item_id): state["_priority_pin_id"] = ""
    mark_posted_region(state, item)
    remember_item(state, item)
    record_published_item(state, item, synthetic_result, manual=True)
    state["_last_queue_release_ts"] = now_ts()
    save_state(state)
    record_admin_activity(state, user_id, "تأیید دستی انتشار با نتیجه نامشخص", item.get("title", ""))
    suffix = (", ".join(str(x) for x in ids) if ids else "بدون شناسه پیام؛ تأیید دستی مدیر")
    admin_render(chat_id, "✅ رکورد به‌عنوان منتشرشده ثبت شد و از صف خارج شد. شناسه‌ها: " + suffix, [[admin_button("تاریخچه", "history")], [admin_button("صف", "queue")]], message_id)


def handle_admin_callback(state, user_id, chat_id, callback, role):
    data = str(callback.get("data", ""))
    msg = callback.get("message", {})
    message_id = safe_int(msg.get("message_id"), 0) or None
    parts = data.split("|")
    action = parts[0]
    admin_answer_callback(callback.get("id", ""))

    area_by_action = {
        "menu": "stats", "stats": "stats", "settings": "settings", "setting": "settings", "setprompt": "settings",
        "news": "history_view" if role == "observer" else "news", "queue": "queue", "queuepage": "queue",
        "q": "queue", "qedit": "queue", "qimg": "queue", "qdelay": "queue", "qpub": "queue", "qdel": "queue",
        "history": "history_view" if role == "observer" else "news", "historypage": "history_view" if role == "observer" else "news", "h": "history_view" if role == "observer" else "news",
        "hedit": "news", "himg": "news", "hdelete": "news", "ai": "ai_view" if role == "observer" else "ai", "aiprovider": "ai_view" if role == "observer" else "ai",
        "aienable": "ai", "aiprimary": "ai", "admins": "admins", "adminadd": "admins", "adminrole": "admins", "sources": "sources", "sourceadd": "sources",
        "logs": "logs", "decisions": "decisions", "tools": "diagnose" if role == "observer" else "tools", "diagnose": "diagnose", "testmode": "tools", "dedupqueue": "queue", "confirm": "settings",
        "aitest": "ai", "primary": "ai", "do": "stats",
        "v53": "tools", "pause": "tools", "resume": "tools", "profile": "tools",
        "blockadd": "tools", "keyadd": "tools", "silenceadd": "tools", "sourcetoggle": "sources",
        "sourcestats": "sources", "metrics": "tools", "historysearch": "news", "exportcsv": "news",
        "qexact": "queue", "qbump": "queue", "qpin": "queue", "qrewrite": "queue", "qrewritecustom": "queue", "qsources": "queue", "qresolve": "queue", "qverify": "queue",
        "find": "news", "terms": "tools", "ruleadd": "tools", "rules": "tools", "ruletoggle": "tools", "ruledelete": "tools", "rulesgate": "tools", "qapprove": "queue",
    }
    area = area_by_action.get(action, "stats")
    if not admin_can(role, area):
        admin_render(chat_id, "این سطح دسترسی اجازه این عملیات را ندارد.", [[admin_button("منوی اصلی", "menu")]], message_id)
        return

    if action == "v53":
        render_v53_menu(state, chat_id, message_id)
    elif action == "pause" and len(parts) > 1:
        pause_mode = parts[1]
        if pause_mode == "1h":
            state["_paused_until"] = now_ts() + 3600; state["_pause_until_important"] = False
        elif pause_mode == "tomorrow":
            tomorrow = now_tehran().date() + timedelta(days=1)
            target = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 7, 0, tzinfo=TEHRAN_TZ or timezone.utc)
            state["_paused_until"] = target.timestamp(); state["_pause_until_important"] = False
        elif pause_mode == "important":
            state["_paused_until"] = 0; state["_pause_until_important"] = True
        save_state(state); record_admin_activity(state, user_id, "توقف زمان‌دار", pause_mode)
        render_v53_menu(state, chat_id, message_id)
    elif action == "resume":
        state["_paused_until"] = 0; state["_pause_until_important"] = False
        save_state(state); record_admin_activity(state, user_id, "ادامه انتشار", "پنل V5.3")
        render_v53_menu(state, chat_id, message_id)
    elif action == "profile" and len(parts) > 1:
        profile = parts[1]
        st = admin_settings(state)
        if profile == "war":
            st.update({"active_start_hour": 0, "afternoon_start_hour": 14, "active_end_hour": 24,
                       "morning_spacing_minutes": 20, "afternoon_spacing_minutes": 20})
            state["_profile"] = "war"
        else:
            st.update({"active_start_hour": 8, "afternoon_start_hour": 14, "active_end_hour": 24,
                       "morning_spacing_minutes": MORNING_POST_SPACING_MINUTES,
                       "afternoon_spacing_minutes": AFTERNOON_POST_SPACING_MINUTES})
            state["_profile"] = "normal"
        save_state(state); record_admin_activity(state, user_id, "تغییر پروفایل", profile)
        render_v53_menu(state, chat_id, message_id)
    elif action == "blockadd":
        ask_admin_input(state, user_id, chat_id, {"type": "block_term"}, "واژه/عبارت ممنوع را بفرست؛ پست خام دارای این عبارت پیش از AI رد می‌شود.", message_id)
    elif action == "keyadd":
        ask_admin_input(state, user_id, chat_id, {"type": "priority_term"}, "واژه/عبارت اولویت‌دار را بفرست؛ خبرهای منطبق امتیاز اولویت می‌گیرند.", message_id)
    elif action == "silenceadd":
        ask_admin_input(state, user_id, chat_id, {"type": "topic_silence"}, "قالب: عبارت,ساعت — نمونه: موشک بالستیک,24. انتشار موارد دارای این عبارت تا پایان مدت متوقف می‌شود.", message_id)
    elif action == "terms":
        render_terms(state, chat_id, message_id)
    elif action == "rules":
        render_auto_rules(state, chat_id, message_id)
    elif action == "ruleadd":
        ask_admin_input(state, user_id, chat_id, {"type": "auto_rule_add"}, "قالب: score>=85;source=ChannelName;bucket=iran OR score>=95;region=iran. در هر گروه شرط‌ها با AND هستند و بین گروه‌ها OR است. score>=... در هر گروه الزامی است.", message_id)
    elif action == "rulesgate":
        state["_rules_gate_enabled"] = not safe_bool(state.get("_rules_gate_enabled"))
        save_state(state); record_admin_activity(state, user_id, "تغییر حالت گیت قوانین", str(state["_rules_gate_enabled"]))
        render_auto_rules(state, chat_id, message_id)
    elif action == "ruletoggle" and len(parts) > 1:
        rule_id = parts[1]
        for rule in state.get("_auto_approve_rules", []):
            if str(rule.get("id")) == rule_id: rule["enabled"] = not safe_bool(rule.get("enabled", True))
        save_state(state); render_auto_rules(state, chat_id, message_id)
    elif action == "ruledelete" and len(parts) > 1:
        state["_auto_approve_rules"] = [x for x in state.get("_auto_approve_rules", []) if str(x.get("id")) != parts[1]]
        save_state(state); render_auto_rules(state, chat_id, message_id)
    elif action == "qresolve" and len(parts) > 2:
        item_id, resolution = parts[1], parts[2]
        item = find_queue_item(state, item_id)
        if not item:
            admin_render(chat_id, "خبر در صف پیدا نشد.", [[admin_button("صف", "queue")]], message_id)
        elif item.get("delivery_status") not in {"uncertain", "partial"}:
            admin_render(chat_id, "این خبر وضعیت نامشخص/ناقص ندارد؛ تغییری انجام نشد.", [[admin_button("بازگشت به خبر", f"q|{item_id}")]], message_id)
        elif resolution == "retry":
            item["delivery_status"] = "queued"
            item["delivery_error"] = ""
            item["delivery_uncertain_ts"] = 0
            item["_delivery_message_ids"] = []
            item["scheduled_at"] = 0
            save_state(state)
            record_admin_activity(state, user_id, "اجازه تلاش مجدد برای ارسال نامشخص", item.get("title", ""))
            admin_render(chat_id, "⚠️ خبر برای تلاش مجدد مجاز شد. اگر بخش‌هایی قبلاً در کانال منتشر شده باشند، امکان تکرار وجود دارد.", [[admin_button("بازکردن خبر", f"q|{item_id}")], [admin_button("صف", "queue")]], message_id)
        elif resolution == "sent":
            ids = [safe_int(x, 0) for x in item.get("_delivery_message_ids", []) if safe_int(x, 0)]
            if not ids and not item.get("message_id"):
                admin_render(chat_id, "هیچ شناسه پیامی ثبت نشده است. اگر کانال را بررسی کردی و مطمئنی پیام منتشر شده، می‌توانی فقط رکورد محلی را بدون شناسه ثبت کنی.", [[admin_button("✅ تأیید انتشار بدون شناسه", f"qresolve|{item_id}|sent_force")], [admin_button("بازگشت به خبر", f"q|{item_id}")]], message_id)
            else:
                _resolve_uncertain_as_sent(state, user_id, chat_id, item, item_id, ids, message_id)
        elif resolution == "sent_force":
            item = find_queue_item(state, item_id)
            if item and item.get("delivery_status") in {"uncertain", "partial"}:
                ids = [safe_int(x, 0) for x in item.get("_delivery_message_ids", []) if safe_int(x, 0)]
                _resolve_uncertain_as_sent(state, user_id, chat_id, item, item_id, ids, message_id)
            else:
                admin_render(chat_id, "خبر در وضعیت قابل حل نیست.", [[admin_button("صف", "queue")]], message_id)
        else:
            admin_render(chat_id, "گزینه حل وضعیت ناشناخته است.", [[admin_button("بازگشت به خبر", f"q|{item_id}")]], message_id)
    elif action == "qapprove" and len(parts) > 1:
        item = find_queue_item(state, parts[1])
        if not item:
            admin_render(chat_id, "خبر پیدا نشد.", [[admin_button("صف", "queue")]], message_id)
        else:
            item["manual_approval_required"] = False; item["manual_review_approved_by"] = user_id; item["manual_review_approved_ts"] = now_ts()
            save_state(state); record_admin_activity(state, user_id, "تأیید دستی خبر", item.get("title", ""))
            render_queue_item(state, chat_id, parts[1], message_id)
    elif action == "metrics":
        render_performance(state, chat_id, message_id)
    elif action == "sourcestats":
        render_source_stats(state, chat_id, message_id)
    elif action == "sourcetoggle" and len(parts) > 1:
        source = str(parts[1]).strip().lstrip("@")
        disabled = set(state.get("_disabled_sources", []))
        if source in disabled: disabled.remove(source)
        else: disabled.add(source)
        state["_disabled_sources"] = sorted(disabled)
        save_state(state); record_admin_activity(state, user_id, "تغییر وضعیت منبع", f"{source}: {'فعال' if source not in disabled else 'غیرفعال'}")
        render_sources(state, chat_id, message_id)
    elif action == "historysearch":
        ask_admin_input(state, user_id, chat_id, {"type": "history_search"}, "عبارت جست‌وجو را بفرست. جست‌وجو در تیتر، متن، نام منبع، bucket و شناسه پیام انجام می‌شود.", message_id)
    elif action == "exportcsv":
        send_history_csv(state, chat_id)
        admin_render(chat_id, "✅ فایل CSV تاریخچه آماده شد.", [[admin_button("بازگشت به V5.3", "v53")]], message_id)
    elif action == "find" and len(parts) > 1:
        render_find_message(state, chat_id, parts[1], message_id)
    elif action == "qexact" and len(parts) > 1:
        ask_admin_input(state, user_id, chat_id, {"type": "schedule_exact", "item_id": parts[1]}, "تاریخ و ساعت را به وقت تهران وارد کن؛ قالب دقیق: YYYY-MM-DD HH:MM (مثال: 2026-10-20 21:30).", message_id)
    elif action == "qbump" and len(parts) > 2:
        item_id, where = parts[1], parts[2]
        item = find_queue_item(state, item_id)
        if not item:
            admin_render(chat_id, "خبر در صف پیدا نشد.", [[admin_button("صف", "queue")]], message_id)
        else:
            if where == "top": item["manual_priority"] = 10000; item["queued_at"] = now_ts()
            else: item["manual_priority"] = -10000; item["queued_at"] = now_ts() + 86400
            save_state(state); record_admin_activity(state, user_id, "تغییر اولویت صف", f"{where}: {item.get('title','')}")
            render_queue_item(state, chat_id, item_id, message_id)
    elif action == "qpin" and len(parts) > 1:
        item_id = parts[1]; item = find_queue_item(state, item_id)
        if not item:
            admin_render(chat_id, "خبر در صف پیدا نشد.", [[admin_button("صف", "queue")]], message_id)
        else:
            item["scheduled_at"] = 0; item["manual_priority"] = 20000
            state["_priority_pin_id"] = item_id
            save_state(state); record_admin_activity(state, user_id, "Pin خبر صف", item.get("title", ""))
            render_queue_item(state, chat_id, item_id, message_id)
    elif action == "qrewrite" and len(parts) > 1:
        if len(parts) > 2:
            rewrite_queue_item_with_style(state, user_id, chat_id, parts[1], parts[2], message_id)
        else:
            rows = [
                [admin_button("کوتاه‌تر", f"qrewrite|{parts[1]}|short"), admin_button("رسمی‌تر", f"qrewrite|{parts[1]}|formal")],
                [admin_button("با زمینه بیشتر", f"qrewrite|{parts[1]}|context"), admin_button("اصلاح تیتر", f"qrewrite|{parts[1]}|headline")],
                [admin_button("واردکردن دستور دلخواه", f"qrewritecustom|{parts[1]}")],
                [admin_button("⬅️ بازگشت به خبر", f"q|{parts[1]}")],
            ]
            admin_render(chat_id, "سبک بازنویسی را انتخاب کن. نسخه قبلی حفظ می‌شود و متن جدید باید پیش از انتشار بازبینی شود.", rows, message_id)
    elif action == "qrewritecustom" and len(parts) > 1:
        ask_admin_input(state, user_id, chat_id, {"type": "rewrite_queue", "item_id": parts[1]}, "دستور بازنویسی دلخواه را بنویس. نسخه قبلی حفظ می‌شود.", message_id)
    elif action == "qverify" and len(parts) > 1:
        review_queue_item_with_ai(state, user_id, chat_id, parts[1], message_id)
    elif action == "qsources" and len(parts) > 1:
        item = find_queue_item(state, parts[1])
        if not item:
            admin_render(chat_id, "خبر در صف پیدا نشد.", [[admin_button("صف", "queue")]], message_id)
        else:
            urls = item.get("source_urls", []) if isinstance(item.get("source_urls"), list) else []
            lines = [f"منابع خبر: {item.get('title','')}", "", "کانال‌ها: " + (", ".join(item.get("source_channels", [])) or "ثبت نشده"), "پیوندها:"]
            lines.extend(urls[:12] or ["پیوند مستقلی ثبت نشده است."])
            admin_render(chat_id, "\n".join(lines), [[admin_button("⬅️ بازگشت به خبر", f"q|{parts[1]}")]], message_id)
    elif action == "menu":
        clear_admin_session(state, user_id)
        admin_show_menu(chat_id, message_id, role)
    elif action == "stats": render_stats(state, chat_id, message_id)
    elif action == "settings": render_settings(state, chat_id, message_id)
    elif action == "setting" and len(parts) > 1:
        key = parts[1]
        if key == "auto":
            st = admin_settings(state); st["auto_publish_enabled"] = not st.get("auto_publish_enabled", True)
            record_admin_activity(state, user_id, "تغییر انتشار خودکار", str(st["auto_publish_enabled"]))
            save_state(state); render_settings(state, chat_id, message_id)
        elif key == "emergency":
            current = admin_settings(state).get("emergency_stop", False)
            verb = "غیرفعال‌کردن" if current else "فعال‌کردن"
            admin_render(chat_id, f"تأیید می‌کنی {verb} توقف اضطراری؟ پنل مدیریت بعد از توقف همچنان فعال می‌ماند.", [[admin_button("✅ تأیید", f"confirm|emergency|1")], [admin_button("لغو", "settings")]], message_id)
    elif action == "setprompt" and len(parts) > 1:
        key = parts[1]
        prompts = {
            "morning_spacing": "فاصله بازه اول را به دقیقه بفرست (۵ تا ۱۴۴۰).",
            "afternoon_spacing": "فاصله بازه دوم را به دقیقه بفرست (۵ تا ۱۴۴۰).",
            "active_hours": "ساعات فعالیت را به شکل start,afternoon_start,end بفرست؛ مثال 8,14,24.",
            "daily_limit": "حداکثر تعداد پست روزانه را بفرست؛ عدد ۰ یعنی بدون سقف.",
            "ai_attempts": "تعداد تلاش برای هر سرویس را بفرست: ۱ یا ۲.",
        }
        ask_admin_input(state, user_id, chat_id, {"type": "setting_value", "key": key}, prompts.get(key, "مقدار جدید را بفرست."), message_id)
    elif action == "news":
        rows = [[admin_button("📥 صف انتشار", "queue")], [admin_button("🗃 تاریخچه منتشرشده", "history")], [admin_button("🧾 دلایل انتخاب/رد", "decisions")], [admin_button("⬅️ منوی اصلی", "menu")]]
        admin_render(chat_id, "📰 مدیریت اخبار", rows, message_id)
    elif action == "queue": render_queue(state, chat_id, message_id)
    elif action == "queuepage": render_queue(state, chat_id, message_id, safe_int(parts[1] if len(parts) > 1 else 0, 0))
    elif action == "q" and len(parts) > 1: render_queue_item(state, chat_id, parts[1], message_id)
    elif action == "qedit" and len(parts) > 1:
        ask_admin_input(state, user_id, chat_id, {"type": "edit_text", "item_id": parts[1]}, "تیتر را در خط اول و بدنه کامل خبر را در خط‌های بعدی بفرست.", message_id)
    elif action == "qimg" and len(parts) > 1:
        ask_admin_input(state, user_id, chat_id, {"type": "edit_image", "item_id": parts[1]}, "یک عکس را همین‌جا ارسال کن یا URL عکس با http/https را بفرست؛ برای حذف رسانه علامت - را بفرست.", message_id)
    elif action == "qdelay" and len(parts) > 1:
        ask_admin_input(state, user_id, chat_id, {"type": "schedule_delay", "item_id": parts[1]}, "تأخیر انتشار را به دقیقه بفرست؛ ۰ یعنی در اولین فرصت مجاز.", message_id)
    elif action in {"history", "historypage"}: render_history(state, chat_id, message_id, safe_int(parts[1] if len(parts) > 1 and action == "historypage" else 0, 0))
    elif action == "h" and len(parts) > 1: render_history_item(state, chat_id, parts[1], message_id)
    elif action == "hedit" and len(parts) > 1:
        if role == "observer":
            admin_render(chat_id, "ناظر فقط اجازه مشاهده تاریخچه را دارد.", [[admin_button("تاریخچه", "history")]], message_id); return
        ask_admin_input(state, user_id, chat_id, {"type": "edit_history_text", "history_id": parts[1]}, "تیتر را در خط اول و بدنه را در خط‌های بعدی بفرست.", message_id)
    elif action == "himg" and len(parts) > 1:
        if role == "observer":
            admin_render(chat_id, "ناظر فقط اجازه مشاهده تاریخچه را دارد.", [[admin_button("تاریخچه", "history")]], message_id); return
        ask_admin_input(state, user_id, chat_id, {"type": "edit_history_image", "history_id": parts[1]}, "عکس را همین‌جا به‌صورت تصویر ارسال کن یا URL عکس را بفرست.", message_id)
    elif action == "ai": render_ai_menu(state, chat_id, message_id)
    elif action == "aiprovider" and len(parts) > 1: render_ai_provider(state, chat_id, parts[1], message_id)
    elif action == "aienable" and len(parts) > 1:
        if role != "owner":
            admin_render(chat_id, "فعال/غیرفعال‌کردن ارائه‌دهندگان فقط در اختیار مالک اصلی است.", [[admin_button("بازگشت", "ai")]], message_id); return
        provider = parts[1]
        if provider in AI_PROVIDER_ORDER:
            enabled = admin_settings(state).setdefault("ai_enabled", {}).get(provider, True)
            if enabled and len(runtime_provider_order(state)) <= 1:
                admin_render(chat_id, "حداقل یک ارائه‌دهنده باید فعال بماند.", [[admin_button("بازگشت", f"aiprovider|{provider}")]], message_id); return
            admin_settings(state).setdefault("ai_enabled", {})[provider] = not enabled
            record_admin_activity(state, user_id, "تغییر وضعیت AI", f"{provider}={not enabled}")
            save_state(state); render_ai_provider(state, chat_id, provider, message_id)
    elif action == "aiprimary" and len(parts) > 1:
        if role != "owner":
            admin_render(chat_id, "تغییر ترتیب سرویس‌ها فقط در اختیار مالک اصلی است.", [[admin_button("AI", "ai")]], message_id); return
        provider = parts[1]
        if provider not in AI_PROVIDER_ORDER or not admin_settings(state).get("ai_enabled", {}).get(provider, True):
            admin_render(chat_id, "برای اصلی‌شدن، ابتدا یک ارائه‌دهنده معتبر و فعال انتخاب کن.", [[admin_button("AI", "ai")]], message_id); return
        old_order = admin_settings(state).get("ai_order", list(AI_PROVIDER_ORDER))
        admin_settings(state)["ai_order"] = [provider] + [x for x in old_order if x != provider]
        record_admin_activity(state, user_id, "تغییر ترتیب AI", " → ".join(admin_settings(state)["ai_order"]))
        save_state(state); render_ai_menu(state, chat_id, message_id)
    elif action == "aitest":
        ask_admin_input(state, user_id, chat_id, {"type": "ai_test"}, "متن خام خبر را بفرست. فقط ترجمه/تحلیل می‌شود؛ در صف و کانال قرار نمی‌گیرد.", message_id)
    elif action == "admins": render_admins(state, chat_id, message_id)
    elif action == "adminadd":
        if role != "owner":
            admin_render(chat_id, "فقط مالک اصلی می‌تواند ادمین اضافه کند.", [[admin_button("منوی اصلی", "menu")]], message_id); return
        ask_admin_input(state, user_id, chat_id, {"type": "add_admin_id"}, "شناسه عددی تلگرام فرد را بفرست؛ نه username.", message_id)
    elif action == "adminrole" and len(parts) > 2:
        if role != "owner": return
        pending = state.get("_admin_sessions", {}).get(str(user_id), {})
        new_id, new_role = parts[1], parts[2]
        if pending.get("type") != "add_admin_role" or str(pending.get("pending_id")) != new_id or new_role not in {"manager", "observer"}:
            admin_render(chat_id, "درخواست افزودن ادمین منقضی یا نامعتبر است.", [[admin_button("مدیریت ادمین‌ها", "admins")]], message_id); return
        if new_id == str(OWNER_TELEGRAM_ID):
            admin_render(chat_id, "مالک اصلی قابل تغییر نیست.", [[admin_button("مدیریت ادمین‌ها", "admins")]], message_id); return
        state.setdefault("_admin_users", {})[new_id] = {"role": new_role, "added_ts": now_ts(), "added_by": user_id}
        clear_admin_session(state, user_id)
        record_admin_activity(state, user_id, "افزودن ادمین", f"{new_id} / {new_role}")
        save_state(state); render_admins(state, chat_id, message_id)
    elif action == "sources": render_sources(state, chat_id, message_id)
    elif action == "sourceadd":
        ask_admin_input(state, user_id, chat_id, {"type": "source_add"}, "نام کاربری کانال عمومی را وارد کن؛ مثال: ChannelName.", message_id)
    elif action == "logs": render_logs(state, chat_id, message_id)
    elif action == "decisions": render_decisions(state, chat_id, message_id)
    elif action == "tools":
        if role == "observer":
            admin_render(chat_id, diagnose_system(state), [[admin_button("تازه‌سازی عیب‌یابی", "diagnose")], [admin_button("منوی اصلی", "menu")]], message_id)
        else:
            render_tools(state, chat_id, message_id)
    elif action == "diagnose":
        admin_render(chat_id, diagnose_system(state), [[admin_button("تازه‌سازی", "diagnose")], [admin_button("منوی اصلی", "menu")]], message_id)
    elif action == "testmode":
        current = is_test_mode(state)
        admin_settings(state)["test_mode_override"] = not current
        record_admin_activity(state, user_id, "تغییر حالت آزمایشی", str(not current))
        save_state(state); render_tools(state, chat_id, message_id)
    elif action == "dedupqueue":
        removed = deduplicate_queue_admin(state)
        record_admin_activity(state, user_id, "پاکسازی تکراری‌های صف", f"{removed} خبر حذف شد")
        admin_render(chat_id, f"پاکسازی تمام شد؛ {removed} مورد تکراری حذف شد.", [[admin_button("بازگشت به صف", "queue")]], message_id)
    elif action == "confirm" and len(parts) > 2:
        subaction, object_id = parts[1], parts[2]
        if subaction == "emergency" and role not in {"owner", "manager"}:
            admin_render(chat_id, "دسترسی کافی نیست.", [[admin_button("منوی اصلی", "menu")]], message_id); return
        if subaction in {"resetsettings", "admin_del"} and role != "owner":
            admin_render(chat_id, "این عملیات فقط در اختیار مالک است.", [[admin_button("منوی اصلی", "menu")]], message_id); return
        if subaction == "emergency":
            admin_confirm_action(state, user_id, chat_id, "emergency", object_id, message_id)
        elif subaction == "resetsettings":
            admin_render(chat_id, "تنظیمات انتشار به حالت پیش‌فرض برگردد؟", [[admin_button("تأیید", "do|resetsettings|0")], [admin_button("لغو", "settings")]], message_id)
        elif subaction == "sourcedel":
            admin_render(chat_id, f"منبع @{object_id} از فهرست حذف شود؟", [[admin_button("تأیید حذف", f"do|sourcedel|{object_id}")], [admin_button("لغو", "sources")]], message_id)
        elif subaction == "admin_del":
            admin_render(chat_id, f"دسترسی شناسه {object_id} حذف شود؟ مالک اصلی قابل حذف نیست.", [[admin_button("تأیید حذف", f"do|admin_del|{object_id}")], [admin_button("لغو", "admins")]], message_id)
        elif subaction == "hdelete":
            admin_render(chat_id, "این پست واقعاً از کانال حذف شود؟ این کار در کانال قابل مشاهده است.", [[admin_button("بله، حذف کن", f"do|hdelete|{object_id}")], [admin_button("لغو", f"h|{object_id}")]], message_id)
        elif subaction == "qpublish":
            admin_render(chat_id, "این خبر بدون انتظار برای زمان‌بندی در کانال منتشر شود؟", [[admin_button("بله، منتشر کن", f"do|qpublish|{object_id}")], [admin_button("لغو", f"q|{object_id}")]], message_id)
        elif subaction == "qdelete":
            admin_render(chat_id, "خبر از صف حذف/رد شود؟", [[admin_button("تأیید حذف", f"do|qdelete|{object_id}")], [admin_button("لغو", f"q|{object_id}")]], message_id)
    elif action == "do" and len(parts) > 2:
        subaction, object_id = parts[1], parts[2]
        if subaction in {"qpublish", "qdelete"} and not admin_can(role, "queue"):
            admin_render(chat_id, "این سطح دسترسی اجازه تغییر صف را ندارد.", [[admin_button("منوی اصلی", "menu")]], message_id); return
        if subaction in {"hdelete", "sourcedel"} and role not in {"owner", "manager"}:
            admin_render(chat_id, "این سطح دسترسی اجازه این تغییر را ندارد.", [[admin_button("منوی اصلی", "menu")]], message_id); return
        if subaction in {"resetsettings", "admin_del"} and role != "owner":
            admin_render(chat_id, "این عملیات فقط برای مالک مجاز است.", [[admin_button("منوی اصلی", "menu")]], message_id); return
        if subaction in {"hdelete", "qpublish", "qdelete", "sourcedel", "resetsettings", "admin_del"}:
            admin_confirm_action(state, user_id, chat_id, subaction, object_id, message_id)
    else:
        admin_render(chat_id, "این گزینه شناخته نشد یا منقضی شده است.", [[admin_button("منوی اصلی", "menu")]], message_id)


def _handle_admin_update_unlocked(state, update):
    user, chat = private_admin_user(update)
    user_id = safe_int(user.get("id"), 0)
    chat_id = safe_int(chat.get("id"), 0)
    if not user_id or not chat_id:
        callback = update.get("callback_query")
        if callback:
            admin_answer_callback(callback.get("id", ""), "پنل فقط در گفت‌وگوی خصوصی کار می‌کند.")
        return
    role = admin_role(state, user_id)
    if not role:
        callback = update.get("callback_query")
        if callback:
            admin_answer_callback(callback.get("id", ""), "دسترسی ندارید.")
        return
    callback = update.get("callback_query")
    if callback:
        handle_admin_callback(state, user_id, chat_id, callback, role)
        save_state(state)
        return
    message = update.get("message", {})
    text = message.get("text", "")
    session = state.get("_admin_sessions", {}).get(str(user_id), {})
    incoming_photos = message.get("photo") or []
    if incoming_photos and session.get("type") in {"edit_image", "edit_history_image"}:
        file_id = incoming_photos[-1].get("file_id")
        if not file_id:
            admin_render(chat_id, "عکس file_id معتبری نداشت؛ دوباره ارسال کن.")
            return
        try:
            if session.get("type") == "edit_image":
                item = find_queue_item(state, session.get("item_id"))
                if not item:
                    raise RuntimeError("خبر از صف حذف شده است.")
                item["photo_file_id"] = file_id
                item["photo"] = None
                item["video"] = None
                save_state(state)
                record_admin_activity(state, user_id, "تغییر عکس خبر صف", item.get("title", ""))
                clear_admin_session(state, user_id)
                admin_render(chat_id, "✅ عکس جدید به خبر صف متصل شد.", [[admin_button("پیش‌نمایش خبر", f"q|{item.get('admin_id')}")], [admin_button("صف", "queue")]])
            else:
                item = find_history_item(state, session.get("history_id"))
                if not item or item.get("deleted"):
                    raise RuntimeError("پست در تاریخچه فعال نیست.")
                old = (item.get("photo"), item.get("photo_file_id"), item.get("media_type"))
                item["photo_file_id"] = file_id
                item["photo"] = None
                item["media_type"] = "photo"
                try:
                    edit_published_media(item)
                except Exception:
                    item["photo"], item["photo_file_id"], item["media_type"] = old
                    raise
                save_state(state)
                record_admin_activity(state, user_id, "تغییر عکس پست منتشرشده", item.get("title", ""))
                clear_admin_session(state, user_id)
                admin_render(chat_id, "✅ عکس پست در کانال تغییر کرد.", [[admin_button("تاریخچه", "history")], [admin_button("منوی اصلی", "menu")]])
        except Exception as exc:
            admin_render(chat_id, "⚠️ تغییر عکس انجام نشد: " + sanitize_error_text(exc), [[admin_button("منوی اصلی", "menu")]])
        return
    if not text:
        return
    if text.startswith("/admin"):
        clear_admin_session(state, user_id)
        admin_show_menu(chat_id, role=role)
    else:
        handle_admin_text(state, user_id, chat_id, text)
    save_state(state)



def handle_admin_update(state, update):
    # Panel operations update the same queue/settings/history as the publishing loop.
    with STATE_LOCK:
        return _handle_admin_update_unlocked(state, update)


def admin_poll_loop(state):
    if not BOT_TOKEN or not OWNER_TELEGRAM_ID:
        log.warning("پنل مدیریت تلگرام غیرفعال است؛ OWNER_TELEGRAM_ID را در Railway Variables تنظیم کن.")
        return
    offset = safe_int(state.get("_telegram_update_offset"), 0)
    failures = 0
    while True:
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
                data={"offset": offset, "timeout": ADMIN_POLL_TIMEOUT_SECONDS,
                      "allowed_updates": json.dumps(["message", "callback_query"])},
                timeout=(10, ADMIN_POLL_TIMEOUT_SECONDS + 10),
            )
            if response.status_code == 409:
                raise RuntimeError("getUpdates با webhook یا poller دیگری تداخل دارد؛ webhook/نمونه دوم را بررسی کن.")
            response.raise_for_status()
            payload = response.json()
            if not payload.get("ok"):
                raise RuntimeError(sanitize_error_text(payload.get("description", "getUpdates failed")))
            failures = 0
            for update in payload.get("result", []):
                try:
                    handle_admin_update(state, update)
                except Exception as exc:
                    log.exception("پردازش یک رویداد پنل ناموفق بود: %s", sanitize_error_text(exc))
                    with STATE_LOCK:
                        record_system_error(state, "Admin", "پردازش رویداد تلگرام", exc)
                        save_state(state)
                offset = max(offset, safe_int(update.get("update_id"), 0) + 1)
                with STATE_LOCK:
                    state["_telegram_update_offset"] = offset
                    save_state(state)
        except Exception as exc:
            failures += 1
            log.warning("پنل مدیریت getUpdates: %s", sanitize_error_text(exc))
            if failures >= 3:
                with STATE_LOCK:
                    record_system_error(state, "Admin", "دریافت به‌روزرسانی تلگرام", exc)
                    save_state(state)
                failures = 0
                time.sleep(10)
            else:
                time.sleep(2)




# ============================================================
# v5.3 utilities: source controls, queue controls, reporting and alerts
# ============================================================

def is_blocked_or_silenced(state, text):
    return bool(block_or_silence_reason(state, text))


def block_or_silence_reason(state, text):
    normalized = normalize_for_match(text)
    for term in state.get("_blocklist_terms", []):
        if normalize_for_match(term) and normalize_for_match(term) in normalized:
            return f"عبارت ممنوع در فهرست بلاک‌لیست: {term}"
    active = []
    now = now_ts()
    for entry in state.get("_topic_silences", []):
        if not isinstance(entry, dict):
            continue
        if safe_float(entry.get("until_ts"), 0) <= now:
            continue
        term = normalize_for_match(entry.get("term", ""))
        if term and term in normalized:
            active.append(entry.get("term", term))
    if active:
        return "سکوت موضوعی موقت: " + ", ".join(active[:3])
    return ""


def cheap_classify_posts(state, posts):
    """پیش‌فیلتر اختیاری Groq ارزان؛ خطای طبقه‌بند fail-open است تا خبر از دست نرود."""
    if not CHEAP_CLASSIFIER_ENABLED or not posts or not GROQ_API_KEYS:
        return posts
    with STATE_LOCK:
        classifier_stats = state.setdefault("_v53_classifier", {"seen": 0, "passed": 0, "rejected": 0, "errors": 0})
        classifier_stats["seen"] = safe_int(classifier_stats.get("seen"), 0) + len(posts)
        save_state(state)
    rows = []
    for i, post in enumerate(posts, 1):
        post["_classifier_id"] = str(i)
        rows.append({"id": str(i), "title": post.get("title", ""), "text": truncate_text(post.get("text", ""), 950)})
    prompt = (
        "Classify public Telegram posts for a Persian military/geopolitical news channel. "
        "Keep only posts containing a substantive news event, official announcement, meaningful development, "
        "equipment/defense technology update, or significant analysis. Reject ads, channel promotion, greetings, "
        "unrelated entertainment, repeated boilerplate, and posts with no substantive information. "
        "Do not decide truth; only relevance. Return strict JSON: {\"keep_ids\":[\"1\",\"2\"]}.\nINPUT="
        + json.dumps(rows, ensure_ascii=False)
    )
    try:
        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEYS[0]}", "Content-Type": "application/json"},
            json={"model": CHEAP_CLASSIFIER_MODEL, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0, "max_tokens": 300, "response_format": {"type": "json_object"}},
            timeout=(8, 25),
        )
        response.raise_for_status()
        payload = response.json()
        content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
        try:
            parsed = json.loads(content)
        except Exception:
            match = re.search(r"\{.*\}", content, flags=re.S)
            if not match:
                raise ValueError("classifier JSON نامعتبر بود")
            parsed = json.loads(match.group(0))
        keep_ids = parsed.get("keep_ids") if isinstance(parsed, dict) else None
        if not isinstance(keep_ids, list):
            raise ValueError("classifier فهرست keep_ids برنگرداند")
        keep = {str(x) for x in keep_ids}
        kept, rejected = [], []
        for post in posts:
            if str(post.get("_classifier_id")) in keep:
                post["classification_model"] = CHEAP_CLASSIFIER_MODEL
                kept.append(post)
            else:
                rejected.append(post)
                src_key = post.get("_source_key")
                uid = post.get("uid")
                if src_key and uid:
                    with STATE_LOCK:
                        entries = state.setdefault(src_key, [])
                        if uid not in entries:
                            entries.append(uid)
                        state[src_key] = entries[-500:]
                channel = str(src_key or "").removeprefix("tg:")
                with STATE_LOCK:
                    stat = state.setdefault("_source_stats", {}).setdefault(channel, {"fetched": 0, "new": 0, "published": 0, "rejected": 0, "errors": 0})
                    stat["rejected"] = safe_int(stat.get("rejected"), 0) + 1
                    record_decision(state, post.get("title") or post.get("text", "")[:140], "رد‌شده", f"پیش‌فیلتر ارزان {CHEAP_CLASSIFIER_MODEL}: فاقد ارزش خبری آشکار")
        with STATE_LOCK:
            classifier_stats = state.setdefault("_v53_classifier", classifier_stats)
            classifier_stats["passed"] = safe_int(classifier_stats.get("passed"), 0) + len(kept)
            classifier_stats["rejected"] = safe_int(classifier_stats.get("rejected"), 0) + len(rejected)
            save_state(state)
        log.info("Cheap classifier %s: kept=%d rejected=%d; صرفه‌جویی قطعی توکن هنوز اندازه‌گیری نشده است", CHEAP_CLASSIFIER_MODEL, len(kept), len(rejected))
        return kept
    except Exception as exc:
        with STATE_LOCK:
            classifier_stats = state.setdefault("_v53_classifier", classifier_stats)
            classifier_stats["errors"] = safe_int(classifier_stats.get("errors"), 0) + 1
            save_state(state)
        log.warning("پیش‌فیلتر ارزان خطا داد؛ برای جلوگیری از حذف خبر، fail-open: %s", sanitize_error_text(exc))
        return posts


def _auto_rule_group_matches(group, item):
    if safe_int(item.get("news_score"), 0) < safe_int(group.get("min_score", 0), 0):
        return False
    sources = {str(x).lstrip("@").lower() for x in item.get("source_channels", [])}
    if group.get("source") and str(group.get("source")).lstrip("@").lower() not in sources:
        return False
    if group.get("bucket") and normalize_for_match(group.get("bucket")) != normalize_for_match(item.get("content_bucket", "")):
        return False
    if group.get("region") and normalize_for_match(group.get("region")) != normalize_for_match(item.get("region", "")):
        return False
    return True


def apply_auto_approve_rules(state, item):
    matched = None
    for rule in state.get("_auto_approve_rules", []):
        if not isinstance(rule, dict) or not safe_bool(rule.get("enabled", True)):
            continue
        groups = rule.get("groups")
        if isinstance(groups, list) and groups:
            ok = any(_auto_rule_group_matches(group, item) for group in groups if isinstance(group, dict))
        else:
            ok = _auto_rule_group_matches(rule, item)
        if ok:
            matched = rule
            break
    if matched:
        item["auto_approved_by_rule"] = matched.get("id")
        item["auto_approved"] = True
        item["manual_approval_required"] = False
    elif safe_bool(state.get("_rules_gate_enabled")):
        item["manual_approval_required"] = True
        item["auto_approved"] = False
    else:
        item["manual_approval_required"] = False
        item["auto_approved"] = False


def render_auto_rules(state, chat_id, message_id=None):
    rules = state.get("_auto_approve_rules", [])
    lines = ["✅ قوانین تأیید خودکار", "", f"حالت بررسی قانون برای همه خبرها: {'روشن (خبرهای بدون تطبیق نیازمند تأیید دستی‌اند)' if state.get('_rules_gate_enabled') else 'خاموش (رفتار قبلی ربات حفظ شده است)'}", ""]
    rows = []
    for rule in rules[-15:]:
        groups = rule.get("groups") if isinstance(rule.get("groups"), list) and rule.get("groups") else [rule]
        group_descriptions = []
        for group in groups:
            desc = f"امتیاز≥{group.get('min_score', '?')}"
            if group.get("source"): desc += f" | منبع:{group.get('source')}"
            if group.get("bucket"): desc += f" | سبد:{group.get('bucket')}"
            if group.get("region"): desc += f" | منطقه:{group.get('region')}"
            group_descriptions.append(desc)
        desc = " OR ".join(group_descriptions)
        lines.append(f"• {'فعال' if rule.get('enabled', True) else 'غیرفعال'} — {desc}")
        rows.append([admin_button(("🟢 " if rule.get('enabled', True) else "⚪ ") + desc[:38], f"ruletoggle|{rule.get('id')}") , admin_button("حذف", f"ruledelete|{rule.get('id')}")])
    if not rules: lines.append("هنوز قانونی ثبت نشده است.")
    rows.extend([[admin_button("➕ افزودن قانون", "ruleadd")], [admin_button("🔁 روشن/خاموش‌کردن گیت قوانین", "rulesgate")], [admin_button("⬅️ V5.3", "v53")]])
    lines.append("\nدر هر گروه، شرط‌ها با AND ارزیابی می‌شوند؛ گروه‌های جداشده با OR کافی است یکی تطبیق داشته باشد. چند قانون مستقل نیز مثل OR عمل می‌کنند. برای نمونه: score>=85;source=ChannelA OR score>=95;bucket=iran. برای الزام تأیید دستی خبرهای بدون قانون منطبق، گیت قوانین را روشن کن؛ حالت پیش‌فرض برای سازگاری با V5 خاموش است.")
    admin_render(chat_id, "\n".join(lines), rows, message_id)


def render_v53_menu(state, chat_id, message_id=None):
    paused_until = safe_float(state.get("_paused_until", 0), 0)
    if paused_until > now_ts():
        pause_text = datetime.fromtimestamp(paused_until, TEHRAN_TZ or timezone.utc).strftime("%Y-%m-%d %H:%M")
    elif state.get("_pause_until_important"):
        pause_text = "تا ورود خبر مهم/فوری"
    else:
        pause_text = "فعال؛ بدون توقف زمان‌دار"
    profile = "حالت جنگ" if state.get("_profile") == "war" else "حالت عادی"
    rows = [
        [admin_button("📈 تحلیل عملکرد", "metrics"), admin_button("📡 آمار منابع", "sourcestats")],
        [admin_button("🔎 جست‌وجوی تاریخچه", "historysearch"), admin_button("📄 خروجی CSV", "exportcsv")],
        [admin_button("🚫 واژه‌ها و سکوت موضوعی", "terms")],
        [admin_button("✅ قوانین تأیید خودکار", "rules")],
        [admin_button("➕ افزودن بلاک‌لیست", "blockadd"), admin_button("⭐ کلیدواژه اولویت", "keyadd")],
        [admin_button("🔕 سکوت موضوعی موقت", "silenceadd")],
        [admin_button("⏸ توقف ۱ ساعت", "pause|1h"), admin_button("⏸ تا فردا ۷ صبح", "pause|tomorrow")],
        [admin_button("⏸ تا خبر مهم", "pause|important"), admin_button("▶️ ادامه", "resume")],
        [admin_button("⚔️ حالت جنگ", "profile|war"), admin_button("🕊 حالت عادی", "profile|normal")],
        [admin_button("📥 صف انتشار", "queue"), admin_button("📡 منابع", "sources")],
        [admin_button("⬅️ منوی اصلی", "menu")],
    ]
    admin_render(chat_id, f"🧰 قابلیت‌های Raptor V{BOT_VERSION}\n\nپروفایل: {profile}\nوضعیت توقف: {pause_text}\nبلاک‌لیست: {len(state.get('_blocklist_terms', []))} عبارت\nکلیدواژه‌های اولویت: {len(state.get('_priority_keywords', []))}\nسکوت‌های فعال: {sum(1 for x in state.get('_topic_silences', []) if safe_float(x.get('until_ts'), 0) > now_ts())}\n\nتوجه: پروفایل‌ها فقط تنظیمات انتشار اخبار موجود را تغییر می‌دهند؛ سیستم خبر مستقل باقی می‌ماند.", rows, message_id)


def render_terms(state, chat_id, message_id=None):
    now = now_ts()
    blocked = state.get("_blocklist_terms", [])[-15:]
    priority = state.get("_priority_keywords", [])[-15:]
    silences = [x for x in state.get("_topic_silences", []) if isinstance(x, dict) and safe_float(x.get("until_ts"), 0) > now]
    lines = ["🚫 قوانین واژگانی و سکوت موضوعی", "", "بلاک‌لیست:"]
    lines.extend([f"• {x}" for x in blocked] or ["• خالی"])
    lines.append("\nکلیدواژه‌های اولویت:")
    lines.extend([f"• {x}" for x in priority] or ["• خالی"])
    lines.append("\nسکوت‌های فعال:")
    for entry in silences[-10:]:
        until = datetime.fromtimestamp(safe_float(entry.get("until_ts"), 0), TEHRAN_TZ or timezone.utc).strftime("%m-%d %H:%M")
        lines.append(f"• {entry.get('term')} تا {until}")
    rows = [
        [admin_button("➕ افزودن عبارت ممنوع", "blockadd")],
        [admin_button("➕ افزودن کلیدواژه اولویت", "keyadd")],
        [admin_button("🔕 افزودن سکوت موضوعی", "silenceadd")],
        [admin_button("⬅️ V5.3", "v53")],
    ]
    admin_render(chat_id, "\n".join(lines), rows, message_id)


def render_source_stats(state, chat_id, message_id=None):
    stats = state.get("_source_stats", {})
    health = state.get("_source_health", {})
    lines = [f"📡 عملکرد منابع — بروزرسانی {now_tehran().strftime('%Y-%m-%d %H:%M:%S')}", "شمارنده‌ها از داده‌های پردازش‌شده توسط همین ربات هستند؛ برای کانال‌های قدیمی آمار تاریخی کامل محسوب نمی‌شوند.", ""]
    for source in admin_settings(state).get("source_channels", SOURCE_CHANNELS):
        source = str(source).lstrip("@")
        st = stats.get(source, {})
        h = health.get(source, {})
        disabled = source in set(state.get("_disabled_sources", []))
        lines.append(f"@{source} {'[غیرفعال]' if disabled else ''}\n  دریافت‌شده: {safe_int(st.get('fetched'),0)} | جدید: {safe_int(st.get('new'),0)}\n  منتشرشده: {safe_int(st.get('published'),0)} | ردشده: {safe_int(st.get('rejected'),0)} | خطا: {safe_int(st.get('errors'),0)}\n  شکست متوالی: {safe_int(h.get('fail_count'),0)}")
    admin_render(chat_id, "\n\n".join(lines), [[admin_button("🔄 تازه‌سازی", "sourcestats")], [admin_button("مدیریت منابع", "sources")], [admin_button("V5.3", "v53")]], message_id)


def _try_tgstat_views(record):
    if not TGSTAT_API_TOKEN or not record.get("message_id"):
        return None
    target = str(TARGET_CHAT_ID or "").strip().lstrip("@")
    if not target or target.startswith("-"):
        return None
    post_url = f"https://t.me/{target}/{safe_int(record.get('message_id'), 0)}"
    try:
        response = requests.get("https://api.tgstat.ru/posts/get", params={"token": TGSTAT_API_TOKEN, "postId": post_url}, timeout=(5, 12))
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") == "ok":
            views = safe_int(payload.get("response", {}).get("views"), -1)
            if views >= 0:
                record["views"] = views
                record["views_checked_ts"] = now_ts()
                return views
    except Exception as exc:
        log.debug("TGStat views lookup failed: %s", sanitize_error_text(exc))
    return None


def render_performance(state, chat_id, message_id=None):
    history = [x for x in state.get("_published_history", []) if not x.get("deleted")]
    recent = [x for x in history if safe_float(x.get("ts"), 0) >= now_ts() - 7 * 86400]
    buckets, providers, sources = {}, {}, {}
    for rec in history:
        bucket = rec.get("content_bucket", "world") or "world"
        buckets[bucket] = buckets.get(bucket, 0) + 1
        provider = rec.get("ai_provider", "unknown") or "unknown"
        providers[provider] = providers.get(provider, 0) + 1
        for src in rec.get("source_channels", []): sources[src] = sources.get(src, 0) + 1
    views = [safe_int(x.get("views"), -1) for x in history if safe_int(x.get("views"), -1) >= 0]
    if TGSTAT_API_TOKEN and history:
        for rec in sorted(history, key=lambda x: safe_float(x.get("ts"), 0), reverse=True)[:5]:
            if now_ts() - safe_float(rec.get("views_checked_ts"), 0) > 6 * 3600:
                _try_tgstat_views(rec)
        views = [safe_int(x.get("views"), -1) for x in history if safe_int(x.get("views"), -1) >= 0]
        save_state(state)
    lines = [f"📈 تحلیل عملکرد — آخرین بروزرسانی {now_tehran().strftime('%Y-%m-%d %H:%M:%S')}", "", f"تعداد پست‌های ثبت‌شده در تاریخچه: {len(history)}", f"انتشار ۷ روز اخیر: {len(recent)}", f"پست‌های دارای آمار بازدید: {len(views)}"]
    if views:
        lines.append(f"میانگین بازدیدِ رکوردهای بررسی‌شده: {sum(views)//len(views)}")
    elif not TGSTAT_API_TOKEN:
        lines.append("بازدید واقعی در دسترس نیست؛ در صورت نیاز، TGSTAT_API_TOKEN را به‌صورت متغیر امن اضافه کن. اعداد بازدید از روی حدس ساخته نمی‌شوند.")
    lines.append("\nموضوعات/سبدهای پرتکرار:")
    lines.extend([f"• {k}: {v}" for k, v in sorted(buckets.items(), key=lambda x: x[1], reverse=True)[:8]] or ["• هنوز داده‌ای نیست"])
    lines.append("\nسهم ارائه‌دهنده در پست‌های منتشرشده:")
    lines.extend([f"• {k}: {v}" for k, v in sorted(providers.items(), key=lambda x: x[1], reverse=True)[:8]] or ["• هنوز داده‌ای نیست"])
    lines.append("\nمنابع پرتکرار در تاریخچه:")
    lines.extend([f"• @{k}: {v}" for k, v in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:8]] or ["• هنوز داده‌ای نیست"])
    topic_weights = {}
    have_view_metrics = any(safe_int(rec.get("views"), -1) >= 0 for rec in recent)
    for rec in recent:
        weight = max(0, safe_int(rec.get("views"), 0)) if have_view_metrics else 1
        title_tokens = [t for t in re.findall(r"[\w\u0600-\u06ff]{3,}", normalize_for_match(rec.get("title", "")))
                        if t not in STOPWORDS and not t.isdigit()]
        for token in set(title_tokens[:12]):
            topic_weights[token] = topic_weights.get(token, 0) + weight
    lines.append("\nموضوعات پرتکرار در تیترهای ۷ روز اخیر:" if not have_view_metrics else "\nموضوعات با بیشترین بازدید تجمیعیِ تیترهای ۷ روز اخیر:")
    lines.extend([f"• {k}: {v if have_view_metrics else 'تکرار '+str(v)}" for k, v in sorted(topic_weights.items(), key=lambda x: x[1], reverse=True)[:10]] or ["• هنوز داده‌ای نیست"])
    if not have_view_metrics:
        lines.append("این فهرست فقط فراوانی واژه‌ها در تیترهاست، نه رتبه‌بندی بر اساس بازدید؛ برای تحلیل بازدید واقعی TGSTAT_API_TOKEN لازم است.")
    classifier = state.get("_v53_classifier", {})
    if CHEAP_CLASSIFIER_ENABLED:
        lines.append(f"\nپیش‌فیلتر ارزان: دیده {safe_int(classifier.get('seen'),0)} | عبور {safe_int(classifier.get('passed'),0)} | رد {safe_int(classifier.get('rejected'),0)} | خطا {safe_int(classifier.get('errors'),0)}")
        lines.append("صرفه‌جویی ۶۰–۸۰٪ فرض نشده؛ اثر واقعی باید از داده مصرف/کیفیت سنجیده شود.")
    admin_render(chat_id, "\n".join(lines), [[admin_button("🔄 تازه‌سازی", "metrics")], [admin_button("📄 خروجی CSV", "exportcsv")], [admin_button("V5.3", "v53")]], message_id)


def render_history_search_results(state, chat_id, query, matches):
    rows = []
    lines = [f"🔎 نتیجه جست‌وجو: {query}", f"یافته‌ها: {len(matches)}", ""]
    for rec in matches[:12]:
        lines.append(f"• {rec.get('title','')[:90]} | msg_id={rec.get('message_id','—')} | AI={rec.get('ai_provider','unknown')}")
        if rec.get("history_id"):
            rows.append([admin_button(str(rec.get("title", "پست"))[:42], f"h|{rec['history_id']}")])
    if not matches: lines.append("موردی پیدا نشد.")
    rows.extend([[admin_button("جست‌وجوی دوباره", "historysearch")], [admin_button("خروجی CSV", "exportcsv")], [admin_button("V5.3", "v53")]])
    admin_render(chat_id, "\n".join(lines), rows)


def render_find_message(state, chat_id, message_id, edit_message_id=None):
    target = str(message_id).strip()
    for rec in reversed(state.get("_published_history", [])):
        ids = {str(rec.get("message_id", ""))}
        ids.update(str(x) for x in rec.get("message_ids", []) if x)
        if target in ids:
            stamp = datetime.fromtimestamp(safe_float(rec.get("ts"), 0), TEHRAN_TZ or timezone.utc).strftime("%Y-%m-%d %H:%M")
            text = (f"🔍 رکورد پیام {target}\n\nعنوان: {rec.get('title','')}\nزمان: {stamp}\nمنابع: {', '.join(rec.get('source_channels', [])) or 'نامشخص'}\nهوش مصنوعی: {rec.get('ai_provider','unknown')}\nامتیاز: {rec.get('news_score','—')}\nسبد: {rec.get('content_bucket','—')}\n\n{rec.get('body','')}")
            admin_render(chat_id, text, [[admin_button("بازکردن رکورد", f"h|{rec.get('history_id')}")], [admin_button("تاریخچه", "history")]], edit_message_id)
            return
    admin_render(chat_id, f"پیام {target} در تاریخچه موجود پیدا نشد. توجه: فقط پیام‌های ثبت‌شده در state قابل جست‌وجو هستند.", [[admin_button("تاریخچه", "history")], [admin_button("V5.3", "v53")]], edit_message_id)


def send_history_csv(state, chat_id):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["history_id", "timestamp", "message_id", "message_ids", "title", "body", "news_score", "verification", "ai_provider", "ai_provider_stage1", "ai_provider_editor", "classification_model", "source_channels", "source_urls", "region", "content_bucket", "views", "deleted"])
    for rec in state.get("_published_history", []):
        writer.writerow([rec.get("history_id", ""), rec.get("ts", ""), rec.get("message_id", ""), ";".join(str(x) for x in rec.get("message_ids", [])), rec.get("title", ""), rec.get("body", ""), rec.get("news_score", ""), rec.get("verification", ""), rec.get("ai_provider", ""), rec.get("ai_provider_stage1", ""), rec.get("ai_provider_editor", ""), rec.get("classification_model", "disabled"), ";".join(rec.get("source_channels", [])), ";".join(rec.get("source_urls", [])), rec.get("region", ""), rec.get("content_bucket", ""), rec.get("views", ""), rec.get("deleted", False)])
    data = output.getvalue().encode("utf-8-sig")
    telegram_request("sendDocument", data={"chat_id": chat_id, "caption": f"تاریخچه Raptor V{BOT_VERSION} — {len(state.get('_published_history', []))} رکورد"}, files={"document": ("raptor_history.csv", data, "text/csv")}, timeout=60)


def send_owner_notice(state, alert_key, text, cooldown_seconds=4*3600):
    if not OWNER_TELEGRAM_ID or not BOT_TOKEN:
        return False
    alerts = state.setdefault("_v53_alerts", {})
    last = safe_float(alerts.get(alert_key), 0)
    if now_ts() - last < cooldown_seconds:
        return False
    try:
        telegram_request("sendMessage", data={"chat_id": OWNER_TELEGRAM_ID, "text": f"🦅 هشدار Raptor V{BOT_VERSION}\n\n{text}"}, timeout=15)
        alerts[alert_key] = now_ts()
        save_state(state)
        return True
    except Exception as exc:
        log.warning("ارسال هشدار به مالک شکست خورد: %s", sanitize_error_text(exc))
        return False


def check_v53_alerts(state):
    enabled = runtime_provider_order(state)
    unhealthy = []
    for provider in enabled:
        configured = bool(AI_PROVIDER_KEYS.get(provider))
        ps = provider_state(state, provider)
        cooling = safe_float(ps.get("cooldown_until"), 0) > now_ts()
        if not configured or cooling:
            unhealthy.append(provider)
    if enabled and len(unhealthy) == len(enabled):
        send_owner_notice(state, "all_ai_down", "همه ارائه‌دهنده‌های AI فعال فاقد کلید یا در دوره انتظار/خطا هستند. بخش AI را بررسی کن.", 3600)
    for source, health in state.get("_source_health", {}).items():
        if safe_int(health.get("fail_count"), 0) >= 3:
            send_owner_notice(state, f"source:{source}", f"منبع @{source} دست‌کم سه بار متوالی شکست خورده است. آخرین خطا: {health.get('last_error','نامشخص')}", 4*3600)
    queue_len = len(state.get("_pending_queue", []))
    if queue_len >= ALERT_QUEUE_HIGH_WATER:
        send_owner_notice(state, "queue_high", f"صف انتشار به {queue_len} خبر رسیده است (آستانه {ALERT_QUEUE_HIGH_WATER}).", 4*3600)
    daily_limit = max(0, safe_int(admin_settings(state).get("daily_post_limit", 0), 0))
    if daily_limit:
        today_count = published_count_today(state)
        threshold = max(1, int(daily_limit * 0.8 + 0.999))
        if today_count >= threshold:
            send_owner_notice(state, "daily_limit_near", f"امروز {today_count} از سقف {daily_limit} پست منتشر شده است.", 4*3600)
    runtime = state.get("_runtime", {})
    last_post = safe_float(runtime.get("last_posted_ts"), 0) or safe_float(runtime.get("started_ts"), now_ts())
    if not is_quiet_hour(now_tehran(), state) and now_ts() - last_post >= ALERT_NO_POST_HOURS * 3600:
        send_owner_notice(state, "no_post", f"حدود {int((now_ts()-last_post)/3600)} ساعت است که پستی ثبت نشده؛ منابع، صف، ساعات سکوت، سقف روزانه و وضعیت AI را بررسی کن.", ALERT_NO_POST_HOURS*3600)


def maybe_send_weekly_summary(state):
    now = now_tehran()
    if now.isoweekday() != 7 or now.hour < 20:
        return
    week_key = f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"
    if state.get("_weekly_summary_last_week") == week_key:
        return
    cutoff = now_ts() - 7*86400
    recent = [x for x in state.get("_published_history", []) if safe_float(x.get("ts"), 0) >= cutoff and not x.get("deleted")]
    bucket_counts, ai_counts = {}, {}
    for rec in recent:
        bucket = rec.get("content_bucket", "world") or "world"
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        provider = rec.get("ai_provider", "unknown") or "unknown"
        ai_counts[provider] = ai_counts.get(provider, 0) + 1
    lines = [f"📊 خلاصه هفتگی رپتور ({week_key})", f"پست‌های ثبت‌شده: {len(recent)}", f"اندازه فعلی صف: {len(state.get('_pending_queue', []))}", "سبدهای پرتکرار:"]
    lines.extend([f"• {k}: {v}" for k, v in sorted(bucket_counts.items(), key=lambda x: x[1], reverse=True)[:5]] or ["• داده‌ای موجود نیست"])
    lines.append("ارائه‌دهنده‌های ثبت‌شده:")
    lines.extend([f"• {k}: {v}" for k, v in sorted(ai_counts.items(), key=lambda x: x[1], reverse=True)[:5]] or ["• داده‌ای موجود نیست"])
    if send_owner_notice(state, "weekly:" + week_key, "\n".join(lines), 1):
        state["_weekly_summary_last_week"] = week_key
        save_state(state)


# ============================================================
# Configuration validation
# ============================================================

def validate_config():
    missing = []
    if not BOT_TOKEN:
        missing.append("BOT_TOKEN")
    if not TARGET_CHAT_ID:
        missing.append("TARGET_CHAT_ID")
    if not SOURCE_CHANNELS:
        missing.append("SOURCE_CHANNELS")
    if not any(AI_PROVIDER_KEYS[p] for p in AI_PROVIDER_ORDER):
        missing.append("حداقل یکی از GROQ_API_KEYS / GEMINI_API_KEYS / MISTRAL_API_KEYS")

    if missing:
        raise RuntimeError(
            "این متغیرهای ضروری تنظیم نشده‌اند: " + ", ".join(missing)
        )

    for provider in AI_PROVIDER_ORDER:
        keys = AI_PROVIDER_KEYS[provider]
        duplicates = len(keys) - len({key_id(k) for k in keys})
        if duplicates:
            raise RuntimeError(
                f"{duplicates} کلید تکراری برای {provider} پیدا شد؛ کلیدها را بررسی کن."
            )


def main():
    validate_config()

    state = load_state()
    initialize_ai_state(state)
    # اگر فرایند وسط ارسال متوقف شده باشد، نتیجه را نامشخص می‌کنیم تا ارسال تکراری خودکار رخ ندهد.
    for queued in state.get("_pending_queue", []):
        if queued.get("delivery_status") == "sending":
            queued["delivery_status"] = "uncertain"
            queued["delivery_uncertain_ts"] = now_ts()
            queued["delivery_error"] = "فرایند هنگام انتشار متوقف شد؛ نتیجه ارسال مشخص نیست و بررسی مدیر لازم است."
    state.setdefault("_runtime", {})["started_ts"] = now_ts()
    state.setdefault("_runtime", {})["last_cycle_error"] = ""
    purge_recent_memories(state)
    purge_recent_inputs(state)
    save_state(state)

    log.info("==============================================")
    log.info("Raptor News Bot v%s شروع شد", BOT_VERSION)
    log.info("Telegram sources: %s", len(runtime_source_channels(state)))
    log.info(
        "AI pipeline: %s | keys: Groq=%s Gemini=%s Mistral=%s",
        ">".join(AI_PROVIDER_LABELS[x] for x in runtime_provider_order(state)) or "بدون ارائه‌دهنده فعال",
        len(GROQ_API_KEYS), len(GEMINI_API_KEYS), len(MISTRAL_API_KEYS),
    )
    log.info(
        "Models: Groq=%s | Gemini=%s | Mistral=%s",
        GROQ_MODEL, GEMINI_MODEL, MISTRAL_MODEL,
    )
    log.info("Editor enabled: %s", EDITOR_ENABLED)
    log.info("AI batch max posts: %s", AI_BATCH_MAX_POSTS)
    log.info("AI max attempts/provider: %s", AI_MAX_ATTEMPTS_PER_PROVIDER)
    log.info("AI min request interval: %.2fs", AI_MIN_REQUEST_INTERVAL_SECONDS)
    log.info("Dedup window: %s minutes | event memory: %s minutes", DEDUP_WINDOW_MINUTES, EVENT_MEMORY_WINDOW_MINUTES)
    log.info("Publish spacing: 08-14=%s دقیقه | 14-24=%s دقیقه", MORNING_POST_SPACING_MINUTES, AFTERNOON_POST_SPACING_MINUTES)
    log.info("TEST_MODE: %s", TEST_MODE)
    log.info("State file: %s", STATE_FILE)
    log.info("Owner ID configured: %s", bool(OWNER_TELEGRAM_ID))
    log.info("Runtime AI order: %s", " -> ".join(runtime_provider_order(state)))
    log.info("==============================================")

    if OWNER_TELEGRAM_ID:
        admin_thread = threading.Thread(target=admin_poll_loop, args=(state,), name="raptor-admin-poll", daemon=True)
        admin_thread.start()
    else:
        log.warning("برای فعال‌شدن /admin، OWNER_TELEGRAM_ID را در Railway Variables ثبت کن.")

    while True:
        started = now_ts()
        try:
            process_once(state)
        except Exception as exc:
            with STATE_LOCK:
                state.setdefault("_runtime", {})["last_cycle_error"] = sanitize_error_text(exc)
                record_system_error(state, "Main", "چرخه اصلی", exc)
                save_state(state)
            log.exception("خطای عمومی در چرخه اصلی: %s", sanitize_error_text(exc))
        elapsed = now_ts() - started
        sleep_for = max(5, POLL_INTERVAL_SECONDS - int(elapsed))
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
