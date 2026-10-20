"""Raptor V5.6 independent analytical-content subsystem.

All analytics records live under state['_analytics']; news queue/configuration is read-only.
Gemini is the default and exclusive provider for analytical research and content operations.
No operation publishes without explicit post approval and image validation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus, urlparse

import requests

log = logging.getLogger("raptor-analytics")
FOOTER = "\n\n#raptor\n————————\n@khaatshekaan"
MAX_POST_CHARS = 1020
MIN_POSTS = 5
MAX_POSTS = 6
MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_TZ = "Asia/Tehran"
SCHEMA_VERSION = 2

class AnalyticsError(RuntimeError):
    """Expected user-facing analytics error."""


def now_ts() -> float:
    return time.time()


def _default_settings() -> dict:
    return {
        "timezone": os.getenv("ANALYTICS_TIMEZONE", DEFAULT_TZ),
        "schedule_mode": "auto", "paused": False,
        "start_hour": 14, "end_hour": 21, "min_news_gap_minutes": 20,
        "require_image": True, "target_chat_id": os.getenv("ANALYTICS_TARGET_CHAT_ID", os.getenv("TARGET_CHAT_ID", "")).strip(),
        "auto_reschedule": True, "daily_project_limit": 1,
        "default_post_count": 6, "default_tone": "تحلیلی، طبیعی و گفت‌وگومحور",
        "updated_at": now_ts(),
    }


def ensure_state(state: dict) -> dict:
    """Migrate analytics schema in place while preserving all pre-existing data."""
    a = state.get("_analytics")
    if not isinstance(a, dict):
        a = {}
        state["_analytics"] = a
    a.setdefault("schema_version", SCHEMA_VERSION)
    a.setdefault("projects", {})
    a.setdefault("settings", {})
    for key, value in _default_settings().items():
        a["settings"].setdefault(key, value)
    a.setdefault("history", [])
    a.setdefault("runtime_errors", [])
    a.setdefault("created_at", now_ts())
    a["schema_version"] = SCHEMA_VERSION
    # Defensive migration: old or manually edited records should not break the UI loop.
    if not isinstance(a["projects"], dict):
        raise AnalyticsError("ساختار پروژه‌های تحلیلی خراب است؛ state را بدون نسخه پشتیبان بازنویسی نکن.")
    for p in a["projects"].values():
        if not isinstance(p, dict):
            continue
        for key, default in {
            "sources": [], "images": [], "posts": [], "errors": [], "versions": [],
            "article_versions": [], "approved_article": False, "approved_posts": False,
            "archived": False, "paused": False, "last_error": "", "history": [], "schedule_mode": "",
        }.items():
            p.setdefault(key, copy.deepcopy(default))
        for post in p.get("posts", []):
            if not isinstance(post, dict):
                continue
            for key, default in {
                "images": [], "image_candidates": [], "attempts": [], "versions": [],
                "status": "draft", "scheduled_ts": None, "telegram_message_id": None,
                "created_at": p.get("created_at", now_ts()), "updated_at": p.get("updated_at", now_ts()),
            }.items():
                post.setdefault(key, copy.deepcopy(default))
            post.setdefault("char_count", len(str(post.get("text", ""))))
    return a


def _append_history(project: dict, action: str, detail: str = "", user_id=None) -> None:
    event = {"at": now_ts(), "action": str(action)[:80], "detail": str(detail or "")[:500]}
    if user_id is not None:
        event["user_id"] = int(user_id)
    project.setdefault("history", []).append(event)
    project["history"] = project["history"][-300:]
    project["updated_at"] = now_ts()
    a_errors = project.setdefault("errors", [])
    if action == "error":
        a_errors.append({"at": event["at"], "error": event["detail"]})
        project["errors"] = a_errors[-100:]


def new_project(state: dict, kind: str, source_text: str, owner_id) -> dict:
    a = ensure_state(state)
    if kind not in {"research", "source_text"}:
        raise AnalyticsError("نوع پروژه نامعتبر است.")
    text = (source_text or "").strip()
    if not text:
        raise AnalyticsError("موضوع یا متن ورودی خالی است.")
    if len(text) > 60000:
        raise AnalyticsError("متن ورودی بیش از حد طولانی است (حداکثر ۶۰۰۰۰ نویسه).")
    pid = uuid.uuid4().hex[:12]
    title = text.splitlines()[0][:100].strip()
    project = {
        "id": pid, "title": title, "kind": kind,
        "source_text": text if kind == "source_text" else "",
        "topic": text if kind == "research" else "", "article": "", "sources": [], "images": [], "posts": [],
        "status": "queued", "owner_id": int(owner_id), "created_at": now_ts(), "updated_at": now_ts(),
        "errors": [], "versions": [], "article_versions": [], "approved_article": False,
        "approved_posts": False, "next_post_index": 0, "publish_day": "", "last_error": "",
        "schedule_mode": "", "archived": False, "paused": False, "history": [], "source_sufficient": None, "ai_provider_override": "gemini",
        "split_note": "", "research_grounding": {},
    }
    a["projects"][pid] = project
    _append_history(project, "created", f"kind={kind}", owner_id)
    return project


def _extract_text(payload: dict) -> str:
    try:
        return "".join(part.get("text", "") for cand in payload.get("candidates", [])[:1]
                       for part in cand.get("content", {}).get("parts", []) if isinstance(part, dict)).strip()
    except Exception:
        return ""


def _redact(value, keys=()) -> str:
    text = str(value or "")
    for secret in keys:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?i)(key=)[^&\s]+", r"\1[REDACTED]", text)
    text = re.sub(r"https://generativelanguage\.googleapis\.com/[^\s]+", "[GEMINI_ENDPOINT_REDACTED]", text)
    return text[:500]


def gemini_call(prompt: str, *, model=None, use_search=False, timeout=90):
    keys = [x.strip() for x in os.getenv("GEMINI_API_KEYS", "").split(",") if x.strip()]
    if not keys:
        raise AnalyticsError("GEMINI_API_KEYS تنظیم نشده است؛ عملیات هوش مصنوعی متوقف شد.")
    # Keep analytics model configuration independent from the news subsystem's GEMINI_MODEL.
    model = model or os.getenv("ANALYTICS_GEMINI_MODEL", "gemini-3.8-flash")
    model = re.sub(r"[^A-Za-z0-9._-]", "", model)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.25, "maxOutputTokens": 8192}}
    if use_search:
        body["tools"] = [{"google_search": {}}]
    errors = []
    for key in keys:
        try:
            r = requests.post(url, params={"key": key}, json=body, timeout=timeout)
            if r.status_code >= 400:
                errors.append(f"HTTP {r.status_code}: {_redact(r.text[:250], keys)}")
                continue
            payload = r.json()
            result = _extract_text(payload)
            if not result:
                errors.append("پاسخ Gemini فاقد متن بود.")
                continue
            return result, payload
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {_redact(exc, keys)}")
    # No automatic Groq/Mistral failover: user must explicitly initiate an alternative workflow.
    raise AnalyticsError("Gemini در تمام کلیدهای تنظیم‌شده ناموفق بود. عملیات متوقف شد و مقاله ناقص تأیید/منتشر نشد. " + " | ".join(errors[-3:]))


def _openai_compatible_call(provider: str, prompt: str, timeout=90):
    """Manual-only alternative provider. The caller must explicitly choose it in the UI."""
    provider = str(provider or "").lower()
    if provider == "groq":
        keys = [x.strip() for x in os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", "")).split(",") if x.strip()]
        url = "https://api.groq.com/openai/v1/chat/completions"
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    elif provider == "mistral":
        keys = [x.strip() for x in os.getenv("MISTRAL_API_KEYS", os.getenv("MISTRAL_API_KEY", "")).split(",") if x.strip()]
        url = "https://api.mistral.ai/v1/chat/completions"
        model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    else:
        raise AnalyticsError("ارائه‌دهنده جایگزین پشتیبانی نمی‌شود.")
    if not keys:
        raise AnalyticsError(f"کلید API برای {provider} تنظیم نشده است؛ پروژه تغییری نکرد.")
    errors = []
    for key in keys:
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json={
                "model": model,
                "messages": [{"role": "system", "content": "دستیار پژوهشی دقیق فارسی‌زبان؛ هیچ منبع یا واقعیتی را جعل نکن."},
                             {"role": "user", "content": prompt}],
                "temperature": 0.25, "max_tokens": 8192,
            }, timeout=timeout)
            if r.status_code >= 400:
                errors.append(f"HTTP {r.status_code}: {_redact(r.text[:200], keys)}")
                continue
            obj = r.json()
            content = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
            if isinstance(content, list):
                content = "".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
            if content and str(content).strip():
                return str(content).strip(), obj
            errors.append("پاسخ فاقد متن بود.")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {_redact(exc, keys)}")
    raise AnalyticsError(f"ارائه‌دهنده جایگزین {provider} ناموفق بود: {' | '.join(errors[-2:])}")


def project_ai_call(project: dict, prompt: str, *, use_search=False, timeout=90):
    """Gemini by default; alternate provider is used only after an explicit admin selection."""
    provider = str(project.get("ai_provider_override") or "gemini").lower()
    if provider == "gemini":
        return gemini_call(prompt, use_search=use_search, timeout=timeout)
    if use_search:
        prompt += "\nتوجه: این درخواست با ارائه‌دهنده جایگزین انجام می‌شود؛ منابع واقعی پیوست‌شده را مبنا قرار بده."
    return _openai_compatible_call(provider, prompt, timeout=timeout)


def web_search(query: str, limit=8):
    """Retrieve real search results via DuckDuckGo HTML; failure is explicit."""
    q = (query or "").strip()
    if not q:
        return []
    try:
        r = requests.get("https://html.duckduckgo.com/html/", params={"q": q},
                         headers={"User-Agent": "Mozilla/5.0 (compatible; RaptorV56/1.0)"}, timeout=20)
        r.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")
        results, seen = [], set()
        for item in soup.select(".result")[:max(1, min(12, int(limit)) * 2)]:
            link = item.select_one("a.result__a")
            snippet = item.select_one(".result__snippet")
            if not link:
                continue
            href = link.get("href", "")
            parsed = urlparse(href)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or href in seen:
                continue
            seen.add(href)
            results.append({"title": link.get_text(" ", strip=True), "url": href,
                            "snippet": snippet.get_text(" ", strip=True) if snippet else "",
                            "retrieved_at": now_ts()})
            if len(results) >= limit:
                break
        return results
    except Exception as exc:
        raise AnalyticsError(f"جست‌وجوی واقعی وب ناموفق بود؛ تحقیق بدون جست‌وجو ادامه داده نشد. {type(exc).__name__}: {_redact(exc)}")


def _split_article_sources(text: str):
    """Keep the publishable article separate from source metadata."""
    raw = (text or "").strip()
    article = raw
    match = re.search(r"(?is)(?:^|\n)\s*(?:SOURCES|منابع|منابع مورد استفاده)\s*:?\s*\n", raw)
    if match:
        article = raw[:match.start()].strip()
    article = re.sub(r"(?is)^\s*(?:ARTICLE|مقاله)\s*:?\s*\n", "", article).strip()
    # Citation markers remain in the research record but are not rendered into channel posts.
    return article


def research_project(project: dict):
    topic = project.get("topic", "").strip()
    if not topic:
        raise AnalyticsError("موضوع تحقیق خالی است.")
    results = web_search(topic + " official technical history defense research", 8)
    if len(results) < 3:
        raise AnalyticsError(f"جست‌وجوی وب فقط {len(results)} نتیجه قابل استفاده داد؛ حداقل ۳ نتیجه لازم است تا تحقیق ضعیف به مقاله تبدیل نشود.")
    sources = [{"title": x["title"], "url": x["url"], "snippet": x["snippet"],
                "retrieved_at": x.get("retrieved_at", now_ts()), "status": "needs_review"} for x in results]
    evidence = "\n\n".join(f"[{i+1}] {s['title']}\nURL: {s['url']}\nخلاصه نتیجه: {s['snippet']}" for i, s in enumerate(sources))
    prompt = (
        "نقش: پژوهشگر و نویسنده فارسی‌زبان. از نتایج جست‌وجوی واقعی پیوست و در صورت نیاز Google Search grounding استفاده کن. "
        "منابع را در گزارش مدیریتی جدا کن؛ نام منابع/مستندات، فهرست منبع و هیچ URL را داخل ARTICLE نیاور و فقط در SOURCES ثبت کن. "
        "هیچ عدد، تاریخ، منبع یا پیوندی جعل نکن. گزارش منابع را جای مقاله نگذار. "
        "میان داده تأییدشده، گزارش تأییدنشده، برآورد/تحلیل، اختلاف منابع و اطلاعات ناکافی تمایز بگذار. "
        "مشخصات نسخه‌های مختلف تجهیزات را مخلوط نکن. تاریخ انتشار منبع و تاریخ رویداد را در صورت وجود تفکیک کن. "
        "اگر شواهد کافی نیست، آن محدودیت را در متن مدیریتی و نه در ادعای قطعی توضیح بده. "
        "خروجی فارسی و دقیق با دو بخش ARTICLE و SOURCES باشد؛ SOURCES فقط در پنل مدیر استفاده می‌شود.\n\n"
        f"موضوع: {topic}\n\nنتایج جست‌وجوی وب:\n{evidence}"
    )
    result, raw = project_ai_call(project, prompt, use_search=(project.get("ai_provider_override") or "gemini") == "gemini")
    article = _split_article_sources(result)
    if len(article) < 100:
        raise AnalyticsError("خروجی تحقیق به مقاله کافی تبدیل نشد؛ پروژه برای بازبینی/تلاش مجدد حفظ شد.")
    project.setdefault("article_versions", []).append({"at": now_ts(), "article": project.get("article", ""), "action": "before_research"})
    project["article"] = article
    project["sources"] = sources
    project["approved_article"] = False
    project["approved_posts"] = False
    for old_post in project.get("posts", []):
        if old_post.get("status") not in {"published", "sending", "uncertain"}:
            old_post["status"] = "needs_review"
            old_post["scheduled_ts"] = None
            old_post["manual_schedule"] = False
    candidate = (raw.get("candidates") or [{}])[0] if isinstance(raw, dict) else {}
    grounding = candidate.get("groundingMetadata") if isinstance(candidate, dict) else None
    known_urls = {str(x.get("url", "")) for x in sources}
    if isinstance(grounding, dict):
        for chunk in grounding.get("groundingChunks", []) or []:
            web = chunk.get("web", {}) if isinstance(chunk, dict) else {}
            url = str(web.get("uri", "")).strip()
            if not url or url in known_urls or not url.startswith("https://"):
                continue
            sources.append({"title": str(web.get("title", "منبع Google Search grounding"))[:240], "url": url,
                            "snippet": "منبع برگشتی از Google Search grounding؛ محتوای صفحه باید توسط مدیر بررسی شود.",
                            "retrieved_at": now_ts(), "status": "needs_review", "origin": "gemini_grounding"})
            known_urls.add(url)
    project["sources"] = sources
    project["research_grounding"] = {
        "search_results": len(sources), "duckduckgo_results": len(results),
        "gemini_grounding_metadata": bool(grounding), "completed_at": now_ts(),
        "provider": project.get("ai_provider_override") or "gemini",
        "method": "DuckDuckGo HTML + Gemini Google Search grounding attempted" if (project.get("ai_provider_override") or "gemini") == "gemini" else "DuckDuckGo HTML + manually selected alternate provider",
    }
    project["status"] = "review_research"
    project["last_error"] = ""
    _append_history(project, "research_completed", f"sources={len(sources)}")
    return project


def _json_from_model(text: str) -> dict:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < start:
        raise AnalyticsError("خروجی مدل JSON معتبر ندارد.")
    try:
        obj = json.loads(t[start:end + 1])
    except Exception as exc:
        raise AnalyticsError(f"JSON خروجی مدل نامعتبر است: {str(exc)[:160]}") from exc
    if not isinstance(obj, dict):
        raise AnalyticsError("ساختار JSON خروجی باید شیء باشد.")
    return obj


def _clean_body(text: str) -> str:
    body = re.sub(r"\[(?:\d+)\]", "", str(text or "")).strip()
    body = re.sub(r"(?m)^\s*(?:منابع|Sources)\s*:\s*.*$", "", body, flags=re.I)
    body = re.sub(r"https?://\S+", "", body, flags=re.I)
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = re.sub(re.escape(FOOTER) + r"\s*$", "", body).strip()
    return body


def _footer_text(text: str) -> str:
    body = _clean_body(text)
    return (body + FOOTER).strip() if body else ""


def _call_split_model(project: dict, source: str, target_count: int, tone: str, strict: bool):
    char_cap = 900 if strict else 925
    prompt = (
        "متن زیر را به پست‌های مستقل اما پیوسته فارسی تلگرام تبدیل کن. خروجی فقط JSON معتبر باشد: "
        '{"posts":[{"title":"...","text":"...","image_query":"..."}],"note":"...","source_sufficient":true}. '
        f"هدف ۵ تا ۶ پست است؛ سقف خروجی حداکثر {MAX_POSTS} پست. تعداد هدف {target_count}. "
        "فقط اگر منبع برای ۵ پست محتوای متمایز کافی ندارد کمتر تولید کن و علت را در note توضیح بده. "
        f"هر متن پیش از footer حداکثر {char_cap} نویسه باشد. هیچ متن را با قطع ناگهانی جمله کوتاه نکن. "
        "هر پست یک موضوع اصلی داشته باشد. پست اول معرفی جذاب و اهمیت موضوع را بگو؛ پست آخر نتیجه را روشن کند و پایان واقعی باشد. "
        "به متن ورودی وفادار بمان؛ اطلاعات، اعداد، تاریخ‌ها و توانمندی جدید نساز. ارجاع‌های [n]، URL و فهرست منابع را وارد متن پست نکن. "
        "برچسب‌های داخلی مانند تأییدشده با منبع، گزارش مستقل‌تأییدنشده، برآورد، اختلاف منابع یا اطلاعات ناکافی را به شکل برچسب/تیتر در کانال نمایش نده؛ اگر لازم است تردید را با نثر طبیعی و دقیق حفظ کن. "
        "لحن طبیعی، دقیق، گفت‌وگومحور و متناسب با کانال فارسی‌زبان نظامی؛ بدون اغراق، بدون یادداشت ویراستاری. "
        "برای هر پست image_query مشخص و مرتبط برای جست‌وجوی عکس واقعی پیشنهاد بده."
        + (" طول هر پست را با حاشیه ایمنی بیشتر نگه دار." if strict else "")
        + f"\nلحن: {tone}\n\nمتن منبع:\n{source}"
    )
    response, _ = project_ai_call(project, prompt)
    obj = _json_from_model(response)
    posts = obj.get("posts")
    if not isinstance(posts, list):
        raise AnalyticsError("فیلد posts در خروجی مدل وجود ندارد.")
    return obj, posts


def split_into_posts(project: dict, count=None, tone=None):
    source = (project.get("article") or project.get("source_text") or "").strip()
    if not source:
        raise AnalyticsError("مقاله یا متن ورودی موجود نیست.")
    target_count = max(MIN_POSTS, min(MAX_POSTS, int(count or MAX_POSTS)))
    tone = tone or os.getenv("ANALYTICS_DEFAULT_TONE", "تحلیلی، طبیعی و گفت‌وگومحور")
    # First model attempt. If post validation fails, ask Gemini to repair before rejecting.
    last_error = ""
    for attempt in range(2):
        try:
            obj, raw_posts = _call_split_model(project, source, target_count, tone, strict=(attempt == 1))
            if not 1 <= len(raw_posts) <= MAX_POSTS:
                raise AnalyticsError("تعداد خروجی پست‌ها خارج از محدوده است.")
            candidate_posts = []
            for i, item in enumerate(raw_posts[:MAX_POSTS], 1):
                if not isinstance(item, dict):
                    continue
                body = _clean_body(item.get("text", ""))
                text = _footer_text(body)
                if not body:
                    continue
                if len(text) > MAX_POST_CHARS or len(text.encode("utf-16-le")) // 2 > 1024:
                    raise AnalyticsError(f"پست {i} بعد از افزودن footer از سقف مجاز عبور کرده است.")
                candidate_posts.append({
                    "id": uuid.uuid4().hex[:12], "number": i,
                    "title": str(item.get("title", f"بخش {i}"))[:140],
                    "text": text, "char_count": len(text),
                    "image_query": str(item.get("image_query", project.get("title", "military equipment")))[:240],
                    "images": [], "image_candidates": [], "status": "draft", "scheduled_ts": None,
                    "telegram_message_id": None, "created_at": now_ts(), "updated_at": now_ts(),
                    "attempts": [], "versions": [], "last_error": "", "manual_schedule": False,
                })
            if not candidate_posts:
                raise AnalyticsError("هیچ پست غیرخالی تولید نشد.")
            validation = validate_post_set(candidate_posts, require_images=False)
            if validation:
                raise AnalyticsError("اعتبارسنجی مستقل پست‌ها ناموفق بود: " + "; ".join(validation[:4]))
            old = project.get("posts", [])
            if old:
                project.setdefault("versions", []).append({"at": now_ts(), "action": "before_regenerate_all", "posts": copy.deepcopy(old)})
                project["versions"] = project["versions"][-50:]
            project["posts"] = candidate_posts
            project["split_note"] = str(obj.get("note", ""))[:500]
            project["source_sufficient"] = obj.get("source_sufficient")
            project["status"] = "review_posts"
            project["approved_posts"] = False
            project["updated_at"] = now_ts()
            _append_history(project, "posts_generated", f"count={len(candidate_posts)}; attempt={attempt+1}")
            return project
        except AnalyticsError as exc:
            last_error = str(exc)
            if attempt == 0:
                continue
    raise AnalyticsError("پس از یک تلاش اصلاحی خروجی معتبر تولید نشد؛ انتشار مسدود ماند. " + last_error)


def search_images(query: str, limit=5):
    """Search Wikimedia Commons and return only trusted direct media URLs."""
    try:
        r = requests.get("https://commons.wikimedia.org/w/api.php", params={
            "action": "query", "generator": "search", "gsrsearch": f'filetype:bitmap {query}',
            "gsrnamespace": 6, "gsrlimit": min(10, max(1, int(limit))), "prop": "imageinfo|info",
            "iiprop": "url|mime|size", "iiurlwidth": 1200, "inprop": "url", "format": "json",
        }, headers={"User-Agent": "RaptorV56/1.0 (Telegram content research bot)"}, timeout=20)
        r.raise_for_status()
        data = r.json()
        out, seen = [], set()
        for page in data.get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            url = info.get("thumburl") or info.get("url", "")
            parsed = urlparse(url)
            if parsed.scheme != "https" or parsed.hostname != "upload.wikimedia.org":
                continue
            if info.get("mime") not in {"image/jpeg", "image/png", "image/webp"} or url in seen:
                continue
            seen.add(url)
            # Verify that the direct media URL can be fetched and really returns an image.
            try:
                check = requests.get(url, headers={"User-Agent": "RaptorV56/1.0", "Range": "bytes=0-1023"},
                                     stream=True, timeout=8, allow_redirects=True)
                final = urlparse(check.url)
                content_type = str(check.headers.get("Content-Type", "")).lower()
                length = int(check.headers.get("Content-Length", "0") or 0)
                accessible = check.status_code in {200, 206} and content_type.startswith("image/") and final.scheme == "https" and final.hostname == "upload.wikimedia.org"
                if length > MAX_IMAGE_BYTES:
                    accessible = False
                check.close()
            except Exception:
                accessible = False
            if not accessible:
                continue
            out.append({"title": page.get("title", ""), "url": url,
                        "source_url": page.get("fullurl", "https://commons.wikimedia.org/wiki/" + quote_plus(page.get("title", ""))),
                        "mime": info.get("mime"), "size": int(info.get("size", 0) or 0),
                        "license_note": "دسترسی مستقیم فایل بررسی شد؛ مجوز بازنشر را در صفحه منبع بررسی کنید.", "verified_at": now_ts()})
            if len(out) >= limit:
                break
        return out
    except Exception as exc:
        log.warning("Wikimedia image search failed: %s", type(exc).__name__)
        return []


def attach_images(project: dict, limit_per_post=3):
    total = 0
    for post in project.get("posts", []):
        if post.get("images"):
            continue
        matches = search_images(post.get("image_query") or project.get("title", "military"), limit_per_post)
        existing = {x.get("url") for x in post.get("image_candidates", [])}
        matches = [x for x in matches if x.get("url") not in existing]
        post.setdefault("image_candidates", []).extend(matches)
        post["image_candidates"] = post["image_candidates"][:12]
        total += len(matches)
    project["updated_at"] = now_ts()
    _append_history(project, "images_searched", f"new_candidates={total}")
    return total


def validate_post_set(posts, require_images=True):
    errors, seen = [], set()
    if not isinstance(posts, list) or not 1 <= len(posts) <= MAX_POSTS:
        return ["تعداد پست باید بین ۱ تا ۶ باشد؛ معمولاً ۵ تا ۶ پست لازم است."]
    for i, post in enumerate(posts, 1):
        text = str(post.get("text", ""))
        body_norm = re.sub(r"\W+", "", text).casefold()
        if not text.strip(): errors.append(f"پست {i} خالی است.")
        if len(text) > MAX_POST_CHARS: errors.append(f"پست {i} دارای {len(text)} نویسه است؛ سقف {MAX_POST_CHARS} است.")
        if len(text.encode("utf-16-le")) // 2 > 1024: errors.append(f"پست {i} از سقف کپشن تلگرام برحسب UTF-16 عبور کرده است.")
        if require_images and not post.get("images") and not post.get("allow_no_image"): errors.append(f"پست {i} تصویر انتخاب‌شده ندارد.")
        if len(post.get("images", [])) > 10: errors.append(f"پست {i} بیش از ۱۰ تصویر دارد؛ آلبوم تلگرام حداکثر ۱۰ مورد می‌پذیرد.")
        if body_norm and body_norm in seen: errors.append(f"پست {i} تکراری است.")
        seen.add(body_norm)
        if not text.endswith(FOOTER): errors.append(f"پست {i} footer اجباری را ندارد.")
        if not post.get("title"): errors.append(f"عنوان پست {i} خالی است.")
        post["number"] = i
        post["char_count"] = len(text)
    return errors


def validate_project_posts(project: dict, require_images=True):
    errors = validate_post_set(project.get("posts", []), require_images=require_images)
    # Fewer than five is acceptable only when the model explicitly noted insufficient source material.
    if len(project.get("posts", [])) < MIN_POSTS and not project.get("source_sufficient") is False:
        errors.append("برای مقاله کامل ۵ تا ۶ پست هدف است؛ تعداد کمتر فقط با ثبت کمبود محتوای منبع پذیرفته می‌شود.")
    return errors


def snapshot_post(post: dict, action: str, actor=None):
    post.setdefault("versions", []).append({"at": now_ts(), "action": str(action)[:80],
                                             "text": post.get("text", ""), "title": post.get("title", ""),
                                             "images": copy.deepcopy(post.get("images", [])),
                                             "status": post.get("status", "draft"), "actor": actor})
    post["versions"] = post["versions"][-50:]


def edit_post(project: dict, post_id: str, text: str, title=None, actor=None):
    post = find_post(project, post_id)
    if not post: raise AnalyticsError("پست موردنظر پیدا نشد.")
    if post.get("status") in {"published", "sending", "uncertain"}:
        raise AnalyticsError("پست منتشرشده یا با نتیجه نامشخص را نمی‌توان در صف تحلیلی بازنویسی کرد؛ ابتدا نسخه/عملیات آن را بررسی کن.")
    body = _clean_body(text)
    final = _footer_text(body)
    if not body: raise AnalyticsError("متن پست خالی است.")
    if len(final) > MAX_POST_CHARS or len(final.encode("utf-16-le")) // 2 > 1024:
        raise AnalyticsError("متن با footer از محدودیت نویسه/کپشن تلگرام عبور می‌کند.")
    snapshot_post(post, "before_manual_edit", actor)
    post["text"] = final
    if title is not None: post["title"] = str(title).strip()[:140]
    post["status"] = "needs_review"
    post["scheduled_ts"] = None
    post["manual_schedule"] = False
    post["updated_at"] = now_ts()
    project["approved_posts"] = False
    _append_history(project, "post_edited", f"post={post_id}", actor)
    return post


def rewrite_post(project: dict, post_id: str, instruction: str, actor=None):
    post = find_post(project, post_id)
    if not post: raise AnalyticsError("پست موردنظر پیدا نشد.")
    if post.get("status") in {"published", "sending", "uncertain"}:
        raise AnalyticsError("پست منتشرشده یا با نتیجه نامشخص قابل بازنویسی در صف نیست.")
    prompt = (
        "فقط پست زیر را مطابق دستور مدیر بازنویسی کن. از منابع متن پروژه فراتر نرو؛ هیچ واقعیت/عدد/تاریخی نساز. "
        "متن فارسی تلگرامی، طبیعی و مستقل باشد؛ در خروجی فقط متن پست را بده، بدون توضیح داخلی و بدون footer. "
        "حداکثر ۹۰۰ نویسه پیش از footer.\nدستور مدیر: " + instruction +
        "\nعنوان: " + str(post.get("title", "")) + "\nمتن فعلی:\n" + _clean_body(post.get("text", "")) +
        "\nمتن منبع پروژه:\n" + str(project.get("article") or project.get("source_text") or "")[:18000]
    )
    response, _ = project_ai_call(project, prompt)
    candidate = _footer_text(response)
    if not candidate or len(candidate) > MAX_POST_CHARS or len(candidate.encode("utf-16-le")) // 2 > 1024:
        raise AnalyticsError("خروجی بازنویسی از اعتبارسنجی مستقل عبور نکرد؛ نسخه قبلی حفظ شد.")
    snapshot_post(post, "before_ai_rewrite:" + instruction[:60], actor)
    post["text"] = candidate
    post["status"] = "needs_review"
    post["scheduled_ts"] = None
    post["manual_schedule"] = False
    post["updated_at"] = now_ts()
    project["approved_posts"] = False
    _append_history(project, "post_rewritten", f"post={post_id}; instruction={instruction[:120]}", actor)
    return post


def find_post(project: dict, post_id: str):
    return next((x for x in project.get("posts", []) if x.get("id") == post_id), None)


def select_image(project: dict, post_id: str, image_index: int):
    post = find_post(project, post_id)
    if not post: raise AnalyticsError("پست موردنظر پیدا نشد.")
    candidates = post.get("image_candidates", [])
    if not 0 <= int(image_index) < len(candidates): raise AnalyticsError("گزینه تصویر معتبر نیست.")
    img = copy.deepcopy(candidates[int(image_index)])
    snapshot_post(post, "before_image_change")
    post["images"] = [img]
    post["updated_at"] = now_ts()
    project["updated_at"] = now_ts()
    _append_history(project, "image_selected", f"post={post_id}; index={image_index}")
    return img


def add_project_image(project: dict, image: dict):
    if not image or not (image.get("file_id") or image.get("url")):
        raise AnalyticsError("تصویر فاقد file_id یا URL معتبر است.")
    project.setdefault("images", []).append(copy.deepcopy(image))
    project["images"] = project["images"][-60:]
    _append_history(project, "project_image_added", f"count={len(project['images'])}")
    return len(project["images"])


def assign_project_image(project: dict, post_id: str, image_index: int, replace=True):
    post = find_post(project, post_id)
    images = project.get("images", [])
    if not post: raise AnalyticsError("پست موردنظر پیدا نشد.")
    if not 0 <= int(image_index) < len(images): raise AnalyticsError("تصویر گالری پیدا نشد.")
    snapshot_post(post, "before_assign_project_image")
    item = copy.deepcopy(images[int(image_index)])
    if replace: post["images"] = [item]
    else: post.setdefault("images", []).append(item)
    post["updated_at"] = now_ts()
    _append_history(project, "project_image_assigned", f"post={post_id}; image={image_index}")
    return post


def remove_post_image(project: dict, post_id: str, index=0):
    post = find_post(project, post_id)
    if not post: raise AnalyticsError("پست موردنظر پیدا نشد.")
    images = post.get("images", [])
    if not images: raise AnalyticsError("این پست تصویر انتخاب‌شده ندارد.")
    if not 0 <= int(index) < len(images): raise AnalyticsError("شماره تصویر نامعتبر است.")
    snapshot_post(post, "before_remove_image")
    images.pop(int(index))
    post["images"] = images
    post["updated_at"] = now_ts()
    return post


def reorder_post_images(project: dict, post_id: str, order):
    post = find_post(project, post_id)
    if not post: raise AnalyticsError("پست موردنظر پیدا نشد.")
    images = post.get("images", [])
    if sorted(order) != list(range(len(images))): raise AnalyticsError("ترتیب تصاویر نامعتبر است.")
    snapshot_post(post, "before_reorder_images")
    post["images"] = [images[i] for i in order]
    post["updated_at"] = now_ts()
    return post


def research_addendum(project: dict, request_text: str):
    if project.get("kind") != "research": raise AnalyticsError("این قابلیت فقط برای پروژه تحقیق Gemini است.")
    query = (request_text or project.get("topic", "")).strip()
    results = web_search(query + " official updated report technical historical", 6)
    if not results: raise AnalyticsError("جست‌وجوی تکمیلی نتیجه‌ای نداد؛ مقاله قبلی حفظ شد.")
    evidence = "\n\n".join(f"[{i+1}] {x['title']}\nURL: {x['url']}\n{x['snippet']}" for i, x in enumerate(results))
    prompt = ("مقاله موجود را با شواهد تازه زیر تکمیل کن. متن قبلی را پاک نکن مگر آنکه دقیقاً اصلاح شود. "
              "ادعاهای جدید را از تحلیل/برآورد جدا کن و چیزی نساز. فقط ARTICLE و SOURCES بده. "
              f"\nدرخواست مدیر: {request_text}\nمقاله فعلی:\n{project.get('article','')}\nشواهد جدید:\n{evidence}")
    response, raw = project_ai_call(project, prompt, use_search=(project.get("ai_provider_override") or "gemini") == "gemini")
    article = _split_article_sources(response)
    if len(article) < 100: raise AnalyticsError("تحقیق تکمیلی خروجی قابل استفاده نداشت؛ مقاله قبلی حفظ شد.")
    project.setdefault("article_versions", []).append({"at": now_ts(), "article": project.get("article", ""), "action": "before_addendum"})
    project["article_versions"] = project["article_versions"][-50:]
    project["article"] = article
    project["approved_article"] = False
    project["approved_posts"] = False
    for old_post in project.get("posts", []):
        if old_post.get("status") not in {"published", "sending", "uncertain"}:
            old_post["status"] = "needs_review"
            old_post["scheduled_ts"] = None
            old_post["manual_schedule"] = False
    existing = {x.get("url") for x in project.get("sources", [])}
    project.setdefault("sources", []).extend({"title": x["title"], "url": x["url"], "snippet": x["snippet"], "retrieved_at": now_ts(), "status": "needs_review"}
                                                for x in results if x.get("url") not in existing)
    project["approved_article"] = False
    project["status"] = "review_research"
    _append_history(project, "research_updated", request_text[:160])
    return project


def _epoch(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            return None
    return None


def news_times(state: dict, day_start: float, day_end: float):
    """Read-only extraction of fixed news times. Never edits news queue/history."""
    times = []
    for item in state.get("_pending_queue", []):
        if item.get("delivery_status") in {"uncertain", "partial", "sending", "sent"}:
            continue
        ts = _epoch(item.get("scheduled_at") or item.get("scheduled_ts") or item.get("publish_at"))
        if ts is not None and day_start <= ts <= day_end:
            times.append(ts)
    for item in state.get("_published_history", []):
        ts = _epoch(item.get("ts") or item.get("published_ts") or item.get("published_at"))
        if ts is not None and day_start <= ts <= day_end:
            times.append(ts)
    return sorted(set(times))


def forecast_news_times(state: dict, day_start: float, day_end: float, tzinfo):
    """Project the bot's configured news cadence for planning only; these are not actual news IDs.

    Exact scheduled_at/published times are always included separately. Forecast slots let the
    analytics planner reserve likely news gaps while the news engine itself remains untouched.
    A new actual news publication is checked again at runtime and can move only auto-scheduled
    analytics posts.
    """
    settings = state.get("_admin_settings", {}) if isinstance(state.get("_admin_settings"), dict) else {}
    start_hour = max(0, min(23, int(settings.get("active_start_hour", 8) or 8)))
    afternoon = max(start_hour, min(23, int(settings.get("afternoon_start_hour", 14) or 14)))
    end_hour = max(1, min(24, int(settings.get("active_end_hour", 24) or 24)))
    morning_gap = max(5, int(settings.get("morning_spacing_minutes", 60) or 60)) * 60
    afternoon_gap = max(5, int(settings.get("afternoon_spacing_minutes", 45) or 45)) * 60
    first_day = datetime.fromtimestamp(day_start, tzinfo).date()
    last_day = datetime.fromtimestamp(day_end, tzinfo).date()
    slots = []
    d = first_day
    while d <= last_day:
        begin = datetime(d.year, d.month, d.day, start_hour, 0, tzinfo=tzinfo)
        split = datetime(d.year, d.month, d.day, afternoon, 0, tzinfo=tzinfo)
        finish = datetime(d.year, d.month, d.day, 0, 0, tzinfo=tzinfo) + timedelta(days=1)
        cursor = begin
        while cursor < finish:
            if cursor.hour < start_hour or cursor.hour >= end_hour:
                break
            gap = morning_gap if cursor < split else afternoon_gap
            ts = cursor.timestamp()
            if day_start <= ts <= day_end:
                slots.append(ts)
            cursor += timedelta(seconds=gap)
        d += timedelta(days=1)
    return slots


def planning_news_times(state: dict, day_start: float, day_end: float, tzinfo):
    actual = news_times(state, day_start, day_end)
    forecasts = forecast_news_times(state, day_start, day_end, tzinfo)
    # Exact schedule is authoritative; merging equal/near times as one slot avoids fake duplicate gaps.
    merged = sorted(actual + forecasts)
    result = []
    for ts in merged:
        if not result or abs(ts - result[-1]) >= 5 * 60:
            result.append(ts)
        elif ts in actual:
            result[-1] = ts
    return result

def _other_analytics_times(a: dict, except_pid: str, day_start: float, day_end: float):
    times = []
    for other in a.get("projects", {}).values():
        if other.get("id") == except_pid:
            continue
        for post in other.get("posts", []):
            if post.get("status") in {"scheduled", "sending", "uncertain", "published"}:
                ts = _epoch(post.get("scheduled_ts") or post.get("published_at"))
                if ts is not None and day_start <= ts <= day_end:
                    times.append(ts)
    return times


def _check_candidate(ts, news, analytic, gap):
    conflicts = [x for x in list(news) + list(analytic) if abs(float(x) - float(ts)) < gap]
    return not conflicts


def propose_project_schedule(state: dict, project: dict, tzinfo):
    """Assign the earliest feasible schedule; mutate analytics records only."""
    a = ensure_state(state); settings = a["settings"]
    if settings.get("schedule_mode", "auto") != "auto":
        raise AnalyticsError("زمان‌بندی دستی فعال است.")
    if not project.get("approved_posts"):
        raise AnalyticsError("پست‌ها هنوز تأیید نهایی مدیر را نگرفته‌اند.")
    errors = validate_project_posts(project, settings.get("require_image", True))
    if errors: raise AnalyticsError("پست‌ها قابل زمان‌بندی نیستند: " + "؛ ".join(errors[:8]))
    all_posts = project.get("posts", [])
    active_posts = [p for p in all_posts if p.get("status") not in {"published", "cancelled", "rejected", "archived"}]
    if not active_posts:
        raise AnalyticsError("تمام پست‌های این پروژه قبلاً منتشر شده‌اند.")
    start_hour, end_hour = int(settings.get("start_hour", 14)), int(settings.get("end_hour", 21))
    gap = max(20, int(settings.get("min_news_gap_minutes", 20))) * 60
    now = datetime.now(tzinfo)
    scheduled_projects = set()
    for other in a["projects"].values():
        if other.get("id") == project.get("id"):
            continue
        for other_post in other.get("posts", []):
            if other_post.get("status") not in {"scheduled", "published", "sending", "uncertain"}:
                continue
            other_ts = _epoch(other_post.get("scheduled_ts") or other_post.get("published_at"))
            if other_ts:
                scheduled_projects.add(datetime.fromtimestamp(other_ts, tzinfo).date().isoformat())
    for day_offset in range(0, 366):
        day = (now + timedelta(days=day_offset)).date(); day_key = day.isoformat()
        if day_key in scheduled_projects: continue
        begin = datetime(day.year, day.month, day.day, start_hour, 0, tzinfo=tzinfo)
        end = datetime(day.year, day.month, day.day, end_hour, 0, tzinfo=tzinfo)
        cursor = max(begin, now + timedelta(minutes=1))
        if cursor >= end: continue
        day_start = datetime(day.year, day.month, day.day, 0, 0, tzinfo=tzinfo).timestamp()
        day_end = day_start + 86400
        news = planning_news_times(state, begin.timestamp() - gap, end.timestamp() + gap, tzinfo)
        other_analytic = _other_analytics_times(a, project.get("id"), day_start - gap, day_end + gap)
        published_today = []
        for post in all_posts:
            if post.get("status") != "published": continue
            published_ts = _epoch(post.get("published_at"))
            if published_ts and day_start <= published_ts <= day_end:
                published_today.append(published_ts)
        assigned = sorted(published_today)
        candidates = []
        feasible = True
        prev_analytic = assigned[-1] if assigned else None
        for index, post in enumerate(active_posts):
            candidate = cursor
            needs_news_before = prev_analytic is not None
            if needs_news_before:
                next_news = [x for x in news if x >= prev_analytic + gap and x < end.timestamp()]
                if not next_news:
                    feasible = False; break
                candidate = max(candidate, datetime.fromtimestamp(next_news[0] + gap, tz=tzinfo))
            while candidate < end:
                ts = candidate.timestamp()
                conflicts = [x for x in news + other_analytic + assigned + candidates if abs(x - ts) < gap]
                if not conflicts: break
                candidate = datetime.fromtimestamp(max(conflicts) + gap, tz=tzinfo)
            if candidate >= end:
                feasible = False; break
            ts = candidate.timestamp()
            if prev_analytic is not None and not any(prev_analytic < n < ts for n in news):
                feasible = False; break
            candidates.append(ts); prev_analytic = ts; cursor = candidate + timedelta(minutes=1)
        if feasible:
            for post, ts in zip(active_posts, candidates):
                if post.get("status") in {"sending", "uncertain"}:
                    raise AnalyticsError("پروژه دارای ارسال نامشخص است؛ زمان‌بندی خودکار متوقف شد.")
                post["scheduled_ts"] = ts; post["status"] = "scheduled"; post["manual_schedule"] = False
                post["updated_at"] = now_ts()
            project["publish_day"] = day_key; project["schedule_mode"] = "auto"
            project["status"] = "scheduled"; project["updated_at"] = now_ts()
            _append_history(project, "schedule_auto", day_key + f"; posts={len(active_posts)}")
            return candidates
    raise AnalyticsError("در ۳۶۵ روز آینده زمان خالی مطابق محدودیت‌ها پیدا نشد؛ صف تحلیلی حفظ شد و صف اخبار تغییر نکرد.")

def propose_manual_schedule(state: dict, project: dict, timestamps, tzinfo):
    """Validate all manual timestamps atomically before mutating project posts."""
    a = ensure_state(state); settings = a["settings"]
    posts = project.get("posts", [])
    if not project.get("approved_posts"): raise AnalyticsError("پست‌ها باید پیش از زمان‌بندی دستی تأیید شوند.")
    errors = validate_project_posts(project, settings.get("require_image", True))
    if errors: raise AnalyticsError("پست‌ها قابل زمان‌بندی نیستند: " + "; ".join(errors[:8]))
    if len(timestamps) != len(posts): raise AnalyticsError(f"باید دقیقاً {len(posts)} زمان وارد کنی.")
    values = [float(x) for x in timestamps]
    local = [datetime.fromtimestamp(x, tzinfo) for x in values]
    if values != sorted(values) or len(set(values)) != len(values): raise AnalyticsError("زمان‌ها باید یکتا و به ترتیب صعودی باشند.")
    start, end = int(settings.get("start_hour", 14)), int(settings.get("end_hour", 21))
    for dt in local:
        if dt.hour < start or dt.hour > end or (dt.hour == end and (dt.minute or dt.second)):
            raise AnalyticsError(f"تمام زمان‌ها باید در بازه {start:02d}:00 تا {end:02d}:00 باشند.")
    gap = max(20, int(settings.get("min_news_gap_minutes", 20))) * 60
    begin = min(datetime(x.year,x.month,x.day,0,0,tzinfo=tzinfo).timestamp() for x in local) - gap
    finish = max(datetime(x.year,x.month,x.day,23,59,tzinfo=tzinfo).timestamp() for x in local) + gap
    news = planning_news_times(state, begin, finish, tzinfo)
    other = _other_analytics_times(a, project.get("id"), begin, finish)
    if any(not _check_candidate(ts, news, other, gap) for ts in values):
        raise AnalyticsError("حداقل یکی از زمان‌ها با خبر یا پست تحلیلی دیگر کمتر از ۲۰ دقیقه فاصله دارد.")
    for i in range(1, len(values)):
        if not any(values[i-1] < n < values[i] for n in news):
            raise AnalyticsError(f"بین پست {i} و {i+1} باید حداقل یک خبر قرار بگیرد.")
    day_keys = {x.date().isoformat() for x in local}
    # The stated rule is one project per day, not one project per timestamp.
    for other_pid, other_project in a["projects"].items():
        if other_pid == project.get("id"): continue
        for op in other_project.get("posts", []):
            if op.get("status") not in {"scheduled", "sending", "uncertain", "published"}: continue
            other_ts = _epoch(op.get("scheduled_ts") or op.get("published_at"))
            if other_ts and datetime.fromtimestamp(other_ts, tzinfo).date().isoformat() in day_keys:
                raise AnalyticsError("در یکی از روزهای انتخاب‌شده پروژه تحلیلی دیگری فعال است؛ قانون یک پروژه در روز رعایت نمی‌شود.")
    # Atomic mutation only after all checks pass.
    for post, ts in zip(posts, values):
        if post.get("status") in {"published", "sending", "uncertain"}:
            raise AnalyticsError("پستی منتشر شده یا دارای نتیجه نامشخص است؛ زمان‌بندی مجدد کل پروژه مجاز نیست.")
    for post, ts in zip(posts, values):
        post["scheduled_ts"] = ts; post["status"] = "scheduled"; post["manual_schedule"] = True; post["updated_at"] = now_ts()
    project["publish_day"] = local[0].date().isoformat() if len(day_keys) == 1 else ",".join(sorted(day_keys))
    project["schedule_mode"] = "manual"; project["status"] = "scheduled"; project["updated_at"] = now_ts()
    _append_history(project, "schedule_manual", ",".join(datetime.fromtimestamp(x,tzinfo).strftime("%Y-%m-%d %H:%M") for x in values))
    return values


def manual_schedule_single(state: dict, project: dict, post: dict, ts: float, tzinfo):
    a = ensure_state(state); settings = a["settings"]
    if not project.get("approved_posts"):
        raise AnalyticsError("پست باید تأیید شده باشد.")
    if post.get("status") in {"published", "sending", "uncertain"}:
        raise AnalyticsError("وضعیت پست اجازه تغییر زمان نمی‌دهد.")
    local = datetime.fromtimestamp(float(ts), tzinfo)
    start, end = int(settings.get("start_hour", 14)), int(settings.get("end_hour", 21))
    if local.hour < start or local.hour > end or (local.hour == end and (local.minute or local.second)):
        raise AnalyticsError("زمان باید داخل بازه تحلیلی باشد.")
    gap = max(20, int(settings.get("min_news_gap_minutes", 20))) * 60
    news = planning_news_times(state, ts-gap, ts+gap, tzinfo)
    if any(abs(n-ts) < gap for n in news):
        raise AnalyticsError("زمان پیشنهادی با زمان خبر/پنجره پیش‌بینی‌شده کمتر از ۲۰ دقیقه فاصله دارد.")
    # Other projects may not occupy this local day. The selected project may have multiple posts
    # on the day, but their order, 20-minute spacing, and an intervening news slot remain mandatory.
    day_key = local.date().isoformat()
    for pid, other_project in a["projects"].items():
        if pid == project.get("id"):
            continue
        for other_post in other_project.get("posts", []):
            if other_post.get("status") not in {"scheduled", "published", "sending", "uncertain"}:
                continue
            other_ts = _epoch(other_post.get("scheduled_ts") or other_post.get("published_at"))
            if other_ts and datetime.fromtimestamp(other_ts, tzinfo).date().isoformat() == day_key:
                raise AnalyticsError("در این روز پروژه تحلیلی دیگری برنامه‌ریزی شده است.")
    related = []
    for other_post in project.get("posts", []):
        if other_post.get("id") == post.get("id") or other_post.get("status") not in {"scheduled", "published", "sending", "uncertain"}:
            continue
        other_ts = _epoch(other_post.get("scheduled_ts") or other_post.get("published_at"))
        if other_ts is None:
            continue
        if abs(other_ts - ts) < gap:
            raise AnalyticsError("زمان با پست تحلیلی دیگری از همین پروژه کمتر از ۲۰ دقیقه فاصله دارد.")
        related.append((other_ts, other_post))
    # Preserve sequence across calendar days too: post N must remain after all scheduled
    # lower-numbered posts and before all higher-numbered posts.
    current_number = int(post.get("number", 0))
    prior = [(t, p) for t, p in related if int(p.get("number", 0)) < current_number]
    later = [(t, p) for t, p in related if int(p.get("number", 0)) > current_number]
    if any(t >= ts for t, _ in prior):
        raise AnalyticsError("ترتیب انتشار پروژه رعایت نشده است؛ این پست باید بعد از تمام پست‌های قبلی باشد.")
    if any(t <= ts for t, _ in later):
        raise AnalyticsError("ترتیب انتشار پروژه رعایت نشده است؛ این پست باید قبل از تمام پست‌های بعدی باشد.")
    if prior:
        prev_ts = max(prior, key=lambda pair: int(pair[1].get("number", 0)))[0]
        interval_news = planning_news_times(state, prev_ts, ts, tzinfo)
        if not any(prev_ts < n < ts for n in interval_news):
            raise AnalyticsError("بین پست قبلی همین پروژه و زمان جدید باید یک خبر قرار بگیرد.")
    if later:
        next_ts = min(later, key=lambda pair: int(pair[1].get("number", 0)))[0]
        interval_news = planning_news_times(state, ts, next_ts, tzinfo)
        if not any(ts < n < next_ts for n in interval_news):
            raise AnalyticsError("بین زمان جدید و پست بعدی همین پروژه باید یک خبر قرار بگیرد.")
    post["scheduled_ts"] = float(ts); post["manual_schedule"] = True; post["status"] = "scheduled"; post["updated_at"] = now_ts()
    day_keys = sorted({datetime.fromtimestamp(_epoch(p.get("scheduled_ts") or p.get("published_at")), tzinfo).date().isoformat()
                       for p in project.get("posts", []) if p.get("status") in {"scheduled", "published", "sending", "uncertain"}
                       and _epoch(p.get("scheduled_ts") or p.get("published_at"))})
    project["schedule_mode"] = "manual"; project["publish_day"] = ",".join(day_keys) or day_key
    project["status"] = "scheduled"; project["updated_at"] = now_ts()
    _append_history(project, "post_rescheduled_manual", f"post={post.get('id')}; at={day_key} {local:%H:%M}")
    return post

def due_posts(state: dict, now=None):
    a = ensure_state(state); now = now or now_ts()
    if a["settings"].get("paused"): return []
    due = []
    for project in a["projects"].values():
        if project.get("archived") or project.get("paused") or project.get("status") == "cancelled": continue
        for post in project.get("posts", []):
            if post.get("status") == "scheduled" and post.get("scheduled_ts") and float(post["scheduled_ts"]) <= now:
                due.append((project, post))
    return sorted(due, key=lambda pair: pair[1].get("scheduled_ts") or 0)


def mark_sending(project: dict, post: dict):
    post["status"] = "sending"
    post.setdefault("attempts", []).append({"started_at": now_ts(), "status": "sending", "scheduled_ts": post.get("scheduled_ts")})
    project["status"] = "publishing"; project["updated_at"] = now_ts()
    _append_history(project, "post_send_started", str(post.get("id")))


def mark_uncertain(project: dict, post: dict, error: str):
    post["status"] = "uncertain"; post["last_error"] = str(error)[:500]
    if post.get("attempts"):
        post["attempts"][-1].update({"status": "uncertain", "error": str(error)[:500], "ended_at": now_ts()})
    project["status"] = "error"; project["last_error"] = str(error)[:500]; project["updated_at"] = now_ts()
    _append_history(project, "post_send_uncertain", f"post={post.get('id')}; {error[:200]}")


def mark_published(project: dict, post: dict, message_id):
    post["status"] = "published"; post["telegram_message_id"] = message_id; post["published_at"] = now_ts(); post["last_error"] = ""
    if post.get("attempts"):
        post["attempts"][-1].update({"status": "published", "message_id": message_id, "ended_at": now_ts()})
    terminal = {"published", "rejected", "cancelled", "archived"}
    project["status"] = "completed" if project.get("posts") and all(x.get("status") in terminal for x in project.get("posts", [])) else "scheduled"
    project["updated_at"] = now_ts()
    _append_history(project, "post_published", f"post={post.get('id')}; message_id={message_id}")


def render_metrics(state: dict):
    a = ensure_state(state)
    projects = list(a["projects"].values())
    posts = [p for pr in projects for p in pr.get("posts", [])]
    return {
        "projects": len(projects), "active_projects": sum(not p.get("archived") and p.get("status") not in {"completed","cancelled","rejected"} for p in projects),
        "posts": len(posts), "draft": sum(p.get("status") in {"draft","needs_review"} for p in posts),
        "scheduled": sum(p.get("status") == "scheduled" for p in posts),
        "published": sum(p.get("status") == "published" for p in posts),
        "failed_or_uncertain": sum(p.get("status") in {"failed","uncertain","error"} for p in posts),
        "images_missing": sum(not p.get("images") for p in posts if p.get("status") not in {"cancelled","archived"}),
        "paused": bool(a["settings"].get("paused")),
    }


def archive_project(project: dict):
    if any(p.get("status") in {"sending", "uncertain"} for p in project.get("posts", [])):
        raise AnalyticsError("ابتدا وضعیت ارسال‌های در حال انجام/نامشخص را بررسی کن.")
    for post in project.get("posts", []):
        if post.get("status") not in {"published", "cancelled"}:
            post["status"] = "archived"; post["scheduled_ts"] = None
    project["archived"] = True; project["status"] = "archived"; project["updated_at"] = now_ts()
    _append_history(project, "project_archived")
    return project


def cancel_project(project: dict):
    if any(p.get("status") in {"sending", "uncertain"} for p in project.get("posts", [])):
        raise AnalyticsError("پروژه دارای ارسال در حال انجام یا نتیجه نامشخص است؛ برای جلوگیری از تکرار، ابتدا بررسی دستی لازم است.")
    project["status"] = "cancelled"
    for post in project.get("posts", []):
        if post.get("status") != "published":
            post["status"] = "cancelled"; post["scheduled_ts"] = None
    project["updated_at"] = now_ts(); _append_history(project, "project_cancelled")
    return project
