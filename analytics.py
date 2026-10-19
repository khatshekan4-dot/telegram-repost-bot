"""Independent Raptor V5.5 analytical-content subsystem.

The subsystem stores data under state['_analytics'] and never mutates the news queue.
It intentionally requires explicit human approval before a project can be scheduled.
"""
from __future__ import annotations
import os, re, json, time, uuid, html, hashlib, logging
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

class AnalyticsError(RuntimeError): pass


def ensure_state(state):
    """Add the V5.5 namespace without altering any existing news fields."""
    with state_lock_context():
        a = state.get("_analytics")
        if not isinstance(a, dict):
            a = {}
            state["_analytics"] = a
        a.setdefault("schema_version", 1)
        a.setdefault("projects", {})
        a.setdefault("settings", {"timezone": os.getenv("ANALYTICS_TIMEZONE", DEFAULT_TZ),
                                   "schedule_mode": "auto", "paused": False,
                                   "start_hour": 14, "end_hour": 21,
                                   "min_news_gap_minutes": 20,
                                   "require_image": True,
                                   "target_chat_id": os.getenv("TARGET_CHAT_ID", "").strip()})
        a.setdefault("history", [])
        return a

# Kept as a no-op context to avoid coupling this module to bot.py's lock implementation.
class state_lock_context:
    def __enter__(self): return self
    def __exit__(self, *args): return False


def now_ts(): return time.time()

def new_project(state, kind, source_text, owner_id):
    a = ensure_state(state)
    if kind not in {"research", "source_text"}: raise AnalyticsError("نوع پروژه نامعتبر است.")
    text = (source_text or "").strip()
    if not text: raise AnalyticsError("موضوع یا متن ورودی خالی است.")
    if len(text) > 60000: raise AnalyticsError("متن ورودی بیش از حد طولانی است (حداکثر ۶۰۰۰۰ نویسه).")
    pid = uuid.uuid4().hex[:12]
    title = text.splitlines()[0][:100].strip()
    project = {"id": pid, "title": title, "kind": kind, "source_text": text if kind == "source_text" else "",
               "topic": text if kind == "research" else "", "article": "", "sources": [], "images": [],
               "posts": [], "status": "queued", "owner_id": int(owner_id), "created_at": now_ts(),
               "updated_at": now_ts(), "errors": [], "versions": [], "approved_article": False,
               "approved_posts": False, "next_post_index": 0, "publish_day": "", "last_error": ""}
    a["projects"][pid] = project
    return project


def _extract_text(payload):
    try:
        return "".join(part.get("text", "") for cand in payload.get("candidates", [])[:1]
                       for part in cand.get("content", {}).get("parts", []) if isinstance(part, dict)).strip()
    except Exception: return ""


def _redact(text, keys=()):
    value=str(text or "")
    for secret in keys:
        if secret: value=value.replace(secret,"[REDACTED]")
    value=re.sub(r"(?i)(key=)[^&\s]+",r"\1[REDACTED]",value)
    return value[:500]


def gemini_call(prompt, *, model=None, use_search=False, timeout=90):
    keys = [x.strip() for x in os.getenv("GEMINI_API_KEYS", "").split(",") if x.strip()]
    if not keys: raise AnalyticsError("برای زیرسیستم تحلیلی، GEMINI_API_KEYS تنظیم نشده است؛ هیچ جایگزین خودکاری اجرا نشد.")
    model = model or os.getenv("ANALYTICS_GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{quote_plus(model)}:generateContent"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.25, "maxOutputTokens": 8192}}
    if use_search: body["tools"] = [{"google_search": {}}]
    errors=[]
    for key in keys:
        try:
            r=requests.post(url, params={"key": key}, json=body, timeout=timeout)
            if r.status_code >= 400:
                # Do not log the URL or API key.
                errors.append(f"HTTP {r.status_code}: {r.text[:300]}")
                continue
            payload=r.json(); result=_extract_text(payload)
            if not result: errors.append("پاسخ Gemini فاقد متن بود."); continue
            return result, payload
        except Exception as e:
            errors.append(f"{type(e).__name__}: {_redact(e, keys)}")
    raise AnalyticsError("Gemini در تمام کلیدهای تنظیم‌شده ناموفق بود. عملیات متوقف شد؛ از پنل درخواست تلاش مجدد یا انتخاب جایگزین کنید. " + " | ".join(errors[-3:]))


def web_search(query, limit=8):
    """Real web search through DuckDuckGo HTML; results are evidence, not AI output."""
    q=(query or "").strip()
    if not q: return []
    try:
        r=requests.get("https://html.duckduckgo.com/html/", params={"q": q},
                       headers={"User-Agent":"Mozilla/5.0 (compatible; RaptorV55/1.0)"}, timeout=20)
        r.raise_for_status()
        from bs4 import BeautifulSoup
        soup=BeautifulSoup(r.text,"html.parser")
        results=[]
        for item in soup.select(".result")[:limit]:
            link=item.select_one("a.result__a"); snippet=item.select_one(".result__snippet")
            if not link: continue
            href=link.get("href","")
            if not href.startswith("http"): continue
            results.append({"title":link.get_text(" ",strip=True),"url":href,
                            "snippet":snippet.get_text(" ",strip=True) if snippet else ""})
        return results
    except Exception as e:
        raise AnalyticsError(f"جست‌وجوی وب ناموفق بود؛ تحقیق به‌عنوان تحقیق اینترنتی ادامه داده نشد. {type(e).__name__}: {str(e)[:180]}")


def research_project(project):
    topic=project["topic"]
    results=web_search(topic + " official technical history defense research", 8)
    if not results: raise AnalyticsError("جست‌وجوی وب نتیجه قابل استفاده‌ای نداد؛ مقاله تولید نشد.")
    sources=[{"title":x["title"],"url":x["url"],"snippet":x["snippet"],"checked_at":now_ts()} for x in results]
    evidence="\n\n".join(f"[{i+1}] {s['title']}\nURL: {s['url']}\nخلاصه نتیجه: {s['snippet']}" for i,s in enumerate(sources))
    prompt=("نقش: پژوهشگر و نویسنده فارسی‌زبان. فقط از شواهد زیر و دانش عمومی با احتیاط استفاده کن. "
            "هیچ عدد، تاریخ، پیوند یا منبعی را جعل نکن. بین واقعیت تأییدشده، گزارش مستقل‌تأییدنشده، برآورد، اختلاف منابع و اطلاعات ناکافی تمایز بگذار. "
            "متن نهایی برای مدیر است و باید منابع قابل‌بررسی را در بخش جداگانه با شماره [n] ارجاع دهد؛ این ارجاعات بعداً از متن کانال حذف می‌شوند. "
            "اگر نتایج وب برای ادعایی کافی نیست، صریحاً بگو. مشخصات نسخه‌های متفاوت تجهیزات را مخلوط نکن. "
            "خروجی را با دو بخش ARTICLE و SOURCES به فارسی ارائه کن؛ مقاله منسجم، کامل، دقیق و مناسب تقسیم به چند پست باشد.\n\n"
            f"موضوع: {topic}\n\nنتایج واقعی جست‌وجوی وب:\n{evidence}")
    article, raw=gemini_call(prompt, use_search=True)
    project["article"]=article
    project["sources"]=sources
    project["research_grounding"]={"search_results":len(sources),"gemini_grounding_metadata":bool(raw.get("candidates",[{}])[0].get("groundingMetadata")),"completed_at":now_ts()}
    project["status"]="review_research"
    project["updated_at"]=now_ts()
    return project


def _json_from_model(text):
    t=(text or "").strip()
    t=re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    start=t.find("{"); end=t.rfind("}")
    if start<0 or end<start: raise AnalyticsError("خروجی مدل JSON معتبر ندارد.")
    try: return json.loads(t[start:end+1])
    except Exception as e: raise AnalyticsError(f"JSON خروجی مدل نامعتبر است: {e}")


def split_into_posts(project, count=None, tone="تحلیلی، طبیعی و گفت‌وگومحور"):
    source=(project.get("article") or project.get("source_text") or "").strip()
    if not source: raise AnalyticsError("مقاله یا متن ورودی موجود نیست.")
    count=max(5,min(6,int(count or 6)))
    prompt=("متن زیر را به مجموعه پست‌های فارسی تلگرام تبدیل کن. خروجی فقط JSON معتبر با ساختار "
            '{"posts":[{"title":"...","text":"...","image_query":"..."}]} باشد. '
            f"تعداد هدف {count} پست است؛ اگر متن برای ۵ پست اطلاعات کافی ندارد، کمتر تولید کن و در فیلد note دلیل را بنویس. "
            "هر متن پست پیش از افزودن footer حداکثر ۹۳۰ نویسه باشد تا footer نیز در سقف ۱۰۲۰ بگنجد. "
            "برای هر پست یک image_query دقیق و واقعی پیشنهاد بده. هیچ واقعیت جدیدی اضافه نکن؛ متن ورودی را وفادارانه بازنویسی کن. "
            "پست اول شروع جذاب، معرفی و اهمیت موضوع داشته باشد؛ میانی‌ها هرکدام یک محور؛ پایانی نتیجه‌گیری و پاسخ روشن بده. "
            "از تکرار عبارت‌ها و ارجاع مبهم پرهیز کن. در صورت وجود ارجاع‌های [n] آن‌ها را در متن قابل انتشار نیاور. "
            f"لحن: {tone}.\n\nمتن منبع:\n{source}")
    generated,_=gemini_call(prompt)
    obj=_json_from_model(generated)
    posts=obj.get("posts")
    if not isinstance(posts,list): raise AnalyticsError("فیلد posts در خروجی وجود ندارد.")
    valid=[]
    for i,item in enumerate(posts[:MAX_POSTS],1):
        text=re.sub(r"\[(?:\d+)\]", "", str(item.get("text", ""))).strip()
        text=re.sub(r"\n{3,}","\n\n",text)
        text=text+FOOTER
        if not text.strip(): continue
        if len(text)>MAX_POST_CHARS:
            # Safe deterministic shortening at sentence boundary; mark for review if still too long.
            body=text[:-len(FOOTER)].strip()
            limit=MAX_POST_CHARS-len(FOOTER)
            body=body[:limit]
            cut=max(body.rfind(". "),body.rfind("؟ "),body.rfind("! "),body.rfind("؛ "))
            if cut>limit*0.65: body=body[:cut+1]
            text=body.rstrip()+FOOTER
        valid.append({"id":uuid.uuid4().hex[:12],"number":i,"title":str(item.get("title",f"بخش {i}"))[:140],
                      "text":text,"char_count":len(text),"image_query":str(item.get("image_query", project.get("title","military equipment")))[:240],
                      "images":[],"status":"draft","scheduled_ts":None,"telegram_message_id":None,
                      "created_at":now_ts(),"updated_at":now_ts(),"attempts":[],"versions":[]})
    if len(valid)<1: raise AnalyticsError("هیچ پست قابل استفاده‌ای تولید نشد.")
    # Validate and keep content in review state; fewer than five is allowed only when source is insufficient.
    for post in valid:
        if len(post["text"])>MAX_POST_CHARS: raise AnalyticsError("اعتبارسنجی مستقل سقف ۱۰۲۰ نویسه را رد کرد.")
    project["posts"]=valid; project["status"]="review_posts"; project["updated_at"]=now_ts()
    return project


def search_images(query, limit=5):
    """Find real media on Wikimedia Commons and validate direct file URLs."""
    try:
        r=requests.get("https://commons.wikimedia.org/w/api.php",params={"action":"query","generator":"search",
          "gsrsearch":f'filetype:bitmap {query}',"gsrnamespace":6,"gsrlimit":min(10,limit),"prop":"imageinfo|info",
          "iiprop":"url|mime|size","iiurlwidth":1200,"inprop":"url","format":"json"},
          headers={"User-Agent":"RaptorV55/1.0 (Telegram content research bot)"},timeout=20)
        r.raise_for_status(); data=r.json(); out=[]
        for page in data.get("query",{}).get("pages",{}).values():
            info=(page.get("imageinfo") or [{}])[0]; url=info.get("thumburl") or info.get("url","")
            parsed=urlparse(url)
            if parsed.scheme!="https" or parsed.hostname not in {"upload.wikimedia.org"}: continue
            if info.get("mime") not in {"image/jpeg","image/png","image/webp"}: continue
            out.append({"title":page.get("title",""),"url":url,"source_url":page.get("fullurl", "https://commons.wikimedia.org/wiki/"+quote_plus(page.get("title",""))),
                        "mime":info.get("mime"),"size":info.get("size",0)})
            if len(out)>=limit: break
        return out
    except Exception as e:
        log.warning("Wikimedia image search failed: %s", type(e).__name__)
        return []


def attach_images(project, limit_per_post=3):
    total=0
    for post in project.get("posts",[]):
        if post.get("images"): continue
        matches=search_images(post.get("image_query") or project.get("title","military"),limit_per_post)
        # Store choices for admin review; don't auto-publish or silently assert relevance.
        post["image_candidates"]=matches
        if matches: total+=len(matches)
    project["updated_at"]=now_ts()
    return total


def validate_project_posts(project, require_images=True):
    posts=project.get("posts",[])
    errors=[]
    if not 1<=len(posts)<=MAX_POSTS: errors.append("تعداد پست باید بین ۱ تا ۶ باشد؛ برای مقاله کامل هدف ۵ تا ۶ است.")
    seen=set()
    for i,p in enumerate(posts,1):
        text=str(p.get("text", "")); norm=re.sub(r"\W+","",text).casefold()
        if not text.strip(): errors.append(f"پست {i} خالی است.")
        if len(text)>MAX_POST_CHARS: errors.append(f"پست {i} دارای {len(text)} نویسه است و از سقف ۱۰۲۰ عبور کرده.")
        if len(text.encode("utf-16-le"))//2 > 1024: errors.append(f"پست {i} برای محدودیت واقعی کپشن تلگرام بیش از ۱۰۲۴ واحد UTF-16 دارد.")
        if not p.get("images") and require_images: errors.append(f"پست {i} تصویر انتخاب‌شده ندارد.")
        if norm in seen: errors.append(f"پست {i} تکراری است.")
        seen.add(norm)
        if not text.endswith(FOOTER): errors.append(f"پست {i} فاقد footer اجباری است.")
        p["char_count"]=len(text)
    return errors


def news_times(state, day_start, day_end):
    times=[]
    # Read-only inspection of news data; this module never writes to the news queue.
    for item in state.get("_pending_queue",[]):
        if item.get("delivery_status") in {"uncertain","partial","sending"}: continue
        ts=item.get("scheduled_ts") or item.get("publish_at") or item.get("scheduled_at")
        if isinstance(ts,(int,float)) and day_start<=float(ts)<=day_end: times.append(float(ts))
    for item in state.get("_published_history",[]):
        ts=item.get("ts") or item.get("published_ts")
        if isinstance(ts,(int,float)) and day_start<=float(ts)<=day_end: times.append(float(ts))
    return sorted(set(times))


def propose_project_schedule(state, project, tzinfo):
    """Schedule one approved project per local day; only analytics records are changed."""
    a=ensure_state(state); settings=a["settings"]
    if settings.get("schedule_mode","auto")!="auto": raise AnalyticsError("زمان‌بندی دستی فعال است.")
    posts=project.get("posts",[])
    errors=validate_project_posts(project, settings.get("require_image",True))
    if errors: raise AnalyticsError("پست‌ها قابل زمان‌بندی نیستند: " + "؛ ".join(errors[:8]))
    # project-level approval is separate from article review.
    if not project.get("approved_posts"): raise AnalyticsError("پست‌ها هنوز تأیید نهایی مدیر را نگرفته‌اند.")
    from datetime import datetime, time as dtime
    now=datetime.now(tzinfo); start_hour=int(settings.get("start_hour",14)); end_hour=int(settings.get("end_hour",21))
    gap=int(settings.get("min_news_gap_minutes",20))*60
    # Keep one project/article per day; existing assigned project blocks the day.
    scheduled_projects={}
    for other in a["projects"].values():
        if other.get("id")!=project.get("id") and other.get("publish_day") and any(p.get("status") in {"scheduled","published","sending","uncertain"} for p in other.get("posts",[])):
            scheduled_projects[other["publish_day"]]=True
    for day_offset in range(0,366):
        day=(now+timedelta(days=day_offset)).date(); day_key=day.isoformat()
        if day_key in scheduled_projects: continue
        begin=datetime.combine(day,dtime(start_hour,0),tzinfo=tzinfo)
        end=datetime.combine(day,dtime(end_hour,0),tzinfo=tzinfo)
        cursor=max(begin,now+timedelta(minutes=1))
        news=news_times(state,begin.timestamp()-gap,end.timestamp()+gap)
        assigned=[]; ok=True
        for post_index, post in enumerate(posts):
            candidate=cursor
            # Every post after the first must have at least one scheduled/published news item
            # between it and the preceding analytical post.
            if post_index > 0:
                intervening=[x for x in news if x >= assigned[-1] + gap]
                if not intervening:
                    ok=False; break
                candidate=datetime.fromtimestamp(intervening[0]+gap,tz=tzinfo)
            while candidate < end:
                ts=candidate.timestamp()
                conflicts=[x for x in news+assigned if abs(x-ts)<gap]
                if not conflicts: break
                # After a conflict, move beyond it; next iteration still checks the news-gap rule.
                candidate=datetime.fromtimestamp(max(conflicts)+gap,tz=tzinfo)
            if candidate>=end: ok=False; break
            assigned.append(candidate.timestamp())
            post["scheduled_ts"]=candidate.timestamp(); post["status"]="scheduled"
            cursor=candidate+timedelta(minutes=1)
        if ok:
            project["publish_day"]=day_key; project["status"]="scheduled"; project["updated_at"]=now_ts()
            return [p["scheduled_ts"] for p in posts]
        for post in posts:
            if post.get("status")=="scheduled": post["status"]="approved"; post["scheduled_ts"]=None
    raise AnalyticsError("در ۳۶۵ روز آینده زمان خالی مطابق محدودیت‌ها پیدا نشد؛ صف تحلیل حفظ شد و اخبار تغییر نکردند.")


def due_posts(state, now=None):
    a=ensure_state(state); now=now or now_ts()
    if a["settings"].get("paused"): return []
    due=[]
    for project in a["projects"].values():
        for post in project.get("posts",[]):
            if post.get("status")=="scheduled" and post.get("scheduled_ts") and post["scheduled_ts"]<=now:
                due.append((project,post))
    return sorted(due,key=lambda pair:pair[1].get("scheduled_ts") or 0)


def mark_sending(project, post):
    post["status"]="sending"; post["attempts"].append({"started_at":now_ts(),"status":"sending"})
    project["status"]="publishing"; project["updated_at"]=now_ts()


def mark_uncertain(project, post, error):
    post["status"]="uncertain"; post["last_error"]=str(error)[:500]
    if post.get("attempts"): post["attempts"][-1].update({"status":"uncertain","error":str(error)[:500],"ended_at":now_ts()})
    project["status"]="error"; project["last_error"]=str(error)[:500]; project["updated_at"]=now_ts()


def mark_published(project, post, message_id):
    post["status"]="published"; post["telegram_message_id"]=message_id; post["published_at"]=now_ts(); post["last_error"]=""
    if post.get("attempts"): post["attempts"][-1].update({"status":"published","message_id":message_id,"ended_at":now_ts()})
    if all(x.get("status")=="published" for x in project.get("posts",[])): project["status"]="completed"
    project["updated_at"]=now_ts()


def propose_manual_schedule(state, project, timestamps, tzinfo):
    """Validate a complete manual schedule before mutating any post."""
    a=ensure_state(state); settings=a["settings"]
    posts=project.get("posts",[])
    if not project.get("approved_posts"): raise AnalyticsError("پست‌ها باید پیش از زمان‌بندی دستی تأیید شوند.")
    errors=validate_project_posts(project, settings.get("require_image",True))
    if errors: raise AnalyticsError("پست‌ها قابل زمان‌بندی نیستند: "+"؛ ".join(errors[:8]))
    if len(timestamps)!=len(posts): raise AnalyticsError(f"باید دقیقاً {len(posts)} زمان وارد کنی.")
    from datetime import datetime
    local=[datetime.fromtimestamp(float(x),tzinfo) for x in timestamps]
    day=local[0].date()
    if any(x.date()!=day for x in local): raise AnalyticsError("تمام پست‌های یک پروژه باید در یک روز قرار بگیرند.")
    start=int(settings.get("start_hour",14)); end=int(settings.get("end_hour",21))
    if any(x.hour<start or (x.hour==end and x.minute>0) or x.hour>end for x in local): raise AnalyticsError(f"تمام زمان‌ها باید در بازه {start:02d}:00 تا {end:02d}:00 باشند.")
    if local!=sorted(local) or len(set(timestamps))!=len(timestamps): raise AnalyticsError("زمان‌ها باید یکتا و به ترتیب صعودی باشند.")
    gap=int(settings.get("min_news_gap_minutes",20))*60
    begin=datetime.combine(day,datetime.min.time(),tzinfo=tzinfo).timestamp()
    finish=begin+86400
    news=news_times(state,begin-gap,finish+gap)
    if any(abs(float(t)-n)<gap for t in timestamps for n in news): raise AnalyticsError("حداقل یکی از زمان‌ها کمتر از فاصله مجاز با خبرهاست.")
    for i in range(1,len(timestamps)):
        if not any(float(timestamps[i-1])<n<float(timestamps[i]) for n in news):
            raise AnalyticsError(f"بین پست {i} و {i+1} باید دست‌کم یک پست خبری ثبت‌شده/زمان‌بندی‌شده باشد.")
    day_key=day.isoformat()
    for other in a["projects"].values():
        if other.get("id")!=project.get("id") and other.get("publish_day")==day_key and any(p.get("status") in {"scheduled","published","sending","uncertain"} for p in other.get("posts",[])):
            raise AnalyticsError("در این روز پروژه تحلیلی دیگری برنامه‌ریزی شده است؛ هر روز فقط یک مقاله.")
    for post,ts in zip(posts,timestamps):
        post["scheduled_ts"]=float(ts); post["status"]="scheduled"; post["updated_at"]=now_ts()
    project["publish_day"]=day_key; project["status"]="scheduled"; project["updated_at"]=now_ts()
    return list(timestamps)
