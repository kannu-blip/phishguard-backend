"""
server.py - PhishGuard AI API.

    uvicorn server:app --host 0.0.0.0 --port 8000

Endpoints
  POST /scan       {url}  -> {score 0-100, verdict, reasons:[{signal, impact}]}
  POST /scan-text  {text} -> rule-based scam-phrase detector (English/Hindi/Hinglish)
  POST /report     user feedback on a wrong result (appended to reports.jsonl)
  GET  /health

How the score is built (all of it is visible in `reasons`):
  1. ML probability from the RandomForest, scaled to 0-100.
  2. SHAP explains that probability: the top 4 features are returned with their
     signed impact in percentage points (+ pushes toward phishing).
  3. Hand-set rule adjustments are added on top: brand impersonation (+25/+35)
     and recently registered domain (+20). These weights are heuristics, not learned.
"""
import asyncio
import json
import os
import re
import time
import unicodedata
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
import joblib
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from features import (
    FEATURE_NAMES, FRIENDLY_NAMES, SHORTENERS, deleet, extract_features_vector,
    is_ip, parse_url, registered_domain,
)

HERE = Path(__file__).parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", HERE / "model.joblib"))
REPORTS_PATH = Path(os.environ.get("REPORTS_PATH", HERE / "reports.jsonl"))

# Verdict thresholds; the extension uses the same numbers (40 banner, 70 block).
SUSPICIOUS_AT, DANGEROUS_AT = 40, 70


def verdict_for(score):
    return "dangerous" if score >= DANGEROUS_AT else "suspicious" if score >= SUSPICIOUS_AT else "safe"


class State:
    model = None
    explainer = None
    pos_idx = 1
    meta = {}
    http = None


STATE = State()


@asynccontextmanager
async def lifespan(app):
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        if bundle.get("feature_names") != FEATURE_NAMES:
            raise RuntimeError("model.joblib was trained with different features - retrain with train.py")
        STATE.model = bundle["model"]
        STATE.pos_idx = list(STATE.model.classes_).index(1)
        STATE.explainer = shap.TreeExplainer(STATE.model)
        STATE.meta = {k: bundle.get(k) for k in ("trained_at", "n_train", "n_test")}
    else:
        print(f"WARNING: {MODEL_PATH} not found. Run train.py first; /scan will return 503.")
    STATE.http = httpx.AsyncClient(timeout=httpx.Timeout(3.0), follow_redirects=True,
                                   headers={"Accept": "application/rdap+json, application/json"})
    yield
    await STATE.http.aclose()


app = FastAPI(title="PhishGuard AI", version="1.0.0", lifespan=lifespan)
# CORS is wide open so the extension can call from any install. Restrict
# allow_origins (and add auth) before exposing this to the internet.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ----------------------------- SHAP handling -----------------------------

def phishing_contributions(explainer, x, n_features):
    """Per-feature SHAP values toward the phishing class for a single row.

    SHAP's output shape depends on version and model: a list of per-class arrays
    (older), (samples, features, classes) (newer), (classes, samples, features),
    or a plain (samples, features) array. Normalize all of them to a 1-D array.
    """
    sv = explainer.shap_values(x)
    if isinstance(sv, list):
        arr = np.asarray(sv[1] if len(sv) > 1 else sv[0])[0]
    else:
        arr = np.asarray(sv)
        if arr.ndim == 3:
            if arr.shape[1] == n_features:            # (samples, features, classes)
                arr = arr[0, :, 1 if arr.shape[2] > 1 else 0]
            elif arr.shape[2] == n_features:          # (classes, samples, features)
                arr = arr[1 if arr.shape[0] > 1 else 0, 0, :]
        elif arr.ndim == 2:
            arr = arr[0]
    arr = np.asarray(arr).reshape(-1)
    if arr.shape[0] != n_features:
        raise ValueError(f"Unexpected SHAP output shape for {n_features} features: {arr.shape}")
    return arr


def model_part(url):
    """Blocking work (model + SHAP); run in a thread pool."""
    feature_values = dict(zip(FEATURE_NAMES, extract_features_vector(url)))
    x = np.asarray([[feature_values[name] for name in FEATURE_NAMES]], dtype=float)
    proba = float(STATE.model.predict_proba(x)[0][STATE.pos_idx])
    contrib = phishing_contributions(STATE.explainer, x, len(FEATURE_NAMES))
    top = np.argsort(-np.abs(contrib))[:4]
    reasons = []
    for i in top:
        impact = round(float(contrib[i]) * 100, 1)
        if abs(impact) < 0.1:
            continue
        feature = FEATURE_NAMES[i]
        signal = FRIENDLY_NAMES[feature]
        if feature == "is_https":
            protocol = "HTTPS" if feature_values[feature] else "HTTP"
            signal = f"HTTPS status: {protocol}"
        reasons.append({"signal": signal, "impact": impact})
    return proba, reasons


# --------------------------- brand impersonation ---------------------------
# brand token -> registered domains that legitimately belong to it.
BRANDS = {
    "paypal": {"paypal.com"},
    "google": {"google.com", "google.co.in"},
    "googlepay": {"google.com"}, "gpay": {"google.com"},
    "microsoft": {"microsoft.com", "live.com", "office.com", "microsoftonline.com", "outlook.com"},
    "apple": {"apple.com", "icloud.com"},
    "amazon": {"amazon.com", "amazon.in", "amazon.co.uk"},
    "amazonpay": {"amazon.in", "amazon.com", "amazonpay.in"},
    "facebook": {"facebook.com", "fb.com"},
    "instagram": {"instagram.com"},
    "whatsapp": {"whatsapp.com"},
    "netflix": {"netflix.com"},
    "chase": {"chase.com"},
    "wellsfargo": {"wellsfargo.com"},
    "bankofamerica": {"bankofamerica.com"},
    "citibank": {"citibank.com", "citi.com"},
    "hsbc": {"hsbc.com", "hsbc.co.in"},
    "barclays": {"barclays.co.uk", "barclays.com"},
    # Indian banks
    "sbi": {"sbi.co.in", "onlinesbi.sbi", "onlinesbi.com", "sbicard.com", "sbilife.co.in"},
    "onlinesbi": {"sbi.co.in", "onlinesbi.sbi", "onlinesbi.com"},
    "hdfcbank": {"hdfcbank.com", "hdfc.com", "hdfclife.com"},
    "icicibank": {"icicibank.com", "icici.com", "icicidirect.com"},
    "axisbank": {"axisbank.com"},
    "kotak": {"kotak.com"},
    "pnbindia": {"pnbindia.in"},
    "bankofbaroda": {"bankofbaroda.in", "bankofbaroda.com"},
    "canarabank": {"canarabank.com"},
    "unionbankofindia": {"unionbankofindia.co.in"},
    "idfcfirstbank": {"idfcfirstbank.com"},
    "yesbank": {"yesbank.in"},
    "indusind": {"indusind.com"},
    # UPI / payment apps
    "paytm": {"paytm.com", "paytmbank.com"},
    "phonepe": {"phonepe.com"},
    "bhim": {"bhimupi.org.in", "npci.org.in"},
    "npci": {"npci.org.in"},
    "mobikwik": {"mobikwik.com"},
    "freecharge": {"freecharge.in"},
    "razorpay": {"razorpay.com"},
}
# Tokens that are also ordinary words: only flag them when they appear as an exact
# hyphen/dot-separated token, never as a typo of or substring inside other words
# ("apply.com" and "purchase.com" must not look like apple/chase impersonation).
COMMON_WORD_BRANDS = {"apple", "chase"}
FUZZY_MIN_LEN = 6        # shorter brands (sbi, bhim, gpay...) only match exactly
SUBSTRING_MIN_LEN = 7    # brand hidden inside a longer label (securepaypalverify)
ALL_LEGIT = set().union(*BRANDS.values())


def edit_distance(a, b):
    """Levenshtein distance (single-row dynamic programming)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def brand_check(host):
    """Return (signal_text, points) or None."""
    if not host or is_ip(host):
        return None
    reg = registered_domain(host)
    if reg in ALL_LEGIT:
        return None
    label = reg.split(".")[0]
    tokens = {t for t in re.split(r"[.\-_]", host) if t}
    for brand in BRANDS:
        # 1) look-alike / typo of the brand in the main domain label or its parts
        if brand not in COMMON_WORD_BRANDS and len(brand) >= FUZZY_MIN_LEN:
            max_d = 1 if len(brand) < 9 else 2
            for cand in {label, *re.split(r"[\-_]", label)}:
                if cand == brand:
                    continue
                if abs(len(cand) - len(brand)) <= max_d:
                    if deleet(cand) == brand:
                        return (f"Imitates '{brand}' with look-alike characters", 35)
                    if edit_distance(cand, brand) <= max_d:
                        return (f"Looks like a misspelling of '{brand}'", 35)
        # 2) real brand name used on a domain that does not belong to the brand
        if brand in tokens or deleet_in(tokens, brand):
            return (f"Uses the name '{brand}' but is not its real website", 25)
        if brand not in COMMON_WORD_BRANDS and len(brand) >= SUBSTRING_MIN_LEN and brand in deleet(label):
            return (f"Uses the name '{brand}' but is not its real website", 25)
    return None


def deleet_in(tokens, brand):
    return any(t != brand and deleet(t) == brand for t in tokens)


# ------------------------------ domain age -------------------------------
_age_cache = {}  # registered domain -> (timestamp, days or None)


async def domain_age_days(host):
    """Days since registration via RDAP, or None. Never raises; 3 s time limit."""
    reg = registered_domain(host)
    if not reg or "." not in reg or is_ip(reg) or reg.endswith((".localhost", ".local", ".internal")):
        return None
    hit = _age_cache.get(reg)
    if hit and time.time() - hit[0] < (86400 if hit[1] is not None else 300):
        return hit[1]
    days = None
    try:
        r = await asyncio.wait_for(STATE.http.get(f"https://rdap.org/domain/{reg}"), timeout=3.0)
        if r.status_code == 200:
            for ev in r.json().get("events", []):
                if ev.get("eventAction") == "registration":
                    when = datetime.fromisoformat(ev["eventDate"].replace("Z", "+00:00"))
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=timezone.utc)
                    days = (datetime.now(timezone.utc) - when).days
                    break
    except Exception:
        days = None  # skip silently: slow/unsupported TLD, rate limit, offline...
    _age_cache[reg] = (time.time(), days)
    return days


# --------------------------------- /scan ---------------------------------
class ScanRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


_scan_cache = {}  # url -> (timestamp, result); 5 minute cache
SCAN_TTL = 300


async def scan_url(url):
    if STATE.model is None:
        raise HTTPException(503, "Model not loaded. Train one with train.py (see README).")
    hit = _scan_cache.get(url)
    if hit and time.time() - hit[0] < SCAN_TTL:
        return hit[1]

    host = parse_url(url)[2]
    (proba, reasons), age = await asyncio.gather(
        run_in_threadpool(model_part, url), domain_age_days(host)
    )

    adjust = 0.0
    brand = brand_check(host)
    if brand:
        reasons.append({"signal": "Brand impersonation", "impact": float(brand[1]), "detail": brand[0]})
        adjust += brand[1]
    if age is not None and age < 30:
        reasons.append({"signal": "Recently registered domain", "impact": 20.0,
                        "detail": f"Registered {age} day(s) ago"})
        adjust += 20

    score = int(max(0, min(100, round(proba * 100 + adjust))))
    reasons.sort(key=lambda r: -abs(r["impact"]))
    result = {
        "url": url, "host": host, "score": score, "verdict": verdict_for(score),
        "model_probability": round(proba * 100, 1), "reasons": reasons,
    }
    if len(_scan_cache) > 2000:
        _scan_cache.clear()
    _scan_cache[url] = (time.time(), result)
    return result


@app.post("/scan")
async def scan(req: ScanRequest):
    return await scan_url(req.url.strip())


# ------------------------------- /scan-text -------------------------------
# (concept, language, label, regex, weight). Within one concept only the highest
# weight counts, so "otp share karo" is not double-counted by English + Hinglish
# rules. Weights are hand-set heuristics. Devanagari patterns avoid \b because
# combining vowel signs are not "word" characters in Python's regex engine.
_F = re.IGNORECASE | re.DOTALL
TEXT_RULES = [
    ("kyc", "en", "KYC update demand", r"\bkyc\b.{0,30}\b(update|expire[sd]?|pending|verif\w*|incomplete|suspend\w*)\b|\b(update|complete|verify)\b.{0,15}\bkyc\b", 30),
    ("kyc", "hinglish", "KYC update demand", r"\bkyc\b.{0,25}\b(karo|karein|karwa\w*|karna|nahi kiya|band)\b", 30),
    ("kyc", "hi", "KYC update demand", r"(केवाईसी|केवायसी).{0,25}(अपडेट|पूरा|बंद|समाप्त|वेरिफाई|वेरीफाई)|(अपडेट|पूरा).{0,15}(केवाईसी|केवायसी)", 30),
    ("block", "en", "Account block / suspension threat", r"\b(account|a/c|card|sim|upi)\b.{0,40}\b(block(ed)?|suspend(ed)?|clos(ed|ure)|deactivat\w*|freez\w*|frozen)\b", 30),
    ("block", "hinglish", "Account block / suspension threat", r"\b(khata|khate|account|sim|number)\b.{0,30}\b(band|block|suspend|freeze)\b", 30),
    ("block", "hi", "Account block / suspension threat", r"(खाता|खाते|अकाउंट|सिम).{0,30}(बंद|ब्लॉक|सस्पेंड|फ्रीज)", 30),
    ("otp", "en", "Asks you to share OTP / PIN / CVV", r"\b(share|send|tell|give|provide|enter|reply with|forward)\b.{0,20}\b(otp|pin|cvv|password)\b", 45),
    ("otp", "hinglish", "Asks you to share OTP / PIN / CVV", r"\b(otp|pin|cvv)\b.{0,20}\b(share|bata\w*|bhej\w*|send)\b|\b(share|bata\w*|bhej\w*)\b.{0,20}\b(otp|pin|cvv)\b", 45),
    ("otp", "hi", "Asks you to share OTP / PIN / CVV", r"(ओटीपी|otp|पिन|सीवीवी).{0,20}(शेयर|बताएं|बताइए|बताओ|भेजें|भेजिए|साझा)|(शेयर|बताएं|बताइए|बताओ|भेजें).{0,20}(ओटीपी|पिन|सीवीवी)", 45),
    ("lottery", "en", "Lottery / prize claim", r"\b(lottery|lucky draw|you (have )?won|jackpot|prize money|congratulations.{0,30}(won|winner|selected))\b", 35),
    ("lottery", "hinglish", "Lottery / prize claim", r"\b(inaam|inam)\b|\b(lakh|crore)\b.{0,30}\b(jeet\w*|won|inaam|lottery)\b|\blucky winner\b", 35),
    ("lottery", "hi", "Lottery / prize claim", r"लॉटरी|इनाम|लकी ड्रॉ|(लाख|करोड़).{0,20}(जीत|इनाम)", 35),
    ("refund", "en", "Refund pending lure", r"\brefund\b.{0,30}\b(pending|process\w*|credited|initiate\w*|claim)\b|\b(pending|claim)\b.{0,20}\brefund\b", 25),
    ("refund", "hinglish", "Refund pending lure", r"\bpaise\b.{0,20}\b(wapas|refund|credit)\b", 25),
    ("refund", "hi", "Refund pending lure", r"रिफंड.{0,20}(पेंडिंग|लंबित|बाकी|प्रोसेस)|(पेंडिंग|लंबित).{0,15}रिफंड", 25),
    ("urgency", "en", "Pressure / urgency wording", r"\b(urgent(ly)?|immediately|within \d+ ?(hours?|hrs|minutes?|mins)|last (chance|warning)|act now|expires? (today|soon))\b", 15),
    ("urgency", "hinglish", "Pressure / urgency wording", r"\b(turant|jaldi|foran)\b|\babhi\s+(click|update|verify|karo)", 15),
    ("urgency", "hi", "Pressure / urgency wording", r"तुरंत|जल्दी|फौरन|अभी.{0,10}(क्लिक|अपडेट|करें)", 15),
    ("link", "en", "Tells you to click a link", r"\b(click|tap|open)\b.{0,20}\b(link|below|here)\b", 10),
    ("link", "hinglish", "Tells you to click a link", r"\blink\s+(par|pe)\b.{0,20}\bclick\b|\bclick karo\b", 10),
    ("link", "hi", "Tells you to click a link", r"लिंक.{0,15}(क्लिक|खोलें|दबाएं)|क्लिक.{0,15}लिंक", 10),
    ("pan", "en", "PAN / Aadhaar update threat", r"\b(pan|aadhaar|aadhar)\b.{0,30}\b(link|update|expire[sd]?|block(ed)?)\b", 25),
    ("power", "en", "Electricity disconnection threat", r"\b(electricity|power)\b.{0,30}\b(disconnect\w*|cut)\b", 25),
]
_COMPILED = [(c, lang, label, re.compile(p, _F), w) for c, lang, label, p, w in TEXT_RULES]

_SHORT_PAT = "|".join(re.escape(s) for s in sorted(SHORTENERS))
_LINK_RE = re.compile(
    rf"(?:https?://[^\s<>\"')\]]+|www\.[^\s<>\"')\]]+|\b(?:{_SHORT_PAT})/[^\s<>\"')\]]+)", re.IGNORECASE
)


def extract_links(text):
    """URLs in free text. Bare domains are only caught for www.* and known shorteners."""
    out = []
    for m in _LINK_RE.findall(text):
        m = m.rstrip(".,;:!?")
        if not re.match(r"https?://", m, re.IGNORECASE):
            m = "http://" + m
        if m not in out:
            out.append(m)
    return out


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.post("/scan-text")
async def scan_text(req: TextRequest):
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFC", req.text)).strip()
    best = {}  # concept -> match with the highest weight
    for concept, lang, label, rx, weight in _COMPILED:
        m = rx.search(text)
        if m and (concept not in best or weight > best[concept]["weight"]):
            best[concept] = {"signal": label, "language": lang, "weight": weight, "excerpt": m.group(0)[:80]}
    matches = sorted(best.values(), key=lambda x: -x["weight"])
    text_score = min(100, sum(m["weight"] for m in matches))

    async def safe_scan(u):
        try:
            return await scan_url(u)
        except HTTPException as e:
            return {"url": u, "error": e.detail}

    links = await asyncio.gather(*(safe_scan(u) for u in extract_links(text)[:5]))
    link_scores = [l["score"] for l in links if "score" in l]
    score = max([text_score, *link_scores])
    return {
        "score": score,
        "verdict": verdict_for(score),
        "text_score": text_score,
        "matches": matches,
        "languages_detected": sorted({m["language"] for m in matches}),
        "links": links,
        "note": "Keyword rules only. No match does not mean the message is safe.",
    }


# --------------------------------- /report --------------------------------
class ReportRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)
    user_says: str = Field(pattern="^(actually_safe|actually_phishing)$")
    score: int | None = None
    verdict: str | None = Field(default=None, max_length=20)


@app.post("/report")
async def report(req: ReportRequest):
    """Append feedback to reports.jsonl. It is NOT used for training automatically;
    review it and add confirmed URLs to your training CSV yourself."""
    row = req.model_dump() | {"ts": datetime.now(timezone.utc).isoformat()}
    with open(REPORTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return {"ok": True}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": STATE.model is not None, **STATE.meta}
