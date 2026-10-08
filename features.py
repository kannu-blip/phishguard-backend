"""
features.py - URL feature extraction for PhishGuard AI.

IMPORTANT: this is the ONLY place where features are computed. train.py,
evaluate.py and server.py all call extract_features_vector(), so training and
serving cannot drift apart. If you change anything in here, retrain
model.joblib - an old model will silently misread the new feature order.
"""
import math
import re
from collections import Counter
from urllib.parse import urlparse

# Common public URL shorteners (a shortened link hides its real destination).
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy", "bit.do",
    "s.id", "t.ly", "lnkd.in", "v.gd", "clck.ru", "tr.im", "u.to",
}

# Words that show up disproportionately in phishing URLs. Many appear in
# legitimate URLs too ("login", "account"), so this is only a hint for the
# model, never a verdict on its own.
SUSPICIOUS_KEYWORDS = (
    "login", "log-in", "signin", "sign-in", "verify", "verification", "update",
    "secure", "security", "account", "banking", "confirm", "password",
    "passwd", "wallet", "kyc", "otp", "suspend", "unlock", "limited", "bonus",
    "claim", "refund", "payment", "invoice", "webscr", "authenticate",
    "recover", "alert", "upi", "reward", "lottery", "prize",
)

# Second-level labels used under country TLDs (co.in, com.au, ...). This is a
# small heuristic instead of the full Public Suffix List so that the backend
# has no network dependency at feature time.
_SECOND_LEVEL = {"co", "com", "org", "net", "gov", "edu", "ac", "nic", "res", "ltd", "plc", "mil"}

_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

# Order matters: this is the column order of the model's input.
FEATURE_NAMES = [
    "url_length", "host_length", "path_length", "query_length",
    "digit_count", "digit_ratio", "digit_count_host", "special_char_count",
    "dot_count", "hyphen_count_host", "at_symbol", "subdomain_count",
    "has_ip", "is_shortener", "keyword_count", "char_substitution",
    "entropy_url", "entropy_host", "is_https", "has_punycode",
    "non_standard_port", "path_depth",
]

# Plain-language names shown to the user for each feature.
FRIENDLY_NAMES = {
    "url_length": "Overall URL length",
    "host_length": "Length of the website address",
    "path_length": "Length of the page path",
    "query_length": "Length of the query string",
    "digit_count": "Number of digits in the URL",
    "digit_ratio": "Share of digits in the URL",
    "digit_count_host": "Digits in the website address",
    "special_char_count": "Unusual symbols in the URL",
    "dot_count": "Number of dots in the URL",
    "hyphen_count_host": "Hyphens in the website address",
    "at_symbol": "'@' symbol in the URL",
    "subdomain_count": "Number of subdomains",
    "has_ip": "Raw IP address instead of a name",
    "is_shortener": "Link-shortener service",
    "keyword_count": "Suspicious words (login, verify, secure...)",
    "char_substitution": "Look-alike character swaps (1/l, 0/o)",
    "entropy_url": "Randomness of the URL",
    "entropy_host": "Randomness of the website address",
    "is_https": "Uses HTTPS encryption",
    "has_punycode": "Disguised international characters",
    "non_standard_port": "Unusual network port",
    "path_depth": "Depth of the page path",
}


def parse_url(url):
    """Return (normalized_url, ParseResult, lowercase hostname).

    Browsers always give a scheme, but pasted text may not, so add one.
    Malformed URLs return an empty hostname rather than raising.
    """
    url = (url or "").strip()
    if "://" not in url:
        url = "http://" + url
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        parsed, host = urlparse("http://"), ""
    return url, parsed, host


def is_ip(host):
    """True for IPv4, IPv6 literals and the decimal/hex IPv4 tricks."""
    if not host:
        return False
    if ":" in host:
        return True
    return bool(
        _IPV4.match(host)
        or re.fullmatch(r"\d{8,10}", host)
        or re.fullmatch(r"0x[0-9a-f]+", host)
    )


def registered_domain(host):
    """Heuristic registrable domain: 'a.b.example.co.in' -> 'example.co.in'."""
    if not host or is_ip(host):
        return host
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    if len(labels[-1]) == 2 and labels[-2] in _SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def subdomain_count(host):
    """Number of subdomain labels, ignoring a leading 'www'."""
    if not host or is_ip(host):
        return 0
    reg = registered_domain(host)
    sub = host[: -len(reg)].rstrip(".") if host.endswith(reg) else ""
    labels = [p for p in sub.split(".") if p]
    if labels and labels[0] == "www":
        labels = labels[1:]
    return len(labels)


def shannon_entropy(text):
    """Shannon entropy in bits per character (0 for empty text)."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


# Digit -> letter look-alikes used to imitate brand names (paypa1, g00gle).
_LEET = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s"}


def deleet(text):
    """Undo look-alike swaps so 'paypa1' compares equal to 'paypal'."""
    out = "".join(_LEET.get(ch, ch) for ch in text)
    return out.replace("vv", "w").replace("rn", "m")


def substitution_count(host):
    """Count digits that sit next to letters inside host labels ('g00gle',
    'paypa1') plus 'vv' pairs (fake 'w'). Pure-number labels do not count."""
    count = 0
    for label in re.split(r"[.\-]", host):
        if sum(ch.isalpha() for ch in label) < 3:
            continue
        for i, ch in enumerate(label):
            if ch in _LEET:
                left = i > 0 and label[i - 1].isalpha()
                right = i + 1 < len(label) and label[i + 1].isalpha()
                if left or right:
                    count += 1
        count += label.count("vv")
    return count


def extract_features(url):
    """Return an ordered dict {feature_name: value} for one URL."""
    norm, parsed, host = parse_url(url)
    raw = (url or "").strip()
    low = raw.lower()
    path = parsed.path or ""
    query = parsed.query or ""
    digits = sum(ch.isdigit() for ch in raw)
    try:
        port = parsed.port
    except ValueError:
        port = None

    feats = {
        "url_length": len(raw),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "digit_count": digits,
        "digit_ratio": digits / len(raw) if raw else 0.0,
        "digit_count_host": sum(ch.isdigit() for ch in host),
        # symbols other than the structural ones every URL has
        "special_char_count": sum(1 for ch in raw if not ch.isalnum() and ch not in "./:"),
        "dot_count": raw.count("."),
        "hyphen_count_host": host.count("-"),
        "at_symbol": int("@" in raw),
        "subdomain_count": subdomain_count(host),
        "has_ip": int(is_ip(host)),
        "is_shortener": int(registered_domain(host) in SHORTENERS),
        "keyword_count": sum(1 for k in SUSPICIOUS_KEYWORDS if k in low),
        "char_substitution": substitution_count(host),
        "entropy_url": shannon_entropy(low),
        "entropy_host": shannon_entropy(host),
        "is_https": int(parsed.scheme == "https"),
        "has_punycode": int("xn--" in host),
        "non_standard_port": int(port not in (None, 80, 443)),
        "path_depth": len([p for p in path.split("/") if p]),
    }
    assert list(feats) == FEATURE_NAMES, "FEATURE_NAMES is out of sync with extract_features"
    return feats


def extract_features_vector(url):
    """Feature values as a plain list, in FEATURE_NAMES order."""
    return [float(v) for v in extract_features(url).values()]
