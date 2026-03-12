# -*- coding: utf-8 -*-
"""
webcrawler.py

Module principal du crawler web.
Crawl récursif en largeur (BFS) avec workers asynchrones parallèles.

Pipeline par page :
    HTML → extraction contenu principal (trafilatura)
         → déduplication (SHA-256)
         → filtre langue (langdetect)
         → filtre qualité (ratio lettres, longueur)
         → sauvegarde SQLite + export JSONL

    PDF  → téléchargement → pdfminer → même pipeline qualité

Modes :
    stay_on_domain=True  → crawl intra-site classique (défaut)
    stay_on_domain=False → crawl global "tout internet" (BFS unique sur tous les start_urls)
"""

import asyncio
import aiohttp
import hashlib
import io
import json
import os
import random
import re
import time
from typing import Optional, Callable, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    import trafilatura
    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False

try:
    from langdetect import detect as _langdetect, LangDetectException
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False

try:
    from pdfminer.high_level import extract_text as _pdfminer_extract
    _HAS_PDFMINER = True
except ImportError:
    _HAS_PDFMINER = False


# ---------------------------------------------------------------------------
# Helpers de filtrage
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _letters_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(c.isalpha() for c in text)
    return letters / len(text)


def _detect_lang(text: str) -> Optional[str]:
    if not _HAS_LANGDETECT or not text:
        return None
    try:
        return _langdetect(text[:3000])
    except LangDetectException:
        return None


def _is_url_blacklisted(url: str, blacklist: list) -> bool:
    url_lower = url.lower()
    return any(token in url_lower for token in blacklist)


def _is_pdf_url(url: str) -> bool:
    """Renvoie True si l'URL pointe vraisemblablement vers un PDF."""
    path = urlparse(url).path.lower().split("?")[0]
    return path.endswith(".pdf")


# ---------------------------------------------------------------------------
# Extraction PDF
# ---------------------------------------------------------------------------

def _extract_pdf_text(data: bytes, url: str) -> Tuple[str, str]:
    """
    Extrait le texte brut d'un PDF (bytes).

    Returns:
        (title, content)
    """
    if not _HAS_PDFMINER:
        return "", ""
    try:
        text = _pdfminer_extract(io.BytesIO(data))
        text = re.sub(r"\s{3,}", "\n", text or "").strip()
        # Titre = nom du fichier sans extension
        filename = urlparse(url).path.rstrip("/").split("/")[-1]
        title = re.sub(r"[_\-]+", " ", filename.rsplit(".", 1)[0]).strip()
        return title, text
    except Exception:
        return "", ""


# ---------------------------------------------------------------------------
# Signatures CAPTCHA
# ---------------------------------------------------------------------------

_CAPTCHA_SIGNALS = [
    "cf-browser-verification", "cf_clearance", "checking your browser",
    "enable javascript and cookies", "ray id",
    "recaptcha", "hcaptcha.com", "www.google.com/recaptcha",
    "captcha", "robot check", "are you a robot", "i am not a robot",
    "verify you are human", "please verify", "access denied",
    "ddos-guard", "datadome",
]

def _is_captcha_page(html: str) -> bool:
    lower = html.lower()
    return any(sig in lower for sig in _CAPTCHA_SIGNALS)


# ---------------------------------------------------------------------------
# Extraction HTML
# ---------------------------------------------------------------------------

def _extract_main_content(html: str, url: str) -> Tuple[str, str]:
    title = ""
    content = ""

    if _HAS_TRAFILATURA:
        extracted = trafilatura.extract(
            html, url=url,
            include_comments=False, include_tables=True,
            no_fallback=False, favor_precision=True,
        )
        content = extracted or ""
        soup = BeautifulSoup(html, "lxml")
        t = soup.find("title")
        title = t.get_text(strip=True) if t else ""
    else:
        soup = BeautifulSoup(html, "lxml")
        t = soup.find("title")
        title = t.get_text(strip=True) if t else ""
        for tag in soup(["script", "style", "noscript", "nav", "footer",
                         "header", "aside", "form", "button", "svg", "iframe"]):
            tag.decompose()
        body = soup.find("body")
        raw = (body or soup).get_text(separator=" ", strip=True)
        content = re.sub(r"\s{2,}", " ", raw).strip()

    return title, content


# ---------------------------------------------------------------------------
# Headers navigateur réalistes
# ---------------------------------------------------------------------------

_BROWSER_HEADERS = [
    {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    },
    {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "fr,fr-FR;q=0.8,en-GB;q=0.5,en;q=0.3",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Sec-Fetch-User": "?1",
    },
]


# ---------------------------------------------------------------------------
# Rotateurs User-Agent / Proxy
# ---------------------------------------------------------------------------

class _UARotator:
    _DEFAULTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0",
    ]

    def __init__(self, user_agents: list):
        self._agents = list(user_agents) if user_agents else self._DEFAULTS

    def get(self) -> str:
        return random.choice(self._agents)

    def get_headers(self) -> dict:
        base = random.choice(_BROWSER_HEADERS).copy()
        base["User-Agent"] = self.get()
        return base


class _ProxyRotator:
    def __init__(self, proxies: list, cooldown: float = 60.0):
        self._proxies: list = list(proxies) if proxies else [None]
        self._cooldown = cooldown
        self._blocked_until: dict = {}
        self._idx: int = 0
        self._lock: Optional[asyncio.Lock] = None

    def _init_lock(self):
        if self._lock is None:
            self._lock = asyncio.Lock()

    async def get(self) -> Optional[str]:
        self._init_lock()
        async with self._lock:
            n = len(self._proxies)
            now = time.time()
            for _ in range(n):
                p = self._proxies[self._idx % n]
                self._idx = (self._idx + 1) % n
                if now >= self._blocked_until.get(p or "__direct__", 0.0):
                    return p
            p = self._proxies[self._idx % n]
            self._idx = (self._idx + 1) % n
            return p

    async def penalize(self, proxy: Optional[str]):
        self._init_lock()
        async with self._lock:
            self._blocked_until[proxy or "__direct__"] = time.time() + self._cooldown


# ---------------------------------------------------------------------------
# Throttling par domaine
# ---------------------------------------------------------------------------

class _DomainThrottler:
    def __init__(self, delay: float, jitter_ratio: float = 0.5):
        self._delay = delay
        self._jitter_ratio = jitter_ratio
        self._last_req: dict = {}
        self._locks: dict = {}
        self._meta_lock: Optional[asyncio.Lock] = None

    def _init(self):
        if self._meta_lock is None:
            self._meta_lock = asyncio.Lock()

    async def _get_lock(self, domain: str) -> asyncio.Lock:
        self._init()
        async with self._meta_lock:
            if domain not in self._locks:
                self._locks[domain] = asyncio.Lock()
            return self._locks[domain]

    async def wait(self, url: str):
        domain = urlparse(url).netloc
        lock = await self._get_lock(domain)
        async with lock:
            now = time.time()
            last = self._last_req.get(domain, 0.0)
            jitter = random.uniform(0, self._delay * self._jitter_ratio)
            wait_t = self._delay + jitter - (now - last)
            if wait_t > 0:
                await asyncio.sleep(wait_t)
            self._last_req[domain] = time.time()


# ---------------------------------------------------------------------------
# FetchResult
# ---------------------------------------------------------------------------

class _FetchResult:
    """Résultat d'un fetch : contenu brut + type détecté."""
    __slots__ = ("html", "pdf_bytes", "is_pdf")

    def __init__(self, html: Optional[str] = None,
                 pdf_bytes: Optional[bytes] = None):
        self.html = html
        self.pdf_bytes = pdf_bytes
        self.is_pdf = pdf_bytes is not None


# ---------------------------------------------------------------------------
# WebCrawler
# ---------------------------------------------------------------------------

class WebCrawler:
    """Crawler web BFS asynchrone avec pipeline de qualité pour dataset LLM."""

    _PROGRESS_EVERY = 50

    def __init__(self, database, config: dict, logger: Optional[Callable] = None):
        self.database = database
        self.config = config
        self.logger = logger or (lambda msg, level="INFO": None)

        self.visited: set = set()
        self.is_running: bool = False

        self.stats = {
            "visited": 0, "requests": 0, "blocked": 0, "errors": 0,
            "saved": 0, "saved_pdf": 0,
            "skipped_url": 0, "skipped_dup": 0,
            "skipped_lang": 0, "skipped_quality": 0,
        }

        self._jsonl_file = None
        self._pbar_visited: Optional[tqdm] = None
        self._pbar_saved: Optional[tqdm] = None

        self._ua_rot = _UARotator(config.get("user_agents", []))
        self._proxy_rot = _ProxyRotator(
            config.get("proxies", []),
            cooldown=config.get("proxy_cooldown", 60.0),
        )
        self._throttler = _DomainThrottler(
            delay=config.get("politeness_delay", 0.3),
            jitter_ratio=config.get("jitter_ratio", 0.5),
        )

        self._site_pages_start: int = 0
        self._site_max_pages: int = config.get("max_pages", 9_999_999)

        self.visited_lock: Optional[asyncio.Lock] = None
        self.stats_lock: Optional[asyncio.Lock] = None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, message: str, level: str = "INFO"):
        if level in ("WARN", "ERROR", "SUCCESS"):
            tqdm.write(self._colorize(message, level))
        elif level == "INFO" and message.startswith(
                ("[SAVED]", "[SITE]", "[INFO]", "=", "  ", "[SITES]")):
            tqdm.write(self._colorize(message, level))

    @staticmethod
    def _colorize(message: str, level: str) -> str:
        COLORS = {"INFO": "\033[0m", "WARN": "\033[93m",
                  "ERROR": "\033[91m", "SUCCESS": "\033[92m"}
        return f"{COLORS.get(level, '')} {message}\033[0m"

    def _update_bars(self, visited_delta: int = 0, saved_delta: int = 0):
        if self._pbar_visited and visited_delta:
            self._pbar_visited.update(visited_delta)
        if self._pbar_saved and saved_delta:
            self._pbar_saved.update(saved_delta)

    def _log_progress(self):
        saved = self.stats["saved"]
        if saved > 0 and saved % self._PROGRESS_EVERY == 0:
            if self._pbar_visited:
                self._pbar_visited.set_postfix({
                    "bloqués": self.stats["blocked"],
                    "erreurs": self.stats["errors"],
                    "dup": self.stats["skipped_dup"],
                    "pdf": self.stats["saved_pdf"],
                }, refresh=True)

    # ------------------------------------------------------------------
    # Réseau
    # ------------------------------------------------------------------

    async def _count_request(self):
        async with self.stats_lock:
            self.stats["requests"] += 1

    async def fetch(self, session: aiohttp.ClientSession, url: str,
                    max_retries: int = 3) -> Optional[_FetchResult]:
        """
        GET avec retry + backoff exponentiel.
        Retourne un _FetchResult (HTML ou PDF bytes) ou None si échec.
        """
        extract_pdf = self.config.get("extract_pdf", True)

        for attempt in range(max_retries):
            await self._throttler.wait(url)
            proxy = await self._proxy_rot.get()
            headers = self._ua_rot.get_headers()

            # Pour les PDF, on accepte aussi application/pdf
            if _is_pdf_url(url) and extract_pdf:
                headers["Accept"] = "application/pdf,*/*;q=0.8"

            try:
                await self._count_request()
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=20),
                    proxy=proxy,
                    headers=headers,
                    allow_redirects=True,
                ) as resp:

                    if resp.status in (403, 429):
                        retry_after = float(resp.headers.get("Retry-After", 0))
                        backoff = retry_after or (2 ** attempt * 2 + random.uniform(0, 2))
                        async with self.stats_lock:
                            self.stats["blocked"] += 1
                        await self._proxy_rot.penalize(proxy)
                        if attempt < max_retries - 1:
                            await asyncio.sleep(min(backoff, 60))
                            continue
                        self.log(f"[BLOCKED] {resp.status} : {url}", "WARN")
                        return None

                    if resp.status in (500, 502, 503, 504):
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
                            continue
                        async with self.stats_lock:
                            self.stats["errors"] += 1
                        return None

                    if resp.status >= 400:
                        async with self.stats_lock:
                            self.stats["errors"] += 1
                        return None

                    content_type = resp.headers.get("content-type", "").lower()

                    # ── PDF ──────────────────────────────────────────
                    if extract_pdf and (
                        "application/pdf" in content_type or _is_pdf_url(url)
                    ):
                        if not _HAS_PDFMINER:
                            self.log(
                                "[WARN] pdfminer.six absent — PDF ignoré. "
                                "pip install pdfminer.six", "WARN"
                            )
                            return None
                        pdf_bytes = await resp.read()
                        return _FetchResult(pdf_bytes=pdf_bytes)

                    # ── HTML ─────────────────────────────────────────
                    if resp.status == 200 and "text/html" in content_type:
                        html = await resp.text()
                        if _is_captcha_page(html):
                            self.log(f"[CAPTCHA] Mur détecté : {url}", "WARN")
                            async with self.stats_lock:
                                self.stats["blocked"] += 1
                            await self._proxy_rot.penalize(proxy)
                            return None
                        return _FetchResult(html=html)

                    return None

            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                async with self.stats_lock:
                    self.stats["errors"] += 1
            except (aiohttp.ClientError, ConnectionResetError):
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                async with self.stats_lock:
                    self.stats["errors"] += 1
            except Exception:
                async with self.stats_lock:
                    self.stats["errors"] += 1
                break

        return None

    # ------------------------------------------------------------------
    # Extraction des liens
    # ------------------------------------------------------------------

    def extract_links(self, base_url: str, html: str) -> set:
        soup = BeautifulSoup(html, "lxml")
        base_domain = urlparse(base_url).netloc
        stay_on_domain = self.config.get("stay_on_domain", True)
        extract_pdf = self.config.get("extract_pdf", True)
        links = set()

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("mailto:", "javascript:", "tel:")):
                continue
            link = urljoin(base_url, href)
            parsed = urlparse(link)
            if parsed.scheme not in ("http", "https"):
                continue
            if stay_on_domain and parsed.netloc and base_domain not in parsed.netloc:
                continue
            # Conserver les PDF si extraction activée, sinon filtrer
            if _is_pdf_url(link) and not extract_pdf:
                continue
            clean = link.split("#")[0].rstrip("/") or link
            links.add(clean)

        return links

    # ------------------------------------------------------------------
    # Pipeline de qualité
    # ------------------------------------------------------------------

    def _passes_pipeline(self, url: str, content: str) -> Tuple[bool, str]:
        cfg = self.config
        words = content.split()
        word_count = len(words)

        if word_count < cfg.get("min_words", 50):
            return False, f"trop court ({word_count} mots)"
        if word_count > cfg.get("max_words", 100_000):
            return False, f"trop long ({word_count} mots)"

        ratio = _letters_ratio(content)
        if ratio < cfg.get("min_letters_ratio", 0.7):
            return False, f"ratio lettres trop faible ({ratio:.2f})"

        target_langs = cfg.get("target_langs", [])
        if target_langs:
            lang = _detect_lang(content)
            if lang not in target_langs:
                return False, f"langue non cible ({lang})"

        return True, ""

    def _write_jsonl(self, record: dict):
        if self._jsonl_file:
            self._jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._jsonl_file.flush()

    # ------------------------------------------------------------------
    # Sauvegarde commune HTML + PDF
    # ------------------------------------------------------------------

    def _save_page(self, url: str, title: str, content: str,
                   depth: int, links: set, source: str = "html") -> bool:
        """Déduplication + qualité + insert DB + JSONL. Retourne True si sauvé."""
        content_hash = _sha256(content)
        if self.database.hash_exists(content_hash):
            return False  # doublon

        ok, reason = self._passes_pipeline(url, content)
        if not ok:
            return False  # qualité/langue insuffisante

        word_count = len(content.split())
        saved = self.database.insert_page(
            url=url, title=title, content=content,
            content_hash=content_hash, depth=depth,
            status_code=200, word_count=word_count,
            link_count=len(links),
        )
        if saved:
            self._write_jsonl({
                "text": content, "url": url,
                "title": title, "word_count": word_count,
                "source": source,
            })
        return saved

    # ------------------------------------------------------------------
    # Traitement d'une URL
    # ------------------------------------------------------------------

    async def process_url(self, url: str, depth: int,
                          session: aiohttp.ClientSession,
                          queue: asyncio.Queue):

        # ── Filtre URL ────────────────────────────────────────────────
        blacklist = self.config.get("url_blacklist", [])
        if _is_url_blacklisted(url, blacklist):
            async with self.stats_lock:
                self.stats["skipped_url"] += 1
            return

        # ── Visited + limites ─────────────────────────────────────────
        async with self.visited_lock:
            if url in self.visited:
                return
            if self.stats["visited"] >= self.config["max_pages"]:
                return
            if depth > self.config["max_depth"]:
                return
            self.visited.add(url)

        async with self.stats_lock:
            self.stats["visited"] += 1
        self._update_bars(visited_delta=1)

        # ── Fetch ─────────────────────────────────────────────────────
        result = await self.fetch(session, url)
        if not result:
            return

        # ── Branche PDF ───────────────────────────────────────────────
        if result.is_pdf:
            title, content = _extract_pdf_text(result.pdf_bytes, url)
            if not content.strip():
                return

            content_hash = _sha256(content)
            if self.database.hash_exists(content_hash):
                async with self.stats_lock:
                    self.stats["skipped_dup"] += 1
                return

            ok, reason = self._passes_pipeline(url, content)
            if not ok:
                async with self.stats_lock:
                    if "langue" in reason:
                        self.stats["skipped_lang"] += 1
                    else:
                        self.stats["skipped_quality"] += 1
                return

            word_count = len(content.split())
            saved = self.database.insert_page(
                url=url, title=title, content=content,
                content_hash=content_hash, depth=depth,
                status_code=200, word_count=word_count, link_count=0,
            )
            if saved:
                async with self.stats_lock:
                    self.stats["saved"] += 1
                    self.stats["saved_pdf"] += 1
                self._update_bars(saved_delta=1)
                self._write_jsonl({
                    "text": content, "url": url,
                    "title": title, "word_count": word_count,
                    "source": "pdf_web",
                })
                tqdm.write(self._colorize(
                    f"[PDF-SAVED] « {title[:60] or url} » — {word_count} mots | {url}",
                    "SUCCESS",
                ))
                self._log_progress()
            return

        # ── Branche HTML ──────────────────────────────────────────────
        html = result.html
        title, content = _extract_main_content(html, url)

        content_hash = _sha256(content)
        if self.database.hash_exists(content_hash):
            async with self.stats_lock:
                self.stats["skipped_dup"] += 1
            links = self.extract_links(url, html)
            async with self.visited_lock:
                for link in links:
                    if link not in self.visited:
                        await queue.put((link, depth + 1))
            return

        ok, reason = self._passes_pipeline(url, content)
        if not ok:
            async with self.stats_lock:
                if "langue" in reason:
                    self.stats["skipped_lang"] += 1
                else:
                    self.stats["skipped_quality"] += 1
            links = self.extract_links(url, html)
            async with self.visited_lock:
                for link in links:
                    if link not in self.visited:
                        await queue.put((link, depth + 1))
            return

        word_count = len(content.split())
        links = self.extract_links(url, html)
        saved = self.database.insert_page(
            url=url, title=title, content=content,
            content_hash=content_hash, depth=depth,
            status_code=200, word_count=word_count,
            link_count=len(links),
        )
        if saved:
            async with self.stats_lock:
                self.stats["saved"] += 1
            self._update_bars(saved_delta=1)
            tqdm.write(self._colorize(
                f"[SAVED] « {title[:70] or url} » — {word_count} mots | {url}",
                "SUCCESS",
            ))
            self._log_progress()
            self._write_jsonl({
                "text": content, "url": url,
                "title": title, "word_count": word_count,
                "source": "html",
            })

        async with self.visited_lock:
            for link in links:
                if link not in self.visited:
                    await queue.put((link, depth + 1))

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    async def worker(self, worker_id: int, session: aiohttp.ClientSession,
                     queue: asyncio.Queue):
        while self.is_running:
            try:
                url, depth = await asyncio.wait_for(queue.get(), timeout=3.0)
                async with self.stats_lock:
                    pages_done = self.stats["visited"]
                if pages_done >= self.config["max_pages"] or pages_done >= self._site_max_pages:
                    queue.task_done()
                    break
                await self.process_url(url, depth, session, queue)
                queue.task_done()
            except asyncio.TimeoutError:
                async with self.stats_lock:
                    done = (self.stats["visited"] >= self.config["max_pages"]
                            or self.stats["visited"] >= self._site_max_pages)
                if done or queue.empty():
                    break
            except Exception as e:
                self.log(f"[ERROR] Worker-{worker_id} : {e}", "ERROR")
                try:
                    queue.task_done()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # BFS
    # ------------------------------------------------------------------

    async def _run_bfs(self, seed_urls: list, session: aiohttp.ClientSession):
        queue: asyncio.Queue = asyncio.Queue()
        for url in seed_urls:
            await queue.put((url, 0))
        workers = [
            asyncio.create_task(self.worker(i + 1, session, queue))
            for i in range(self.config["num_workers"])
        ]
        await asyncio.gather(*workers, return_exceptions=True)

    # ------------------------------------------------------------------
    # Point d'entrée
    # ------------------------------------------------------------------

    async def _crawl_site(self, start_url: str, session: aiohttp.ClientSession):
        async with self.visited_lock:
            self.visited.clear()
        max_per_site = self.config.get("max_pages_per_site", self.config["max_pages"])
        async with self.stats_lock:
            self._site_pages_start = self.stats["visited"]
            self._site_max_pages = self._site_pages_start + max_per_site
        self.log(f"[SITE] Début : {start_url}", "INFO")
        await self._run_bfs([start_url], session)
        async with self.stats_lock:
            n = self.stats["visited"] - self._site_pages_start
        self.log(f"[SITE] Fin : {start_url} — {n} pages visitées", "INFO")

    async def _crawl_global(self, start_urls: list, session: aiohttp.ClientSession):
        self._site_max_pages = self.config["max_pages"]
        self.log(f"[INFO] Mode crawl global — {len(start_urls)} URL(s) de départ", "INFO")
        await self._run_bfs(start_urls, session)

    async def run_all(self, start_urls: list) -> dict:
        self.is_running = True
        self.visited_lock = asyncio.Lock()
        self.stats_lock = asyncio.Lock()
        start_time = time.time()
        max_pages = self.config["max_pages"]

        jsonl_path = self.config.get("jsonl_output", "data/datasets/dataset.jsonl")
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        self._jsonl_file = open(jsonl_path, "a", encoding="utf-8")

        self._pbar_visited = tqdm(
            total=max_pages, desc="  Visitées", unit="page", colour="cyan",
            position=0, leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
        )
        self._pbar_saved = tqdm(
            total=max_pages, desc="  Sauvées ", unit="page", colour="green",
            position=1, leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]{postfix}",
        )

        connector = aiohttp.TCPConnector(
            limit=self.config["concurrency"],
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )

        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                if not self.config.get("stay_on_domain", True):
                    await self._crawl_global(start_urls, session)
                else:
                    for i, url in enumerate(start_urls):
                        if not self.is_running:
                            break
                        async with self.stats_lock:
                            global_done = self.stats["visited"] >= self.config["max_pages"]
                        if global_done:
                            self.log("[INFO] Limite globale atteinte — arrêt.", "INFO")
                            break
                        self.log(f"\n[SITES] {i + 1}/{len(start_urls)} → {url}", "INFO")
                        await self._crawl_site(url, session)
        finally:
            if self._jsonl_file:
                self._jsonl_file.close()
                self._jsonl_file = None
            for pb in (self._pbar_visited, self._pbar_saved):
                if pb:
                    pb.close()
            self._pbar_visited = self._pbar_saved = None

        elapsed = time.time() - start_time
        speed = self.stats["visited"] / elapsed if elapsed > 0 else 0

        sep = "=" * 70
        for line in [
            f"\n{sep}", "  STATISTIQUES FINALES", sep,
            f"  Pages visitées       : {self.stats['visited']}",
            f"  Pages sauvées        : {self.stats['saved']}",
            f"    dont PDF web       : {self.stats['saved_pdf']}",
            f"  Requêtes HTTP        : {self.stats['requests']}",
            f"  Accès bloqués        : {self.stats['blocked']}",
            f"  Erreurs              : {self.stats['errors']}",
            f"  Skip URL blacklist   : {self.stats['skipped_url']}",
            f"  Skip doublons        : {self.stats['skipped_dup']}",
            f"  Skip langue          : {self.stats['skipped_lang']}",
            f"  Skip qualité         : {self.stats['skipped_quality']}",
            f"  Durée                : {elapsed:.2f} s",
            f"  Vitesse              : {speed:.2f} pages/s",
            f"  Dataset JSONL        : {jsonl_path}",
            sep,
        ]:
            tqdm.write(line)

        return self.stats

    async def run(self, start_url: str) -> dict:
        return await self.run_all([start_url])

    def stop(self):
        self.is_running = False