
# -*- coding: utf-8 -*-
"""
Module principal du crawler web.
Crawl récursif en largeur (BFS) avec workers asynchrones parallèles.

Pipeline par page :
    HTML → extraction contenu principal (trafilatura)
         → déduplication (SHA-256)
         → filtre langue (langdetect)
         → filtre qualité (ratio lettres, longueur)
         → sauvegarde SQLite + export JSONL
"""

import asyncio
import aiohttp
import hashlib
import json
import os
import re
import time
from typing import Optional, Callable
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

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


# ---------------------------------------------------------------------------
# Helpers de filtrage
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _letters_ratio(text: str) -> float:
    """Proportion de caractères alphabétiques dans le texte."""
    if not text:
        return 0.0
    letters = sum(c.isalpha() for c in text)
    return letters / len(text)


def _detect_lang(text: str) -> Optional[str]:
    if not _HAS_LANGDETECT or not text:
        return None
    try:
        return _langdetect(text[:3000])   # 3 000 chars suffisent pour la détection
    except LangDetectException:
        return None


def _is_url_blacklisted(url: str, blacklist: list[str]) -> bool:
    url_lower = url.lower()
    return any(token in url_lower for token in blacklist)


def _extract_main_content(html: str, url: str) -> tuple[str, str]:
    """
    Extrait le contenu principal (hors menus/sidebar/boilerplate).
    Utilise trafilatura si dispo, sinon fallback BeautifulSoup.

    Returns:
        (title, content)
    """
    title = ""
    content = ""

    if _HAS_TRAFILATURA:
        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_precision=True,
        )
        content = extracted or ""
        # Titre via BeautifulSoup (léger)
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
# WebCrawler
# ---------------------------------------------------------------------------

class WebCrawler:
    """Crawler web BFS asynchrone avec pipeline de qualité pour dataset LLM."""

    def __init__(self, database, config: dict, logger: Optional[Callable] = None):
        """
        Args:
            database: Instance de Database.
            config:   Dictionnaire de configuration (voir Crawler/config.py).
            logger:   Callable(message, level) pour le logging.
        """
        self.database = database
        self.config = config
        self.logger = logger or (lambda msg, level="INFO": print(msg))

        self.visited: set = set()
        self.is_running: bool = False

        self.stats = {
            "visited": 0,
            "requests": 0,
            "blocked": 0,
            "errors": 0,
            "saved": 0,
            "skipped_url": 0,
            "skipped_dup": 0,
            "skipped_lang": 0,
            "skipped_quality": 0,
        }

        self._jsonl_file = None  # handle ouvert pendant le crawl

        # Locks initialisés dans run()
        self.visited_lock: Optional[asyncio.Lock] = None
        self.stats_lock: Optional[asyncio.Lock] = None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, message: str, level: str = "INFO"):
        self.logger(message, level)

    # ------------------------------------------------------------------
    # Réseau
    # ------------------------------------------------------------------

    async def _count_request(self):
        async with self.stats_lock:
            self.stats["requests"] += 1

    async def check_access(self, session: aiohttp.ClientSession, url: str) -> bool:
        """HEAD rapide pour vérifier que l'URL est accessible."""
        try:
            await self._count_request()
            async with session.head(url, timeout=aiohttp.ClientTimeout(total=5),
                                    allow_redirects=True) as resp:
                if resp.status == 403:
                    self.log(f"[WARN] 403 Forbidden : {url}", "WARN")
                    async with self.stats_lock:
                        self.stats["blocked"] += 1
                    return False
                if resp.status == 429:
                    self.log(f"[WARN] 429 Too Many Requests : {url}", "WARN")
                    async with self.stats_lock:
                        self.stats["blocked"] += 1
                    return False
                if resp.status >= 400:
                    self.log(f"[WARN] HTTP {resp.status} : {url}", "WARN")
                    async with self.stats_lock:
                        self.stats["errors"] += 1
                    return False
                return True
        except asyncio.TimeoutError:
            self.log(f"[WARN] Timeout (HEAD) : {url}", "WARN")
            async with self.stats_lock:
                self.stats["errors"] += 1
            return False
        except Exception:
            async with self.stats_lock:
                self.stats["errors"] += 1
            return False

    async def fetch(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """GET et retourne le HTML brut, ou None si échec/non-HTML."""
        try:
            await self._count_request()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 403:
                    self.log(f"[BLOCKED] 403 : {url}", "ERROR")
                    async with self.stats_lock:
                        self.stats["blocked"] += 1
                    return None
                if resp.status == 429:
                    self.log(f"[BLOCKED] 429 : {url}", "ERROR")
                    async with self.stats_lock:
                        self.stats["blocked"] += 1
                    return None
                content_type = resp.headers.get("content-type", "")
                if resp.status == 200 and "text/html" in content_type:
                    return await resp.text()
                return None
        except asyncio.TimeoutError:
            async with self.stats_lock:
                self.stats["errors"] += 1
        except Exception:
            async with self.stats_lock:
                self.stats["errors"] += 1
        return None

    # ------------------------------------------------------------------
    # Extraction des liens
    # ------------------------------------------------------------------

    def extract_links(self, base_url: str, html: str) -> set:
        """Retourne les liens internes (même domaine) présents dans la page."""
        soup = BeautifulSoup(html, "lxml")
        base_domain = urlparse(base_url).netloc
        links = set()
        filtered = 0

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("mailto:", "javascript:", "tel:")):
                filtered += 1
                continue
            link = urljoin(base_url, href)
            parsed = urlparse(link)
            if parsed.scheme not in ("http", "https"):
                filtered += 1
                continue
            if parsed.netloc == "" or base_domain in parsed.netloc:
                clean = link.split("#")[0].rstrip("/") or link
                links.add(clean)
            else:
                filtered += 1

        self.log(f"   [LINKS] Valides={len(links)} | Filtrés={filtered}", "INFO")
        return links

    # ------------------------------------------------------------------
    # Pipeline de qualité
    # ------------------------------------------------------------------

    def _passes_pipeline(self, url: str, content: str) -> tuple[bool, str]:
        """
        Applique tous les filtres dans l'ordre.

        Returns:
            (ok, reason)  — reason est non-vide si filtré.
        """
        cfg = self.config
        words = content.split()
        word_count = len(words)

        # 1. Longueur minimale
        if word_count < cfg.get("min_words", 50):
            return False, f"trop court ({word_count} mots)"

        # 2. Longueur maximale (évite les pages-dump géantes)
        max_words = cfg.get("max_words", 100_000)
        if word_count > max_words:
            return False, f"trop long ({word_count} mots)"

        # 3. Ratio lettres / total caractères
        ratio = _letters_ratio(content)
        if ratio < cfg.get("min_letters_ratio", 0.7):
            return False, f"ratio lettres trop faible ({ratio:.2f})"

        # 4. Langue
        target_langs = cfg.get("target_langs", [])
        if target_langs:
            lang = _detect_lang(content)
            if lang not in target_langs:
                return False, f"langue non cible ({lang})"

        return True, ""

    def _write_jsonl(self, record: dict):
        """Écrit un enregistrement dans le fichier JSONL (thread-safe via GIL)."""
        if self._jsonl_file:
            self._jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._jsonl_file.flush()

    # ------------------------------------------------------------------
    # Traitement d'une URL
    # ------------------------------------------------------------------

    async def process_url(self, url: str, depth: int,
                          session: aiohttp.ClientSession,
                          queue: asyncio.Queue):
        """Pipeline complet : fetch → extract → filtre → save."""
        # --- Filtre URL ---
        blacklist = self.config.get("url_blacklist", [])
        if _is_url_blacklisted(url, blacklist):
            self.log(f"   [SKIP-URL] Blacklistée : {url}", "INFO")
            async with self.stats_lock:
                self.stats["skipped_url"] += 1
            return

        # --- Double-check visited + limites ---
        async with self.visited_lock:
            if url in self.visited:
                return
            if self.stats["visited"] >= self.config["max_pages"]:
                return
            if depth > self.config["max_depth"]:
                self.log(f"[SKIP] Profondeur max atteinte : {url}", "WARN")
                return

        # --- Vérification HEAD ---
        if not await self.check_access(session, url):
            return

        # --- Marquer comme visité ---
        async with self.visited_lock:
            if url in self.visited:
                return
            self.visited.add(url)

        async with self.stats_lock:
            self.stats["visited"] += 1
            progress = self.stats["visited"]

        self.log(
            f"\n[CRAWL] depth={depth} | {progress}/{self.config['max_pages']} | {url}",
            "INFO",
        )

        # --- Fetch ---
        html = await self.fetch(session, url)
        if not html:
            return

        # --- Extraction contenu principal ---
        title, content = _extract_main_content(html, url)

        # --- Déduplication ---
        content_hash = _sha256(content)
        if self.database.hash_exists(content_hash):
            self.log(f"   [SKIP-DUP] Contenu en double : {url}", "INFO")
            async with self.stats_lock:
                self.stats["skipped_dup"] += 1
            # On extrait quand même les liens pour continuer l'exploration
            links = self.extract_links(url, html)
            async with self.visited_lock:
                for link in links:
                    if link not in self.visited:
                        await queue.put((link, depth + 1))
            return

        # --- Pipeline qualité + langue ---
        ok, reason = self._passes_pipeline(url, content)
        if not ok:
            self.log(f"   [SKIP-Q] {reason} : {url}", "INFO")
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

        # --- Sauvegarde ---
        word_count = len(content.split())
        links = self.extract_links(url, html)
        saved = self.database.insert_page(
            url=url,
            title=title,
            content=content,
            content_hash=content_hash,
            depth=depth,
            status_code=200,
            word_count=word_count,
            link_count=len(links),
        )
        if saved:
            async with self.stats_lock:
                self.stats["saved"] += 1
            self.log(
                f"   [SAVED] « {title[:60] or url} » — {word_count} mots",
                "INFO",
            )
            # Export JSONL
            self._write_jsonl({
                "text": content,
                "url": url,
                "title": title,
                "word_count": word_count,
            })

        # --- Enfile les liens ---
        new_links = 0
        async with self.visited_lock:
            for link in links:
                if link not in self.visited:
                    await queue.put((link, depth + 1))
                    new_links += 1

        self.log(
            f"   [QUEUE] +{new_links} liens | file={queue.qsize()} URLs", "INFO"
        )

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    async def worker(self, worker_id: int, session: aiohttp.ClientSession,
                     queue: asyncio.Queue):
        """Worker qui dépile et traite les URLs."""
        self.log(f"[WORKER-{worker_id}] Démarré", "INFO")
        while self.is_running:
            try:
                url, depth = await asyncio.wait_for(queue.get(), timeout=2.0)

                async with self.stats_lock:
                    pages_done = self.stats["visited"]
                if pages_done >= self.config["max_pages"]:
                    queue.task_done()
                    break

                await self.process_url(url, depth, session, queue)
                queue.task_done()
                await asyncio.sleep(self.config.get("politeness_delay", 0.1))

            except asyncio.TimeoutError:
                async with self.stats_lock:
                    done = self.stats["visited"] >= self.config["max_pages"]
                if done or queue.empty():
                    break
            except Exception as e:
                self.log(f"[ERROR] Worker-{worker_id} : {e}", "ERROR")
                try:
                    queue.task_done()
                except Exception:
                    pass

        self.log(f"[WORKER-{worker_id}] Terminé", "INFO")

    # ------------------------------------------------------------------
    # BFS
    # ------------------------------------------------------------------

    async def crawl_bfs(self, start_url: str, session: aiohttp.ClientSession):
        """Lance le crawl BFS avec workers parallèles."""
        queue: asyncio.Queue = asyncio.Queue()
        await queue.put((start_url, 0))

        num_workers = self.config["num_workers"]
        self.log(f"[INFO] Lancement de {num_workers} workers parallèles", "INFO")

        workers = [
            asyncio.create_task(self.worker(i + 1, session, queue))
            for i in range(num_workers)
        ]
        await asyncio.gather(*workers, return_exceptions=True)
        self.log("[INFO] Tous les workers ont terminé", "INFO")

    # ------------------------------------------------------------------
    # Point d'entrée
    # ------------------------------------------------------------------

    async def run(self, start_url: str) -> dict:
        """Lance le crawl complet depuis start_url."""
        self.is_running = True
        self.visited_lock = asyncio.Lock()
        self.stats_lock = asyncio.Lock()
        start_time = time.time()

        # Ouvrir le fichier JSONL
        jsonl_path = self.config.get("jsonl_output", "data/datasets/dataset.jsonl")
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        self._jsonl_file = open(jsonl_path, "a", encoding="utf-8")

        headers = {
            "User-Agent": self.config["user_agent"],
            "Accept-Encoding": "gzip, br",
        }
        connector = aiohttp.TCPConnector(
            limit=self.config["concurrency"],
            ttl_dns_cache=300,
        )

        try:
            async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
                await self.crawl_bfs(start_url, session)
        finally:
            if self._jsonl_file:
                self._jsonl_file.close()
                self._jsonl_file = None

        elapsed = time.time() - start_time
        speed = self.stats["visited"] / elapsed if elapsed > 0 else 0

        sep = "=" * 70
        self.log(f"\n{sep}", "INFO")
        self.log("  STATISTIQUES FINALES", "INFO")
        self.log(sep, "INFO")
        self.log(f"  Pages visitées       : {self.stats['visited']}", "INFO")
        self.log(f"  Pages sauvées        : {self.stats['saved']}", "INFO")
        self.log(f"  Requêtes HTTP        : {self.stats['requests']}", "INFO")
        self.log(f"  Accès bloqués        : {self.stats['blocked']}", "WARN")
        self.log(f"  Erreurs              : {self.stats['errors']}", "INFO")
        self.log(f"  Skip URL blacklist   : {self.stats['skipped_url']}", "INFO")
        self.log(f"  Skip doublons        : {self.stats['skipped_dup']}", "INFO")
        self.log(f"  Skip langue          : {self.stats['skipped_lang']}", "INFO")
        self.log(f"  Skip qualité         : {self.stats['skipped_quality']}", "INFO")
        self.log(f"  Durée                : {elapsed:.2f} s", "INFO")
        self.log(f"  Vitesse              : {speed:.2f} pages/s", "INFO")
        self.log(f"  Dataset JSONL        : {jsonl_path}", "INFO")
        self.log(sep, "INFO")

        return self.stats

    def stop(self):
        """Arrête le crawler proprement."""
        self.is_running = False