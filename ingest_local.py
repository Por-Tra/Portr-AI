# -*- coding: utf-8 -*-
"""
ingest_local.py

Ingestion de fichiers locaux vers la base de données du crawler.
=> Plus rapide que le webcrawler pour les sources déjà téléchargées (exemple et conseille : Wikipédia ZIM),

Formats supportés :
    .pdf   → extraction texte via pdfminer.six
    .txt   → lecture UTF-8 directe
    .md    → lecture UTF-8 directe (Markdown → texte brut)
    .zim   → base Kiwix/Wikipédia (libzim) — itère sur tous les articles

Dépendances :
    pip install pdfminer.six langdetect libzim beautifulsoup4 lxml

Usage :
    python ingest_local.py                              # GUI
    python ingest_local.py --files wikipedia_fr.zim     # Wikipédia FR complet
    python ingest_local.py --dir ./mes_docs             # dossier entier
    python ingest_local.py --dir ./docs -r              # récursif
    python ingest_local.py --files a.pdf b.txt c.zim    # mix de formats
"""

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
import sys
from pathlib import Path
from typing import Optional, Iterator, Callable

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Crawler.config import CRAWLER_CONFIG
from data.DataBase.database import Database

try:
    from pdfminer.high_level import extract_text as _pdfminer_extract
    _HAS_PDFMINER = True
except ImportError:
    _HAS_PDFMINER = False

try:
    from langdetect import detect as _langdetect, LangDetectException
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False

try:
    import libzim.reader as _zim_reader
    from bs4 import BeautifulSoup as _BS
    _HAS_LIBZIM = True
except ImportError:
    _HAS_LIBZIM = False

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".rst", ".zim"}
REMOVABLE_HTML_TAGS = (
    "script", "style", "noscript", "nav", "footer",
    "header", "aside", "form", "button", "svg",
    "iframe", "figure",
)
REMOVABLE_WIKI_SELECTORS = (
    ".hatnote",
    ".navbox",
    ".infobox",
    ".sidebar",
    ".toc",
    ".reflist",
    ".reference",
    ".mw-editsection",
)
DEFAULT_CHECKPOINT_DIR = Path("data/datasets/ingest_checkpoints")


# ---------------------------------------------------------------------------
# Helpers communs
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _letters_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isalpha() for c in text) / len(text)


def _detect_lang(text: str) -> Optional[str]:
    if not _HAS_LANGDETECT or not text:
        return None
    try:
        return _langdetect(text[:3000])
    except LangDetectException:
        return None


def _clean_text(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _html_to_text(html: str) -> str:
    """Convertit du HTML en texte brut propre (utilisé pour les articles ZIM)."""
    try:
        soup = _BS(html, "lxml")
        content_root = (
            soup.find(id="mw-content-text")
            or soup.find("main")
            or soup.find("article")
            or soup.body
            or soup
        )

        for tag in content_root(REMOVABLE_HTML_TAGS):
            tag.decompose()

        # Nettoyer les blocs MediaWiki les plus verbeux sans rescanner tout le DOM.
        for selector in REMOVABLE_WIKI_SELECTORS:
            for el in content_root.select(selector):
                el.decompose()

        text = content_root.get_text(separator=" ", strip=True)
        return _clean_text(re.sub(r"\s{2,}", " ", text))
    except Exception:
        return ""


def _checkpoint_file(checkpoint_dir: Path, path: Path) -> Path:
    source_id = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    return checkpoint_dir / f"{path.stem}_{source_id}.json"


def _load_checkpoint(checkpoint_path: Path, source_path: Path) -> Optional[dict]:
    if not checkpoint_path.exists():
        return None

    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    expected_size = source_path.stat().st_size
    expected_mtime = int(source_path.stat().st_mtime)
    if (
        payload.get("source_path") != str(source_path.resolve())
        or payload.get("source_size") != expected_size
        or payload.get("source_mtime") != expected_mtime
    ):
        return None

    return payload


def _save_checkpoint(
    checkpoint_path: Path,
    source_path: Path,
    next_entry_index: int,
    saved_count: int,
    scanned_count: int,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    source_stat = source_path.stat()
    payload = {
        "source_path": str(source_path.resolve()),
        "source_size": source_stat.st_size,
        "source_mtime": int(source_stat.st_mtime),
        "next_entry_index": next_entry_index,
        "saved_count": saved_count,
        "scanned_count": scanned_count,
    }
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=checkpoint_path.parent,
        delete=False,
        suffix=".tmp",
    ) as tmp_file:
        json.dump(payload, tmp_file, ensure_ascii=False, indent=2)
        temp_name = tmp_file.name
    os.replace(temp_name, checkpoint_path)


def _clear_checkpoint(checkpoint_path: Path) -> None:
    try:
        checkpoint_path.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Extracteurs fichiers classiques
# ---------------------------------------------------------------------------

def extract_pdf(path: Path) -> tuple:
    if not _HAS_PDFMINER:
        raise RuntimeError(
            "pdfminer.six est requis.\n  pip install pdfminer.six"
        )
    with open(path, "rb") as f:
        raw = _pdfminer_extract(io.BytesIO(f.read()))
    content = _clean_text(raw or "")
    title = re.sub(r"[_\-]+", " ", path.stem).strip()
    return title, content


def extract_text_file(path: Path) -> tuple:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            raw = path.read_text(encoding=enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        raise UnicodeDecodeError(f"Impossible de décoder {path}")

    if path.suffix.lower() in (".md", ".rst"):
        raw = re.sub(r"#{1,6}\s+", "", raw)
        raw = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", raw)
        raw = re.sub(r"`{1,3}[^`]*`{1,3}", "", raw)
        raw = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", raw)
        raw = re.sub(r"^\s*[-*+]\s+", "", raw, flags=re.MULTILINE)

    content = _clean_text(raw)
    title = re.sub(r"[_\-]+", " ", path.stem).strip()
    return title, content


def extract_file(path: Path) -> tuple:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf(path)
    if ext in (".txt", ".md", ".rst"):
        return extract_text_file(path)
    raise ValueError(f"Format non supporté : {ext}")


# ---------------------------------------------------------------------------
# Itérateur ZIM
# ---------------------------------------------------------------------------

def _iter_zim_articles(
    archive,
    zim_name: str,
    progress_callback: Optional[Callable[[int], None]] = None,
    progress_every: int = 250,
    start_index: int = 0,
) -> Iterator[tuple]:
    """
    Itère sur tous les articles HTML d'une archive ZIM.
    Compatible avec les deux générations d'API libzim (ancienne et moderne).

    Yields:
        (title: str, content: str, url: str)
    """
    # ── Déterminer la méthode d'itération disponible ─────────────────
    # API moderne (libzim >= 1.0) : archive est itérable directement
    # API ancienne                : get_entry_by_id(i) sur entry_count
    use_iter = hasattr(archive, "__iter__")

    def _get_entries():
        if use_iter:
            for index, entry in enumerate(archive):
                if index < start_index:
                    continue
                yield index, entry
        else:
            for i in range(start_index, archive.entry_count):
                try:
                    yield i, archive._get_entry_by_id(i)
                except Exception:
                    continue

    scanned_entries = start_index
    pending_progress = 0

    for entry_index, entry in _get_entries():
        scanned_entries = entry_index + 1
        pending_progress += 1
        if progress_callback is not None and pending_progress >= progress_every:
            progress_callback(pending_progress, scanned_entries)
            pending_progress = 0

        try:
            # Ignorer les redirections
            if entry.is_redirect:
                continue

            item = entry.get_item()
            mimetype = item.mimetype.lower() if item.mimetype else ""

            # Ne garder que les pages HTML
            if "text/html" not in mimetype:
                continue

            raw_html = bytes(item.content).decode("utf-8", errors="replace")
            title = entry.title or entry.path.split("/")[-1]
            content = _html_to_text(raw_html)

            if not content:
                continue

            url = f"zim://{zim_name}/{entry.path}"
            yield entry_index, title, content, url

        except Exception:
            continue  # article corrompu ou non lisible → on passe

    if progress_callback is not None and pending_progress:
        progress_callback(pending_progress, scanned_entries)


def ingest_zim(path: Path, db: Database, config: dict,
               jsonl_file, stats: dict, commit_every: int = 100,
               flush_every: int = 100, progress_every: int = 250,
               checkpoint_every: int = 2_000,
               checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
               resume: bool = True) -> None:
    """Ingère tous les articles d'un fichier ZIM dans la base."""

    if not _HAS_LIBZIM:
        print(
            "\n  [ERREUR] libzim non installé.\n"
            "  pip install libzim\n"
            "  Sur Linux : apt install libzim-dev  puis  pip install libzim"
        )
        stats["errors"] += 1
        return

    try:
        archive = _zim_reader.Archive(str(path))
    except Exception as e:
        print(f"\n  [ERREUR] Impossible d'ouvrir {path.name} : {e}")
        stats["errors"] += 1
        return

    target_langs = config.get("target_langs", [])
    min_w       = config.get("min_words", 10)
    max_w       = config.get("max_words", 100_000)
    min_ratio   = config.get("min_letters_ratio", 0.70)
    zim_name    = path.stem
    checkpoint_path = _checkpoint_file(checkpoint_dir, path)
    checkpoint = _load_checkpoint(checkpoint_path, path) if resume else None
    start_index = max(0, int((checkpoint or {}).get("next_entry_index", 0)))
    resumed_saved = int((checkpoint or {}).get("saved_count", 0))

    try:
        from tqdm import tqdm
        pbar = tqdm(
            total=archive.entry_count,
            initial=start_index,
            desc=f"  ZIM {zim_name[:30]}",
            unit="ent", # abréviation pour "entrée"
            colour="blue",
            dynamic_ncols=True,
            mininterval=0.5,
        )
    except ImportError:
        pbar = None

    print(f"\n  [ZIM] {path.name} — {archive.entry_count} entrées")
    if start_index:
        print(
            f"  [ZIM] Reprise au scan {start_index}/{archive.entry_count} "
            f"({resumed_saved} article(s) déjà sauvegardé(s) sur cette progression)"
        )
    article_count = 0
    pending_db_writes = 0
    pending_jsonl_writes = 0
    current_scanned_index = start_index
    last_checkpoint_index = start_index

    def _persist_progress(force: bool = False) -> None:
        nonlocal pending_db_writes, pending_jsonl_writes, last_checkpoint_index
        if not force and current_scanned_index - last_checkpoint_index < checkpoint_every:
            return

        if pending_db_writes:
            db.commit()
            pending_db_writes = 0
        if pending_jsonl_writes:
            jsonl_file.flush()
            pending_jsonl_writes = 0

        _save_checkpoint(
            checkpoint_path=checkpoint_path,
            source_path=path,
            next_entry_index=current_scanned_index,
            saved_count=stats["saved"],
            scanned_count=current_scanned_index,
        )
        last_checkpoint_index = current_scanned_index

    def _update_progress(increment: int, scanned_index: int) -> None:
        nonlocal current_scanned_index
        current_scanned_index = scanned_index
        if pbar is None:
            _persist_progress()
            return

        pbar.update(increment)
        pbar.set_postfix({
            "articles": article_count,
            "sauves": stats["saved"],
            "dup": stats["skipped_dup"],
            "qual": stats["skipped_quality"],
            "lang": stats["skipped_lang"],
        }, refresh=False)
        _persist_progress()

    try:
        for entry_index, title, content, url in _iter_zim_articles(
            archive,
            zim_name,
            progress_callback=_update_progress,
            progress_every=progress_every,
            start_index=start_index,
        ):
            article_count += 1
            current_scanned_index = entry_index + 1

            # ── Déduplication ─────────────────────────────────────────────
            content_hash = _sha256(content)
            if db.hash_exists(content_hash):
                stats["skipped_dup"] += 1
                continue

            # ── Longueur ──────────────────────────────────────────────────
            words = content.split()
            word_count = len(words)
            if word_count < min_w or word_count > max_w:
                stats["skipped_quality"] += 1
                continue

            # ── Ratio lettres ─────────────────────────────────────────────
            if _letters_ratio(content) < min_ratio:
                stats["skipped_quality"] += 1
                continue

            # ── Langue ────────────────────────────────────────────────────
            if target_langs:
                lang = _detect_lang(content)
                if lang not in target_langs:
                    stats["skipped_lang"] += 1
                    continue

            # ── Sauvegarde DB ─────────────────────────────────────────────
            saved = db.insert_page(
                url=url, title=title, content=content,
                content_hash=content_hash, depth=0,
                status_code=200, word_count=word_count, link_count=0,
                commit=False,
            )

            if saved:
                jsonl_file.write(json.dumps({
                    "text": content, "url": url, "title": title,
                    "word_count": word_count, "source": "zim",
                }, ensure_ascii=False) + "\n")
                pending_db_writes += 1
                pending_jsonl_writes += 1
                stats["saved"] += 1

                if pending_db_writes >= commit_every:
                    db.commit()
                    pending_db_writes = 0

                if pending_jsonl_writes >= flush_every:
                    jsonl_file.flush()
                    pending_jsonl_writes = 0
            else:
                stats["errors"] += 1
    finally:
        _persist_progress(force=True)
        if pbar is not None:
            pbar.close()

    _clear_checkpoint(checkpoint_path)
    print(f"\n  [ZIM] Terminé — {article_count} articles lus, "
          f"{stats['saved']} sauvés au total.")


# ---------------------------------------------------------------------------
# Sélecteur de fichiers GUI (tkinter)
# ---------------------------------------------------------------------------

def pick_files_gui() -> list:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("[ERREUR] tkinter n'est pas disponible. Utilisez --files ou --dir.")
        sys.exit(1)

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    filetypes = [
        ("Tous les formats supportés", "*.pdf *.txt *.md *.rst *.zim"),
        ("ZIM (Kiwix / Wikipédia)",     "*.zim"),
        ("PDF",                          "*.pdf"),
        ("Texte",                        "*.txt"),
        ("Markdown",                     "*.md *.rst"),
        ("Tous",                         "*.*"),
    ]
    paths = filedialog.askopenfilenames(
        title="Sélectionner les fichiers à ingérer",
        filetypes=filetypes,
        multiple=True,
    )
    root.destroy()
    return [Path(p) for p in paths]


# ---------------------------------------------------------------------------
# Pipeline d'ingestion principal
# ---------------------------------------------------------------------------

def ingest_files(files: list, db: Database, config: dict, jsonl_path: str,
                 commit_every: int = 100, flush_every: int = 100,
                 progress_every: int = 250, checkpoint_every: int = 2_000,
                 checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
                 resume: bool = True) -> dict:
    """Ingère une liste de fichiers (dispatch ZIM / classiques)."""

    zim_files   = [f for f in files if f.suffix.lower() == ".zim"]
    other_files = [f for f in files if f.suffix.lower() != ".zim"]

    stats = {
        "total": len(files),
        "saved": 0,
        "skipped_dup": 0,
        "skipped_quality": 0,
        "skipped_lang": 0,
        "errors": 0,
    }

    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)

    target_langs = config.get("target_langs", [])
    min_w        = config.get("min_words", 10)
    max_w        = config.get("max_words", 100_000)
    min_ratio    = config.get("min_letters_ratio", 0.70)

    with open(jsonl_path, "a", encoding="utf-8") as jsonl_file:

        # ── Fichiers classiques (PDF / TXT / MD) ──────────────────────
        for i, path in enumerate(other_files, 1):
            print(f"  [{i}/{len(other_files)}] {path.name}", end=" ... ", flush=True)

            try:
                title, content = extract_file(path)
            except Exception as e:
                print(f"ERREUR ({e})")
                stats["errors"] += 1
                continue

            if not content.strip():
                print("vide — ignoré")
                stats["skipped_quality"] += 1
                continue

            content_hash = _sha256(content)
            if db.hash_exists(content_hash):
                print("doublon — ignoré")
                stats["skipped_dup"] += 1
                continue

            words = content.split()
            word_count = len(words)
            if word_count < min_w:
                print(f"trop court ({word_count} mots) — ignoré")
                stats["skipped_quality"] += 1
                continue
            if word_count > max_w:
                print(f"trop long ({word_count} mots) — ignoré")
                stats["skipped_quality"] += 1
                continue

            if _letters_ratio(content) < min_ratio:
                print("ratio lettres faible — ignoré")
                stats["skipped_quality"] += 1
                continue

            if target_langs:
                lang = _detect_lang(content)
                if lang not in target_langs:
                    print(f"langue non cible ({lang}) — ignoré")
                    stats["skipped_lang"] += 1
                    continue

            url = path.resolve().as_uri()
            saved = db.insert_page(
                url=url, title=title, content=content,
                content_hash=content_hash, depth=0,
                status_code=200, word_count=word_count, link_count=0,
            )

            if saved:
                jsonl_file.write(json.dumps({
                    "text": content, "url": url, "title": title,
                    "word_count": word_count,
                    "source": f"local_{path.suffix.lstrip('.').lower()}",
                }, ensure_ascii=False) + "\n")
                jsonl_file.flush()
                stats["saved"] += 1
                print(f"OK — {word_count} mots")
            else:
                print("erreur DB")
                stats["errors"] += 1

        # ── Fichiers ZIM ──────────────────────────────────────────────
        for zim_path in zim_files:
            ingest_zim(
                path=zim_path, db=db, config=config,
                jsonl_file=jsonl_file, stats=stats,
                commit_every=commit_every,
                flush_every=flush_every,
                progress_every=progress_every,
                checkpoint_every=checkpoint_every,
                checkpoint_dir=checkpoint_dir,
                resume=resume,
            )

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _collect_files(args) -> list:
    files = []

    if args.files:
        for f in args.files:
            p = Path(f)
            if not p.exists():
                print(f"[WARN] Fichier introuvable : {p}")
            elif p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                print(f"[WARN] Format non supporté : {p.name}")
            else:
                files.append(p)

    if args.dir:
        folder = Path(args.dir)
        if not folder.is_dir():
            print(f"[ERREUR] Dossier introuvable : {folder}")
            sys.exit(1)
        glob = folder.rglob if args.recursive else folder.glob
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(sorted(glob(f"*{ext}")))

    if not files and not args.files and not args.dir:
        print("Ouverture du sélecteur de fichiers…")
        files = pick_files_gui()
        if not files:
            print("Aucun fichier sélectionné — abandon.")
            sys.exit(0)

    seen, unique = set(), []
    for f in files:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Ingestion de fichiers locaux (PDF, TXT, MD, ZIM) → base crawler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python ingest_local.py                              # GUI\n"
            "  python ingest_local.py --files wikipedia_fr.zim     # Wikipédia FR\n"
            "  python ingest_local.py --dir ./mes_docs             # dossier\n"
            "  python ingest_local.py --dir ./docs -r              # récursif\n"
            "  python ingest_local.py --files a.pdf b.txt c.zim    # mix\n"
        ),
    )
    parser.add_argument("--files", nargs="+", metavar="FICHIER",
                        help="Fichiers à ingérer (PDF, TXT, MD, ZIM)")
    parser.add_argument("--dir", metavar="DOSSIER",
                        help="Dossier à ingérer")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Récursif (avec --dir)")
    parser.add_argument("--db",
                        default=CRAWLER_CONFIG.get("db_file", "data/datasets/crawler.db"),
                        help="Chemin vers la base SQLite")
    parser.add_argument("--jsonl",
                        default=CRAWLER_CONFIG.get("jsonl_output", "data/datasets/dataset.jsonl"),
                        help="Chemin du fichier JSONL de sortie")
    parser.add_argument("--commit-every", type=int, default=100,
                        help="Valider la base tous les N articles ZIM sauvegardés")
    parser.add_argument("--flush-every", type=int, default=100,
                        help="Vider le buffer JSONL tous les N articles ZIM sauvegardés")
    parser.add_argument("--progress-every", type=int, default=250,
                        help="Mettre à jour la progression tous les N éléments ZIM scannés")
    parser.add_argument("--checkpoint-every", type=int, default=2000,
                        help="Enregistrer un checkpoint ZIM tous les N éléments scannés")
    parser.add_argument("--checkpoint-dir",
                        default=str(DEFAULT_CHECKPOINT_DIR),
                        help="Dossier des checkpoints de reprise ZIM")
    parser.add_argument("--no-resume", action="store_true",
                        help="Désactive la reprise automatique sur checkpoint pour les ZIM")

    args = parser.parse_args()
    files = _collect_files(args)

    if not files:
        print("Aucun fichier à traiter.")
        sys.exit(0)

    zim_count = sum(1 for f in files if f.suffix.lower() == ".zim")

    sep = "=" * 65
    print(sep)
    print(f"  INGESTION LOCALE — {len(files)} fichier(s)")
    if zim_count:
        print(f"  [WARN]  {zim_count} fichier(s) ZIM détecté(s) — peut prendre plusieurs heures")
    print(sep)
    for f in files:
        tag = "[ZIM]" if f.suffix.lower() == ".zim" else "     "
        print(f"  {tag} {f}")
    print(sep)

    with Database(db_file=args.db) as db:
        stats = ingest_files(
            files=files, db=db,
            config=CRAWLER_CONFIG, jsonl_path=args.jsonl,
            commit_every=max(1, args.commit_every),
            flush_every=max(1, args.flush_every),
            progress_every=max(1, args.progress_every),
            checkpoint_every=max(1, args.checkpoint_every),
            checkpoint_dir=Path(args.checkpoint_dir),
            resume=not args.no_resume,
        )

        print(sep)
        print("  RÉSULTATS")
        print(sep)
        print(f"  Fichiers traités    : {stats['total']}")
        print(f"  Sauvegardés         : {stats['saved']}")
        print(f"  Doublons ignorés    : {stats['skipped_dup']}")
        print(f"  Qualité/langue KO   : {stats['skipped_quality'] + stats['skipped_lang']}")
        print(f"  Erreurs             : {stats['errors']}")
        print(sep)

        db_stats = db.get_statistics()
        if db_stats:
            print(f"\n  Base totale         : {db_stats['total_pages']} pages")
            print(f"  Domaines/sources    : {db_stats['unique_domains']}")
            print(f"  Moy. mots/page      : {db_stats['avg_words']}")


if __name__ == "__main__":
    main()