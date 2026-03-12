# -*- coding: utf-8 -*-
"""
Module de gestion de la base de données SQLite avec FTS5.
Stocke les pages crawlées (texte uniquement).
"""

import sqlite3
import os
from urllib.parse import urlparse


class Database:
    """Base de données SQLite avec index FTS5 pour la recherche full-text."""

    def __init__(self, db_file: str = "data/crawler.db"):
        """
        Args:
            db_file: Chemin vers le fichier .db (créé automatiquement).
        """
        self.db_file = db_file
        self.conn: sqlite3.Connection | None = None
        self._initialize()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize(self) -> bool:
        """Crée les tables, triggers et index si nécessaire."""
        try:
            db_dir = os.path.dirname(self.db_file)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)

            self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            cur = self.conn.cursor()

            # Table principale
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pages (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    url          TEXT UNIQUE NOT NULL,
                    domain       TEXT NOT NULL,
                    title        TEXT,
                    content      TEXT,
                    content_hash TEXT,
                    depth        INTEGER,
                    status_code  INTEGER,
                    crawled_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    word_count   INTEGER,
                    link_count   INTEGER
                )
            """)

            # Migration : ajouter content_hash si absent (base existante)
            existing_cols = {row[1] for row in cur.execute("PRAGMA table_info(pages)")}
            if "content_hash" not in existing_cols:
                cur.execute("ALTER TABLE pages ADD COLUMN content_hash TEXT")

            # Table FTS5 (recherche full-text)
            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
                    url,
                    domain,
                    title,
                    content,
                    content='pages',
                    content_rowid='id',
                    tokenize='porter unicode61'
                )
            """)

            # Triggers de synchronisation FTS5
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS pages_ai AFTER INSERT ON pages BEGIN
                    INSERT INTO pages_fts(rowid, url, domain, title, content)
                    VALUES (new.id, new.url, new.domain, new.title, new.content);
                END
            """)
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS pages_ad AFTER DELETE ON pages BEGIN
                    DELETE FROM pages_fts WHERE rowid = old.id;
                END
            """)
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS pages_au AFTER UPDATE ON pages BEGIN
                    UPDATE pages_fts
                    SET url=new.url, domain=new.domain, title=new.title, content=new.content
                    WHERE rowid = new.id;
                END
            """)

            # Index
            cur.execute("CREATE INDEX IF NOT EXISTS idx_domain       ON pages(domain)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_crawled_at   ON pages(crawled_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_depth        ON pages(depth)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON pages(content_hash)")

            self.conn.commit()
            return True

        except sqlite3.Error as e:
            print(f"[DB ERROR] Initialisation : {e}")
            return False

    # ------------------------------------------------------------------
    # Écriture
    # ------------------------------------------------------------------

    def insert_page(self, url: str, title: str, content: str, content_hash: str,
                    depth: int, status_code: int, word_count: int,
                    link_count: int, commit: bool = True) -> bool:
        """
        Insère ou met à jour une page.

        Returns:
            True si l'opération a réussi.
        """
        try:
            domain = urlparse(url).netloc
            self.conn.execute("""
                INSERT INTO pages
                    (url, domain, title, content, content_hash, depth,
                     status_code, word_count, link_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    title        = excluded.title,
                    content      = excluded.content,
                    content_hash = excluded.content_hash,
                    depth        = excluded.depth,
                    status_code  = excluded.status_code,
                    crawled_at   = CURRENT_TIMESTAMP,
                    word_count   = excluded.word_count,
                    link_count   = excluded.link_count
            """, (url, domain, title, content, content_hash, depth,
                  status_code, word_count, link_count))
            if commit:
                self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"[DB ERROR] insert_page : {e}")
            return False

    def commit(self) -> bool:
        """Valide explicitement la transaction courante."""
        try:
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"[DB ERROR] commit : {e}")
            return False

    def hash_exists(self, content_hash: str) -> bool:
        """Vérifie si un contenu avec ce hash SHA-256 existe déjà."""
        try:
            row = self.conn.execute(
                "SELECT 1 FROM pages WHERE content_hash = ? LIMIT 1",
                (content_hash,)
            ).fetchone()
            return row is not None
        except sqlite3.Error:
            return False

    # ------------------------------------------------------------------
    # Lecture
    # ------------------------------------------------------------------

    def get_page_count(self) -> int:
        """Retourne le nombre total de pages stockées."""
        try:
            row = self.conn.execute("SELECT COUNT(*) FROM pages").fetchone()
            return row[0]
        except sqlite3.Error:
            return 0

    def export_jsonl(self, output_path: str) -> int:
        """
        Exporte toutes les pages vers un fichier JSONL.
        Chaque ligne : {"text": ..., "url": ..., "title": ..., "word_count": ...}

        Returns:
            Nombre de lignes exportées.
        """
        import json
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        count = 0
        try:
            rows = self.conn.execute(
                "SELECT url, title, content, word_count FROM pages ORDER BY id"
            ).fetchall()
            with open(output_path, "w", encoding="utf-8") as f:
                for row in rows:
                    record = {
                        "text": row["content"] or "",
                        "url": row["url"],
                        "title": row["title"] or "",
                        "word_count": row["word_count"] or 0,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
        except (sqlite3.Error, OSError) as e:
            print(f"[DB ERROR] export_jsonl : {e}")
        return count

    def search_full_text(self, query: str, limit: int = 20) -> list:
        """
        Recherche full-text via FTS5.

        Args:
            query: Requête texte (ex: "machine learning").
            limit: Nombre max de résultats.

        Returns:
            Liste de sqlite3.Row.
        """
        try:
            return self.conn.execute("""
                SELECT p.url, p.title, p.domain, p.word_count,
                       snippet(pages_fts, 3, '<b>', '</b>', '...', 30) AS snippet
                FROM pages p
                JOIN pages_fts ON pages_fts.rowid = p.id
                WHERE pages_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit)).fetchall()
        except sqlite3.Error as e:
            print(f"[DB ERROR] search_full_text : {e}")
            return []

    def get_statistics(self) -> sqlite3.Row | None:
        """Retourne des statistiques globales sur la base."""
        try:
            return self.conn.execute("""
                SELECT
                    COUNT(*)            AS total_pages,
                    COUNT(DISTINCT domain) AS unique_domains,
                    CAST(AVG(word_count) AS INTEGER) AS avg_words,
                    MAX(word_count)     AS max_words,
                    SUM(link_count)     AS total_links
                FROM pages
            """).fetchone()
        except sqlite3.Error as e:
            print(f"[DB ERROR] get_statistics : {e}")
            return None

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------

    def close(self):
        """Ferme la connexion."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
