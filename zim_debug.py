# -*- coding: utf-8 -*-
"""zim_debug.py v2 — inspecte l'API réelle de libzim installée"""
import sys

try:
    import libzim.reader as zr
    import libzim
    # print(f"libzim version : {libzim.__version__}")
except ImportError:
    print("[ERREUR] libzim non installé")
    sys.exit(1)

path = sys.argv[1] if len(sys.argv) > 1 else None
if not path:
    print("Usage : python zim_debug.py fichier.zim")
    sys.exit(1)

archive = zr.Archive(path)
print(f"entry_count : {archive.entry_count}")

# ── Toutes les méthodes/attributs de Archive ──────────────────────────────
print("\n=== dir(archive) ===")
members = [m for m in dir(archive) if not m.startswith("__")]
for m in members:
    print(f"  {m}")

# ── Essayer toutes les méthodes d'accès par index ─────────────────────────
print("\n=== Test méthodes d'accès ===")
for method_name in ["get_entry_by_id", "_get_entry_by_id", "get_entry_by_path",
                    "get_entry_by_title", "get_article", "get_by_id", "entries"]:
    has = hasattr(archive, method_name)
    print(f"  {method_name:30s} : {'OUI' if has else 'non'}")

# ── Inspecter Entry et Item si on peut en obtenir un ─────────────────────
print("\n=== Tentative d'accès à la main entry ===")
try:
    main = archive.main_entry
    print(f"  main_entry.path  : {main.path!r}")
    print(f"  main_entry.title : {main.title!r}")
    print(f"  is_redirect      : {main.is_redirect}")
    print(f"  dir(main_entry)  : {[m for m in dir(main) if not m.startswith('__')]}")
    item = main.get_item()
    print(f"  item.mimetype    : {item.mimetype!r}")
    print(f"  dir(item)        : {[m for m in dir(item) if not m.startswith('__')]}")
except Exception as e:
    print(f"  ERREUR: {e}")

# ── Tester archive[0], archive[key], etc. ────────────────────────────────
print("\n=== Test archive[0] et archive.__getitem__ ===")
for expr in ["archive[0]", "archive['A/Main_Page']", "archive.get_entry_by_path('A/Main_Page')"]:
    try:
        result = eval(expr)
        print(f"  {expr} → {result}")
    except Exception as e:
        print(f"  {expr} → ERREUR: {e}")