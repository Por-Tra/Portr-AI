# Web - Documentation + Algorithm Studio

Ce dossier contient un site web local autonome avec 2 parties:

1. Documentation technique complete
2. Visualisation pedagogique des algorithmes (style visualizer pas-a-pas)

Tout est confine dans ce dossier `web/`.
Aucun code existant du projet principal n est modifie.

## Fichiers

- index.html: structure de la page
- styles.css: style responsive desktop/mobile
- app.js: logique docs + moteur de visualisation pas-a-pas

## Partie 1 - Documentation

La partie documentation inclut:

- Index des modules
- Recherche plein texte (module, symbole, methode)
- Fiche detaillee de chaque classe
- Liste complete des methodes par classe
- Documentation de chaque methode (signature + description)

Modules/classes documentes:

- model/attention.py -> Attention
- model/transformer_block.py -> MLP, TransformerBlock
- model/embedding.py -> Embedding
- model/gpt_model.py -> GPT, GPTConfig
- data/DataBase/database.py -> Database
- data/dataset.py -> TextDataset
- tokenizer/tokenizer.py -> Tokenizer

## Partie 2 - Algorithm Studio

Visualisation pedagogique pas-a-pas avec:

- Pseudo-code
- Timeline d execution
- Step-by-step / Play / Pause / Reset
- Vitesse configurable
- Etat courant visualise

Algorithmes disponibles:

1) Attention (QK^T + masque causal + softmax)
- Parametres: tokens, heads, temperature, causal mask
- Visualisation: heatmap des scores/poids

2) TextDataset (fenetre glissante)
- Parametres: token IDs, seq_length
- Visualisation: paires (x,y) par index

3) MLP (Linear -> activation -> Linear)
- Parametres: vecteur entree, hidden multiplier, activation
- Visualisation: barres entree/sortie par etape

4) GPT Generate (boucle autoregressive)
- Parametres: seed tokens, max_length, vocab_size, top_k
- Visualisation: contexte, top-k logits, token ajoute

5) BPE Merge (Tokenizer)
- Parametres: texte, nombre de merges
- Visualisation: paire la plus frequente + mise a jour word_freq

## Data / DB Inspector

Le studio inclut aussi un inspecteur de donnees JSONL:

- Charge un fichier JSONL
- Ou colle du JSONL brut
- Affiche stats (records, avg words, top domains)
- Affiche preview de lignes

Pour lier la DB du projet:

1. Exporter depuis la DB existante avec `Database.export_jsonl(output_path)`
2. Charger le JSONL dans le panneau "Data / DB Inspector"

## Lancement

Option simple:
- Ouvrir `index.html` dans un navigateur.

Option recommandee:
- Servir `web/` via un serveur statique local puis ouvrir l URL locale.

## Notes techniques

- Site 100% front-end (HTML/CSS/JS)
- Aucune dependance npm
- Pas de backend impose
- Compatible usage local
