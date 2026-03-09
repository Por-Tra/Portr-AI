# Cahier des charges -- Créer une IA génératrice de texte type GPT (from scratch)

Auteur : Lucas\
Objectif : comprendre et implémenter une IA génératrice de texte moderne
en Python, sans API externes, en maîtrisant chaque étape.

------------------------------------------------------------------------

# 1. Objectif global du projet

Créer un **modèle de langage auto‑régressif** capable de prédire le
prochain token d'une phrase.

Pipeline complet :

Corpus texte → Nettoyage → Tokenisation → Dataset → Embedding →
Transformer (attention) → Entraînement → Génération de texte

------------------------------------------------------------------------

# 2. Prérequis théoriques

Domaines à comprendre :

Algèbre linéaire - vecteurs - matrices - produit matriciel - produit
scalaire

Probabilités - distribution - softmax - log likelihood

Deep Learning - gradient - backpropagation - descente de gradient

------------------------------------------------------------------------

# 3. Bibliothèques Python autorisées

Bibliothèques recommandées :

numpy → calcul matriciel

torch → autograd → GPU → tensor operations

tokenizers (optionnel) → tokenisation rapide

matplotlib → visualisation

tqdm → progression entraînement

datasets (optionnel) → charger corpus

Important : Ne pas utiliser directement des modèles LLM pré‑entraînés.

------------------------------------------------------------------------

# 4. Architecture générale du projet

Structure recommandée :

project/

data/ dataset.py

tokenizer/ tokenizer.py

model/ embedding.py attention.py transformer_block.py gpt_model.py

training/ train.py loss.py optimizer.py

inference/ generate.py

utils/ config.py

------------------------------------------------------------------------

# 5. Étape 1 --- Dataset texte

But : récupérer un corpus.

Sources possibles :

Wikipedia dump\
OpenWebText\
Project Gutenberg\
Common Crawl

Étapes :

1 nettoyer le texte 2 retirer HTML 3 normaliser 4 transformer en corpus brut

------------------------------------------------------------------------

# 6. Étape 2 --- Tokenisation

But : transformer le texte en nombres.

Exemple :

"bonjour comment ça va"

→

\[154, 892, 45, 991\]

Algorithmes possibles :

Word Tokenization

Byte Pair Encoding (BPE)
https://sebastianraschka.com/blog/2025/bpe-from-scratch.html
https://youtu.be/VrZTHO1E76w?si=8NGodlWKTbPC16Uu

Unigram LM

SentencePiece

Recommandé : BPE

Concept :

vocabulaire initial = caractères

fusion progressive des tokens fréquents

------------------------------------------------------------------------

# 7. Étape 3 --- Construction du Dataset

Créer des séquences d'entraînement.

Exemple :

Input : "je mange une"

Target : "mange une pomme"

Approche GPT :

séquence glissante

texte : "je mange une pomme rouge"

input : je mange une pomme

target : mange une pomme rouge

------------------------------------------------------------------------

# 8. Étape 4 --- Embedding

But : transformer un token en vecteur dense.

Dimension typique :

128 256 512 768

Formule :

embedding = table\[token_id\]

------------------------------------------------------------------------

# 9. Étape 5 --- Position Encoding

Le transformer ne comprend pas l'ordre.

Solution : ajouter une information de position.

Méthodes :

Sinusoidal positional encoding

Learned positional embedding

Formule sinus :

PE(pos,2i)=sin(pos/10000\^(2i/d))

PE(pos,2i+1)=cos(pos/10000\^(2i/d))

------------------------------------------------------------------------

# 10. Étape 6 --- Self Attention

Algorithme central des transformers.

Calcul :

Q = XWq K = XWk V = XWv

Attention :

softmax(QKᵀ / sqrt(d)) V

Concept :

chaque token regarde les autres tokens.

------------------------------------------------------------------------

# 11. Étape 7 --- Multi Head Attention

Plusieurs attentions en parallèle.

heads typiques :

4 8 16

But : apprendre plusieurs relations.

------------------------------------------------------------------------

# 12. Étape 8 --- Transformer Block

Structure :

LayerNorm MultiHeadAttention Residual LayerNorm FeedForward Residual

Feed Forward :

FFN(x) = max(0, xW1+b1)W2+b2

------------------------------------------------------------------------

# 13. Étape 9 --- Modèle GPT complet

Architecture :

Embedding + Positional encoding + N Transformer Blocks + Linear head +
Softmax

Sortie :

probabilité de chaque token.

------------------------------------------------------------------------

# 14. Étape 10 --- Fonction de perte

Cross Entropy Loss

Formule :

L = - log(p(token_correct))

------------------------------------------------------------------------

# 15. Étape 11 --- Optimisation

Optimiseur recommandé :

AdamW

Hyperparamètres typiques :

learning_rate = 3e-4

batch_size = 32

sequence_length = 256

------------------------------------------------------------------------

# 16. Étape 12 --- Entraînement

Boucle :

for batch in dataset

forward

loss

backward

update weights

------------------------------------------------------------------------

# 17. Étape 13 --- Génération de texte

Algorithme :

1 prendre prompt 2 prédire prochain token 3 ajouter token 4 recommencer

Stratégies :

Greedy

Top-k

Top-p (nucleus sampling)

Temperature

------------------------------------------------------------------------

# 18. Étape 14 --- Optimisations

techniques avancées :

Flash Attention

Gradient Checkpointing

Mixed Precision

KV Cache

------------------------------------------------------------------------

# 19. Étape 15 --- Évaluation

Mesures :

Perplexity

BLEU score

Human evaluation

------------------------------------------------------------------------

# 20. Étape 16 --- Scaling

Augmenter :

taille modèle nombre de couches taille dataset

loi empirique :

plus de données = meilleur modèle

------------------------------------------------------------------------

# 21. Roadmap de développement

Phase 1

Tokenizer BPE

Phase 2

Dataset loader

Phase 3

Embedding

Phase 4

Self attention

Phase 5

Transformer block

Phase 6

GPT model

Phase 7

Training

Phase 8

Text generation

------------------------------------------------------------------------

# 22. Objectif réaliste

Pour un premier modèle :

vocab_size = 8000 embedding = 256 layers = 4 heads = 4 sequence = 128

dataset \~ 50MB texte

GPU recommandé.

------------------------------------------------------------------------

# 23. Ressources clés à étudier

Articles :

Attention Is All You Need

GPT‑2 Paper

The Illustrated Transformer

------------------------------------------------------------------------

# 24. Conseils

Commencer par un mini modèle.

Tester chaque composant séparément.

Visualiser l'attention.

Ne pas essayer GPT‑3 directement.
