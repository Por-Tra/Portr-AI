
# OPTIMISER TOUT LE CODE
* Prioritaire:
    - Tokenizer


Réparer l'embeding:

    Traceback (most recent call last):
  File "C:\IA_Generatrice\main.py", line 143, in <module>
    main()
  File "C:\IA_Generatrice\main.py", line 117, in main
    embedding = embedding.Embedding(td.VOCAB_SIZE, SEQ_LENGTH, 256)
                ^^^^^^^^^
UnboundLocalError: cannot access local variable 'embedding' where it is not associated with a value
((.venv) ) 




Dataset texte
Tokenisation (BPE)
Construction du dataset
Embedding
Position encoding
Self-attention
Multi-head attention
Transformer block
Modèle GPT
Fonction de perte
Optimisation (AdamW)
Entraînement
Génération de texte
Scaling et optimisation