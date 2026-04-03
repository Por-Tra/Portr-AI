3. Ajouter un système pour récupérer les performances du modèle, type perte,  - faire dans un csv

Ajouter des input pour paramètré le système dans le main.py
 
 
 -------

Opti + correction
attention.py -> Pb de génération 
main.py -> Le cache binaire écrit avec array("I") (uint32) est lu via torch.from_file(..., dtype=torch.int32). nn.Embedding attend du torch.long (int64). TextDataset convertit bien les tenseurs déjà long — mais ce chemin teste isinstance(x, torch.Tensor) et vérifie x.dtype != torch.long, ce qui déclenchera une conversion lazy à chaque __getitem__ au lieu d'une conversion propre en amont.
generate.py -> La fonction remplace </w> par "" puis substitue ▁ par espace. Mais les sous-mots BPE intermédiaires (ex: "bonne" → ["bo", "nne"]) perdent leur marqueur de fin et sont concaténés sans espace avec le mot suivant. Le décodage produit du texte sans espaces entre les mots.

gpt_model.py -> À chaque step, K et V sont concaténés → le cache grandit jusqu'à seq_length. Mais il n'y a pas de vérification que T_full ≤ seq_length. Si max_length = seq_length, le cache déborde dès le premier token généré au-delà du prefill.

transformer_block.py -> fc1 et fc2 utilisent le même std=0.02 pour l'init, alors que GPT-2 scale uniquement fc2 (projection résiduelle). De plus, GPT._apply_scaled_init() ré-initialise fc2 avec le bon std, mais MLP.__init__ l'a déjà écrasé avec std=0.02 → double initialisation, ordre dépendant.

optimizer.py -> "fused" in torch.optim.AdamW.__init__.__doc__ est non documenté et peut casser sur des builds custom. Utiliser inspect.signature() ou un try/except est plus robuste.

main.py -> Pour le français, 30k tokens est le minimum raisonnable (CamemBERT utilise 32k, LLaMA-fr 64k). Avec LIMIT_CORPUS = VOCAB_SIZE × 10 = 300k lignes, le tokenizer verra suffisamment de données — mais avec une fréquence min_pair_freq=2, beaucoup de paires rares seront fusionnées trop tôt. Recommandé : min_pair_freq=5 pour un vocab de 30k.

main.py -> En LOW_RAM_MODE, RandomSampler avec replacement=True peut tirer plusieurs fois le même batch dans une epoch, et ne jamais voir certains tokens. Sur 500k tokens c'est acceptable, mais sur un petit corpus ça crée un biais fort. Préférer replacement=False ou un shuffle standard.

train.py -> Lors d'une reprise de checkpoint, le scheduler est chargé avec son état passé, mais total_steps est recalculé à partir de l'epoch 0. Si EPOCHS change entre deux runs, le scheduler cosine sera désynchro et le LR sera incorrect pour les epochs restantes.


combinaison de pb qui ralentissent le model

Mise à jour du srcipt qui traite les données -> pas assez précis et ne fait pas les répétitions 

Supression de la doc de merde