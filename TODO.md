1. Améliorer les perf pour qu'il utilise moins de CPU -- 70% actuellement
2. Régler le bug C:\IA_Generatrice\training\train.py:152: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
3. Ajouter un système pour récupérer les performances du modèle, type perte,  - faire dans un csv
4. Ajouter un système de checkpoint pour pouvoir reprendre l'entraînement plus tard et un autre système qui sauvegarde l'état du modèle après la phase d'entrainement pour pouvoir l'utiliser un peu quand on veut
