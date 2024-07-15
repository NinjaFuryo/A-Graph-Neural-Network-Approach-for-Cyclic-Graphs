class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001):
        """
        Initialise EarlyStopping avec une patience donnée (nombre d'époques sans amélioration avant d'arrêter)
        et une delta minimale pour considérer l'amélioration de la perte.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.stop_training = False

    def check_early_stop(self, val_loss):
        """
        Vérifie si l'arrêt anticipé doit être déclenché en fonction de la perte de validation.
        """
        # Vérifier si la perte de validation est meilleure que la meilleure perte connue, plus le min_delta
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.epochs_without_improvement = 0  # Réinitialise le compteur d'époques sans amélioration
        else:
            # Incrémenter le compteur d'époques sans amélioration
            self.epochs_without_improvement += 1

        # Si le nombre d'époques sans amélioration dépasse la patience, marquer que l'entraînement doit s'arrêter
        if self.epochs_without_improvement >= self.patience:
            self.stop_training = True

        # Renvoie True si l'entraînement doit s'arrêter, False sinon
        return self.stop_training
