import os
import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    """
    A modular trainer for supervised image classification.
    Designed for experiments involving loss landscape geometry,
    where clean separation of training, evaluation, and checkpointing
    is crucial.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        epochs=10,
        device="cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        # Cosine decay works well for landscape experiments
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        self.best_acc = 0.0


    def _train_one_epoch(self):
        self.model.train()

        total_loss = 0.0
        correct = 0
        seen = 0

        for batch, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += preds.eq(y).sum().item()
            seen += y.size(0)

        avg_loss = total_loss / seen
        accuracy = 100.0 * correct / seen

        return avg_loss, accuracy

    
    def _evaluate(self):
        self.model.eval()

        total_loss = 0.0
        correct = 0
        seen = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)

                total_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                correct += preds.eq(y).sum().item()
                seen += y.size(0)

        avg_loss = total_loss / seen
        accuracy = 100.0 * correct / seen

        return avg_loss, accuracy

    
    def _maybe_save(self, acc, save_path):
        if save_path is None:
            return

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.model.state_dict(), save_path)

    
    def fit(self, save_path=None):
        """
        Runs the full training loop.
        Returns: best validation accuracy.
        """

        for epoch in range(self.epochs):
            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc = self._evaluate()

            print(
                f"Epoch {epoch+1}/{self.epochs} "
                f"| Train Acc: {train_acc:.2f}% "
                f"| Val Acc: {val_acc:.2f}%"
            )

            self._maybe_save(val_acc, save_path)
            self.scheduler.step()

        return self.best_acc
