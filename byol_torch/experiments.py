"""
Experimentation with BYOL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearClassifier(nn.Module):
    """
    Experimentation with linear layers
    """
    def __init__(self, encoder, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.encoder = encoder
    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x)
        return x

"""
TODO
"""
class KNNClassifier(nn.Module):
    """
    Experimentation with KNN
    """
    def __init__(self, encoder, input_dim, output_dim):
        super().__init__()
        self.encoder = encoder
        self.input_dim = input_dim
        self.output_dim = output_dim
    def forward(self, x):
        x = self.encoder(x)
        return x
    def train(self, x, y):
        pass
    def predict(self, x):
        pass

"""
Given an encoder, train a linear classifier on top of it for a given number of epochs
then evaluate the performance of the linear classifier on the given test set
"""
class LinearExperimentationRegime:
    def __init__(self, encoder, input_dim, output_dim, train_loader, test_loader, epochs=80, lr=1e-3):
        self.classifier = LinearClassifer(encoder, input_dim, output_dim)
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
    
    def train(self):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            for x, y in self.train_loader:
                optimizer.zero_grad()
                logits = self.classifier(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
    
    def evaluate(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                logits = self.classifier(x)
                _, predicted = torch.max(logits.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total
    
    def run(self):
        self.train()
        return self.evaluate()