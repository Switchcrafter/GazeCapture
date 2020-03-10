import torch
import torch.nn as nn

class MultiCriterion(nn.Module):
    def __init__(self, criteria, weights, reduction='mean'):
        super(MultiCriterion, self).__init__()
        self.weights = weights
        self.criteria  = criteria

        # Normalize weights here to sum upto 1
        self.weights = [float(w)/sum(self.weights) for w in self.weights]

    def forward(self, input, target):
        return sum([self.weights[i] * self.criteria[i](reduction='mean').forward(input, target) for i in range(len(self.criteria))])
    
    def backward(self, retain_graph):
        return sum([self.weights[i] * self.criteria[i](reduction='mean').backward(retain_graph=retain_graph) for i in range(len(self.criteria))])

# Custom loss functions
class LogCoshLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        value = torch.log(torch.cosh(ey_t + 1e-12))
        if self.reduction=='mean':
            return torch.mean(value)
        elif self.reduction=='sum':
            return torch.sum(value)

class TanhLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        value = ey_t * torch.tanh(ey_t)
        if self.reduction=='mean':
            return torch.mean(value)
        elif self.reduction=='sum':
            return torch.sum(value)

class SigmoidLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        value = 2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t
        if self.reduction=='mean':
            return torch.mean(value)
        elif self.reduction=='sum':
            return torch.sum(value)