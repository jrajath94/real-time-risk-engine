import numpy as np

class VaRCalculator:
    def __init__(self, confidence: float = 0.95, samples: int = 10000):
        self.confidence = confidence
        self.samples = samples
    
    def calculate(self, returns: np.ndarray) -> float:
        """Calculate Value at Risk."""
        mean = np.mean(returns)
        std = np.std(returns)
        simulated = np.random.normal(mean, std, self.samples)
        var = np.percentile(simulated, (1 - self.confidence) * 100)
        return var
    
    def calculate_cvar(self, returns: np.ndarray) -> float:
        """Calculate Conditional VaR."""
        var = self.calculate(returns)
        mean = np.mean(returns)
        std = np.std(returns)
        simulated = np.random.normal(mean, std, self.samples)
        cvar = np.mean(simulated[simulated <= var])
        return cvar
