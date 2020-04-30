class ActivationFunction:
    def reru(self, signal: float) -> float:
        output = signal if signal > 0 else 0
        return output
