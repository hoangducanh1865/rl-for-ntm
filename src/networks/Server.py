import torch

class Server:
    def __init__(self, scale = 0.8):
        self.scale = scale
    
    def tuning_gradient (self, model):
        """Tuning the gradient of the model"""
        for param in model.parameters():
            param.grad *= self.scale
