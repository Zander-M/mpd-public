"""
    Accelerate Helper functions
"""

def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model