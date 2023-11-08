import torch

def save_model(arch, model, save_path, **kwargs):
    torch.save({
        'arch': arch,
        'state_dict': model.state_dict(),
        **kwargs
    }, save_path)

def load_model(model_path):
    info = torch.load(model_path)
    arch = info['arch']
    model = arch()
    model.load_state_dict(info['state_dict'])
    return model, info