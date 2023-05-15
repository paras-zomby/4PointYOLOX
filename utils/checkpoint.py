import torch
import os

def save_ckpt(epoch, model, optimizer, history, save_path):
    torch.save(
            {
                    'epoch':                epoch,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history data':         history,
                    }, os.path.join(save_path, f'epoch_{epoch:04d}.ckpt')
            )


def load_ckpt(model, optimizer, load_path, device):
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    history = checkpoint['history data']
    return model, optimizer, epoch, history
