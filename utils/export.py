import torch
from MyYOLOX import YOLOX


def export_model(model, model_path, output_path, input_size, device):
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    dummy_input = torch.randn(1, 3, *input_size, device=device)
    torch.onnx.export(
            model, dummy_input, output_path, verbose=True, export_params=True,
            input_names=['model_input'], output_names=['model_output'], opset_version=12
            )


if __name__ == '__main__':
    net = YOLOX(num_class=24)
    export_model(
            net, "../pick_up_models/new_v2.3_210.pt", "../model.onnx", (480, 640),
            torch.device("cuda")
            )
