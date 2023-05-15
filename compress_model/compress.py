import torch
import nncf

from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args


def quantization_model(model, dataloader, config_path):
    # Load a configuration file to specify compression
    nncf_config = NNCFConfig.from_json(config_path)

    # Provide data loaders for compression algorithm initialization, if necessary
    nncf_config = register_default_init_args(nncf_config, dataloader)

    # Apply the specified compression algorithms to the model
    compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)

    # Now use compressed_model as a usual torch.nn.Module
    # to fine-tune compression parameters along with the model weights

    # ... the rest of the usual PyTorch-powered training pipeline

    # Export to ONNX or .pth when done fine-tuning
    compression_ctrl.export_model("compressed_model.onnx")
    torch.save(compressed_model.state_dict(), "compressed_model.pth")


def quantization_model_q(model_fp):
    import torch.quantization.quantize_fx as quantize_fx
    import copy

    # we need to deepcopy if we still want to keep model_fp unchanged after quantization since quantization apis change the input model
    model_to_quantize = copy.deepcopy(model_fp.eval())
    model_to_quantize.eval()
    qconfig_dict = {"": torch.quantization.default_dynamic_qconfig}
    # prepare
    model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_dict)
    # no calibration needed when we only have dynamici/weight_only quantization
    # quantize
    model_quantized = quantize_fx.convert_fx(model_prepared)
    return model_quantized
