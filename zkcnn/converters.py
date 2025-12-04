from einops import rearrange
import torch

def export_model(model, image, filename):
    """
    Exports the model weights and a single image to the format expected by zkCNN.
    """
    with open(filename, 'w') as f:
        # 1. Export Image
        # Image is (C, H, W). zkCNN reads C loops, then H, then W.
        # einops 'c h w -> (c h w)' flattens in C-major order which matches zkCNN
        img_np = rearrange(image.cpu().detach().numpy(), 'c h w -> (c h w)')
        f.write("\n".join(map(str, img_np)))
        f.write("\n")
        
        # 2. Export Weights and Biases
        # The order depends on the specific C++ model definition, 
        # but for LeNet/VGG in this repo it's generally sequential layer export.
        layers = [module for module in model.modules() if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))]
        
        for layer in layers:
            w_data = layer.weight.data.cpu().detach().numpy()
            if len(w_data.shape) == 4: # Conv2d
                # out, in, k1, k2 -> flatten
                weights = rearrange(w_data, 'o i k1 k2 -> (o i k1 k2)')
            elif len(w_data.shape) == 2: # Linear
                # out, in -> flatten
                weights = rearrange(w_data, 'o i -> (o i)')
            else:
                 weights = w_data.flatten()

            f.write("\n".join(map(str, weights)))
            f.write("\n")
            
            # Bias
            if layer.bias is not None:
                b_data = layer.bias.data.cpu().detach().numpy()
                # Bias is 1D (out), just flatten
                bias = rearrange(b_data, 'o -> (o)')
                f.write("\n".join(map(str, bias)))
                f.write("\n")
            else:
                # Warning: zkCNN usually expects bias. If missing, we might desync.
                print(f"Warning: Layer {layer} has no bias")
