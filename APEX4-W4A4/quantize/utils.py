import torch
import torch.nn as nn
from quantize.matmul_had import *

def add_lora_to_linear_layer(linear_layer, r=8, lora_alpha=32, lora_dropout=0.0):
    """
    为 nn.Linear 层添加 LoRA 参数和功能。
    """
    if not isinstance(linear_layer, nn.Linear):
        raise ValueError("The layer is not a nn.Linear layer.")

    # 冻结原始权重
    linear_layer.weight.requires_grad = False
    if linear_layer.bias is not None:
        linear_layer.bias.requires_grad = False

    # LoRA 参数
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features
    linear_layer.lora_r = r
    linear_layer.lora_alpha = lora_alpha
    linear_layer.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
    linear_layer.scaling = lora_alpha / r

    # 初始化 LoRA 权重
    linear_layer.lora_A = nn.Parameter(torch.randn(r, in_features).float() * 0.0005)
    linear_layer.lora_B = nn.Parameter(torch.randn(out_features, r).float() * 0.001)

    # 修改 forward 方法
    def linear_forward(x):
        result = nn.functional.linear(x, linear_layer.weight, linear_layer.bias)
        if linear_layer.training:
            lora_out = linear_layer.lora_dropout(x) @ linear_layer.lora_A.T
            lora_out = lora_out @ linear_layer.lora_B.T
            result += linear_layer.scaling * lora_out
        return result

    linear_layer.forward = linear_forward

def add_lora_to_llama_decoder_layer(decoder_layer, r=256, lora_alpha=32, lora_dropout=0.0):
    """
    为 LlamaDecoderLayer 中的所有 nn.Linear 层添加 LoRA 功能。
    """
    for name, module in decoder_layer.named_modules():
        if isinstance(module, nn.Linear):
            add_lora_to_linear_layer(module, r, lora_alpha, lora_dropout)

def merge_lora_weights_in_linear_layer(linear_layer):
    """
    将 LoRA 权重合并到线性层的原始权重中，并移除 LoRA 参数。
    """
    if hasattr(linear_layer, 'lora_A') and hasattr(linear_layer, 'lora_B'):
        delta_weight = (linear_layer.scaling * (linear_layer.lora_B @ linear_layer.lora_A)).to(linear_layer.weight.device)
        linear_layer.weight.data += delta_weight

        # 移除 LoRA 参数
        del linear_layer.lora_A
        del linear_layer.lora_B
        del linear_layer.lora_r
        del linear_layer.lora_alpha
        del linear_layer.lora_dropout
        del linear_layer.scaling

        # 重新启用原始权重的梯度（如果需要进一步训练）
        linear_layer.weight.requires_grad = True
        if linear_layer.bias is not None:
            linear_layer.bias.requires_grad = True

        # 恢复原始的 forward 方法
        def linear_forward(x):
            return nn.functional.linear(x, linear_layer.weight, linear_layer.bias)
        linear_layer.forward = linear_forward

def merge_lora_weights_in_decoder_layer(decoder_layer):
    """
    将 LlamaDecoderLayer 中所有线性层的 LoRA 权重合并。
    """
    for module in decoder_layer.modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, 'lora_A'):
                merge_lora_weights_in_linear_layer(module)

def min_max_quantize(input_tensor, bits = 8, group=-1, asym=True, max_scale = torch.Tensor([4]), min_scale = torch.Tensor([4])):
    #assert qmin < qmax, "qmin should be less than qmax"
    shape=input_tensor.shape
    qmax = 2**bits -1
    qmin = 0
    
    if group != input_tensor.shape[1]:
        input_tensor = input_tensor.view(shape[0], -1, group)     #[Out, Inp] -> [-1, group]
    else:
        input_tensor = input_tensor.unsqueeze(1)

    #print(input_tensor.size())
    if asym:
        min_val = input_tensor.min(-1)[0].mul(torch.sigmoid(min_scale.t())).unsqueeze(-1)
        max_val = input_tensor.max(-1)[0].mul(torch.sigmoid(max_scale.t())).unsqueeze(-1)
        scale = ((max_val - min_val) / (qmax - qmin)).clamp(min=1e-8)
        zero_point = -min_val / scale.clamp(min=1e-8)
    else:
        max_val = input_tensor.abs().max(-1)[0].unsqueeze(-1)
        min_val = -max_val

        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = (qmax + 1)/2

    quantized_tensor = (input_tensor) / scale + zero_point
    
    quantized_tensor = quantized_tensor.clamp(qmin, qmax).round()# - quantized_tensor).detach() + quantized_tensor
        
    return quantized_tensor, scale, zero_point

def fht_matrix(n):
    """ Generate a Hadamard matrix of order n (n must be a power of 2) """
    if n == 1:
        return torch.tensor([[1.]])
    else:
        H_n_1 = fht_matrix(n // 2)
        return torch.cat([torch.cat([H_n_1, H_n_1], dim=1),
                          torch.cat([H_n_1, -H_n_1], dim=1)], dim=0)

def fast_hadamard_transform(x):
    """
    Perform the fast Hadamard transform on a batch of vectors.
    x should be a 2D tensor with shape (batch_size, n), where n is a power of 2.
    """
    n = x.size(1)
    H = fht_matrix(n).to(x.device)
    return torch.matmul(x, H)

def inverse_fast_hadamard_transform(y):
    """
    Perform the inverse fast Hadamard transform on a batch of vectors.
    y should be a 2D tensor with shape (batch_size, n), where n is a power of 2.
    """
    n = y.size(1)
    H = fht_matrix(n).to(y.device)
    return torch.matmul(y, H) / n
    
if __name__ == "__main__":
    # Example usage
    batch_size = 2048
    n = 8192  # n must be a power of 2
    x = torch.randn(batch_size, n).cuda()  # Create a random batch of vectors

    # Perform Fast Hadamard Transform
    y = fast_hadamard_transform(x)
    #y = matmul_hadU(x)
    #x_reconstructed = matmul_hadU(y, transpose=True)
    #print(matmul_hadU(x)-y)
    # Perform Inverse Fast Hadamard Transform
    x_reconstructed = inverse_fast_hadamard_transform(y)

    print(x)
    print(y)
    print(x_reconstructed - x)  # Display original, transformed, and reconstructed 
