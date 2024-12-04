import torch
import ctypes
import torch.nn as nn
from functools import partial
import argparse

import performance
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def test(test_shape, axis, test_dtype, eps, device):
    print(
        f"Testing Layernorm on {device} with test_shape:{test_shape}, axis:{axis} ,dtype:{test_dtype}, eps:{eps}"
    )
    ndim = len(test_shape)
    normlize_shape = []
    for i in range(axis, ndim):
        normlize_shape.append(test_shape[i])
    size = 1
    behindsize = 1
    for i in range(ndim):
        size *= test_shape[i]
        if (i >= axis):
            behindsize *= test_shape[i]
    input = torch.rand(test_shape, device=device, dtype=test_dtype, requires_grad=False)
    scale = torch.rand(normlize_shape, device=device, dtype=test_dtype, requires_grad=False)
    bias = torch.rand(normlize_shape, device=device, dtype=test_dtype, requires_grad=False)
    output = torch.rand(test_shape, device=device, dtype=test_dtype, requires_grad=False)

    input_ptr = ctypes.cast(input.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    scale_ptr = ctypes.cast(scale.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    bias_ptr = ctypes.cast(bias.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    layer_norm = nn.LayerNorm(normlize_shape, elementwise_affine=True, eps = eps)
    layer_norm.weight.data = scale
    layer_norm.bias.data = bias
    
    if test_dtype == torch.float32:
        if device == "cuda":
            torch_layernorm_time = performance.CudaProfile((layer_norm.forward, (input,)))  # 以毫秒为单位
            lib.layernorm_nv_f32.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_layernorm_time = \
            performance.CudaProfile((lib.layernorm_nv_f32, (input_ptr, scale_ptr, bias_ptr, output_ptr, eps, size, behindsize)))
        if device == "cpu":
            torch_layernorm_time = performance.CpuProfile((layer_norm.forward, (input,)))  # 以毫秒为单位
            lib.layernorm_cpu_f32.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_layernorm_time = \
            performance.CpuProfile((lib.layernorm_cpu_f32, (input_ptr, scale_ptr, bias_ptr, output_ptr, eps, size, behindsize)))
        if device == "mlu":
            torch_layernorm_time = performance.BangProfile((layer_norm.forward, (input,)))  # 以毫秒为单位
            '''
            lib.layernorm_bang_f32.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_layernorm_time = \
            performance.BangProfile((lib.layernorm_bang_f32, (input_ptr, scale_ptr, bias_ptr, output_ptr, eps, size, behindsize)))
            '''
            lib.layernorm_cnnl_f32.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float
            ]
            import numpy as np
            np_array = np.array(test_shape, dtype=np.int32)
            ctypes_array = np_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            custom_layernorm_time = \
            performance.BangProfile((lib.layernorm_cnnl_f32, (input_ptr, scale_ptr, bias_ptr, output_ptr, ctypes_array, ndim, axis, eps)))
            
    if test_dtype == torch.float16:
        if device == "cuda":
            torch_layernorm_time = performance.CudaProfile((layer_norm.forward, (input,)))  # 以毫秒为单位
            lib.layernorm_nv_f16.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_layernorm_time = \
            performance.CudaProfile((lib.layernorm_nv_f16, (input_ptr, scale_ptr, bias_ptr, output_ptr, eps, size, behindsize)))
        if device == "cpu":
            torch_layernorm_time = performance.CpuProfile((layer_norm.forward, (input,)))  # 以毫秒为单位
            lib.layernorm_cpu_f16.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_layernorm_time = \
            performance.CpuProfile((lib.layernorm_cpu_f16, (input_ptr, scale_ptr, bias_ptr, output_ptr, eps, size, behindsize)))
        if device == "mlu":
            torch_layernorm_time = performance.BangProfile((layer_norm.forward, (input,)))  # 以毫秒为单位
            lib.layernorm_bang_f16.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_layernorm_time = \
            performance.BangProfile((lib.layernorm_bang_f16, (input_ptr, scale_ptr, bias_ptr, output_ptr, eps, size, behindsize)))
    performance.logBenchmark(torch_layernorm_time, custom_layernorm_time)

    # 将结果转换回 PyTorch 张量以进行比较
    tmpa = layer_norm.forward(input).to('cpu').detach().numpy().flatten()
    
    tmpb = output.to('cpu').detach().numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))

# 解析命令行参数
parser = argparse.ArgumentParser(description="Test layernorm on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        # test_shape, axis, test_dtype, eps, device
        ((700, 1200, 24), 1, torch.float32, 1e-5, 'cuda'),
        ((700, 1200, 24), 0, torch.float32, 1e-5, 'cuda'),
        ((700, 1200, 24), 2, torch.float32, 1e-5, 'cuda'),

        ((700, 1200, 24), 1, torch.float16, 1e-5, 'cuda'),
        ((700, 1200, 24), 0, torch.float16, 1e-5, 'cuda'),
        ((700, 1200, 24), 2, torch.float16, 1e-5, 'cuda'),

        ((700, 1200), 1, torch.float32, 1e-5, 'mlu'),
        #((700, 1200, 24), 0, torch.float32, 1e-5, 'mlu'),
        ((700, 1200, 24), 2, torch.float32, 1e-5, 'mlu'),

        ((700, 1200, 24), 1, torch.float16, 1e-5, 'mlu'),
        #((700, 1200, 24), 0, torch.float16, 1e-5, 'mlu'),
        ((700, 1200, 24), 2, torch.float16, 1e-5, 'mlu'),

        ((7, 12, 24), 1, torch.float32, 1e-5, 'cpu'),
        ((7, 12, 24), 0, torch.float32, 1e-5, 'cpu'),
        ((7, 12, 24), 2, torch.float32, 1e-5, 'cpu'),

        ((7, 12, 24), 1, torch.float16, 1e-5, 'cpu'),
        ((7, 12, 24), 0, torch.float16, 1e-5, 'cpu'),
        ((7, 12, 24), 2, torch.float16, 1e-5, 'cpu'),
         
]
filtered_test_cases = [
    (test_shape,axis, test_dtype, eps, device)
    for test_shape, axis, test_dtype, eps, device in test_cases
    if device == args.device
]
if args.device == 'mlu':
    import torch_mlu
# 执行过滤后的测试用例
for test_shape, axis, test_dtype, eps, device in filtered_test_cases:
    test(test_shape, axis, test_dtype, eps, device)
