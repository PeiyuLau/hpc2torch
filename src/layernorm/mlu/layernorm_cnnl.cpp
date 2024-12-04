#include "cnnl.h"
#include "cnnl_extra.h"
#include <vector>
#include <iostream>
#include <stdexcept>

template<typename T>
void layernormCnnlDevice(void const *source, void const *weight, void const *bias, void *destination, int *shape, int nDim, int axis, float eps, cnnlHandle_t &handle, cnrtQueue_t &queue)
{
    if(axis<=0){
     throw std::invalid_argument("Error: Expected axis > 0, but received axis: " + std::to_string(axis));
    }
    cnnlTensorDescriptor_t yDesc, xDesc, filter_bias_desc, mean_rstd_desc;
    cnnlCreateTensorDescriptor(&yDesc);
    cnnlCreateTensorDescriptor(&xDesc);
    cnnlCreateTensorDescriptor(&filter_bias_desc);
    cnnlCreateTensorDescriptor(&mean_rstd_desc);

    std::vector<int> inDim(nDim);
    std::vector<int> outDim(nDim);
    std::vector<int> filter_biasDim(nDim - axis);
    std::vector<int> mean_rstdDim(axis);
    size_t mean_rstd_size=1;
    for (int i = 0; i < nDim; i++) {
        inDim[i] = shape[i];
        outDim[i] = shape[i];
        if(i >= axis){
            filter_biasDim[i - axis] = shape[i];            
        }
        else{
            mean_rstdDim[i] = shape[i];
            mean_rstd_size *= shape[i];
        }
    }
   
    size_t dtype_size = 0;
    if(sizeof(T) == 2){
        cnnlSetTensorDescriptor(
            xDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
            inDim.size(), inDim.data());
        cnnlSetTensorDescriptor(
            yDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
            outDim.size(), outDim.data());
        cnnlSetTensorDescriptor(
            filter_bias_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
            filter_biasDim.size(), filter_biasDim.data());
        cnnlSetTensorDescriptor(
            mean_rstd_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
            mean_rstdDim.size(), mean_rstdDim.data());
        cnnlGetSizeOfDataType(CNNL_DTYPE_HALF, &dtype_size);
       
    }
    else if(sizeof(T) == 4){
        cnnlSetTensorDescriptor(
            xDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
            inDim.size(), inDim.data());
        cnnlSetTensorDescriptor(
            yDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
            outDim.size(), outDim.data());
        cnnlSetTensorDescriptor(
            filter_bias_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
            filter_biasDim.size(), filter_biasDim.data());
        cnnlSetTensorDescriptor(
            mean_rstd_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
            mean_rstdDim.size(), mean_rstdDim.data());
        cnnlGetSizeOfDataType(CNNL_DTYPE_FLOAT, &dtype_size);
        
    }
   
    T *mean_dev, *rstd_dev;
    size_t size_mean_rstd = (size_t)mean_rstd_size * dtype_size;
    
    CNRT_CHECK(cnrtMalloc((void **)&mean_dev, size_mean_rstd));
    CNRT_CHECK(cnrtMalloc((void **)&rstd_dev, size_mean_rstd));

    size_t wsSize;
    cnnlGetLayerNormOpWorkspaceSize(handle, axis, xDesc, &wsSize);

    void *workspace;
    cnrtMalloc(&workspace, wsSize);
    cnnlLayerNormForward(handle,
                        xDesc,
                        source,
                        axis,
                        filter_bias_desc,
                        weight,
                        bias,
                        eps,
                        workspace,
                        wsSize,
                        yDesc,
                        destination,
                        mean_rstd_desc,
                        mean_dev,
                        rstd_dev);
    

    CNRT_CHECK(cnrtQueueSync(queue));

    cnrtFree(workspace);
    
    cnnlDestroyTensorDescriptor(xDesc);
    cnnlDestroyTensorDescriptor(yDesc);
    cnnlDestroyTensorDescriptor(filter_bias_desc);
    cnnlDestroyTensorDescriptor(mean_rstd_desc);
}
template<typename T>
void layernormCnnl(void const *input, void const *scale, void const *bias, void *output, int *shape, int nDim, int axis, float eps)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。

    layernormCnnlDevice<T>(input, scale, bias, output, shape, nDim, axis, eps, handle, queue);
    
    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));

    
}
extern "C" void layernorm_cnnl_f32(void const *input, void const *scale, void const *bias, void *output, int *shape, int nDim, int axis, float eps){
    layernormCnnl<float>(input, scale, bias, output, shape, nDim, axis, eps);
}
extern "C" void layernorm_cnnl_f16(void const *input, void const *scale, void const *bias, void *output, int *shape, int nDim, int axis, float eps){
    layernormCnnl<uint16_t>(input, scale, bias, output, shape, nDim, axis, eps);
}
