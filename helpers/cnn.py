
import numpy as np

def convolutional(x, kernels):

    # Get kernel height and width
    kh, kw = kernels[0].shape[0], kernels[0].shape[1]

    all_results = []
    all_patches = []
    all_relu_mask = []
    
    for kernel in kernels:
        feature_map = []
        feature_mask = []
        kernel_patches = []

        # Loop through each patches available in provided input
        for i in range(x.shape[0] - kh + 1):
            row = []
            row_mask = []
            for j in range(x.shape[1] - kw + 1):

                # Get a patch same size as kernel
                patch = x[i:i+kh, j:j+kw]
                kernel_patches.append(patch)
                
                # Compute dot results of two matrixes
                dot_sum = np.sum(patch * kernel)
                relu = dot_sum if dot_sum > 0 else 0.01 * dot_sum

                # Store a Leaky ReLU mask value (slope) to use in backprop
                mask = 1.0 if dot_sum > 0 else 0.01

                row.append(relu)
                row_mask.append(mask)
            
            # Append row to the current feature map
            feature_map.append(row)
            feature_mask.append(row_mask)
        
        # Append feature map produced by kernel
        all_results.append(np.array(feature_map))
        all_relu_mask.append(np.array(feature_mask))
        all_patches.append(kernel_patches)
    
    return np.array(all_results), all_patches, np.array(all_relu_mask)


def backpropagation_cnn(grad_from_next, patches, relu_mask, kernels, learning_rate):
    for k in range(len(kernels)):
        kernel_grad = np.zeros_like(kernels[k])

        feature_grad = grad_from_next[k]
        mask = relu_mask[k]
        kernel_patches = patches[k]

        patch_idx = 0
        for i in range(feature_grad.shape[0]):
            for j in range(feature_grad.shape[1]):
                # Get gradient from next layer
                grad_val = feature_grad[i, j]

                # Apply Leaky ReLU derivative
                grad_val *= mask[i, j]

                patch = kernel_patches[patch_idx]

                kernel_grad += grad_val * patch
                patch_idx += 1

        # Update kernel values
        kernels[k] -= learning_rate * kernel_grad

    return kernels




def mean_pooling(feature_maps, pool_size = (2, 2), stride = (1, 1)):
    
    results = []

    for fmap in feature_maps:
        result_map = []

    # Loop through each patches available in provided input
        for i in range(0, fmap.shape[0] - pool_size[0] + 1, stride[0]):
            row = []
            for j in range(0, fmap.shape[1] - pool_size[1] + 1, stride[1]):

                # Get a patch same size as kernel
                patch = fmap[i:i+pool_size[0], j:j+pool_size[1]]
                mean_pool = np.average(patch)
                row.append(mean_pool)
            result_map.append(row)
        
        results.append(np.array(result_map))
    
    return np.array(results)


def max_pooling(feature_maps, pool_size = (2, 2), stride = (1, 1)):
    
    results = []

    for fmap in feature_maps:
        result_map = []

    # Loop through each patches available in provided input
        for i in range(0, fmap.shape[0] - pool_size[0] + 1, stride[0]):
            row = []
            for j in range(0, fmap.shape[1] - pool_size[1] + 1, stride[1]):

                # Get a patch same size as kernel
                patch = fmap[i:i+pool_size[0], j:j+pool_size[1]]
                mean_pool = np.max(patch)
                row.append(mean_pool)
            result_map.append(row)
        
        results.append(np.array(result_map))
    
    return np.array(results)