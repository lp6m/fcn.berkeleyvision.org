
def calc_convsize(insize, padding, stride, kernel):
    tmp = insize + 2 * padding - kernel
    divisible = (tmp % stride == 0)
    outsize = tmp//stride + 1
    return outsize, divisible

def calc_deconvsize(insize, padding, stride, kernel):
    """
    https://qiita.com/kamata1729/items/41adf8b99a7ce070f79a
    """
    outsize = (insize - 1) * stride - 2 * padding + kernel
    return outsize

original_alexnet = [
    #name, padding, stride, kernel
    ['conv1', 100, 4, 11],
    ['maxpool1', 0, 2, 3],
    ['conv2', 2, 1, 5],
    ['maxpool2', 0, 2, 3],
    ['conv3', 1, 1, 3],
    ['conv4', 1, 1, 3],
    ['conv5', 1, 1, 3],
    ['maxpool3', 0, 2, 3],
    ['fc6', 0, 1, 6],
    ['fc7', 0, 1, 1],
    ['score_fr', 0, 1, 1],
    ['upscore', 0, 32, 63],
]

def check_alexnet_output_shape(insize, layers):
    print('input:', insize)
    currentsize = insize
    all_divisible = True
    for layer in layers:
        name, padding, stride, kernel = layer
        flag = None
        if 'maxpool' in name:
            #maxpool
            currentsize = currentsize // 2
        elif name != 'upscore':
            #conv
            currentsize, flag = calc_convsize(currentsize, padding, stride, kernel)
            all_divisible = (all_divisible & flag)
        else:
            #deconv
            currentsize = calc_deconvsize(currentsize, padding, stride, kernel)
        print(name, currentsize, flag)        
    print('all divisible:', all_divisible)
    return currentsize

def calculate_offsets(layers):
    """
    calculate offset size of last layer
    https://developer.nvidia.com/blog/image-segmentation-using-digits-5/
    """
    #calc scaling
    accum_scaling = {}
    current_scaling = 1
    for layer in layers:
        name, _, stride, kernel = layer
        if 'maxpool' in name:
            current_scaling *= 2
        else:
            current_scaling *= stride
        accum_scaling[name] = current_scaling if name != 'upscore' else 1
        
    #calc offset for each layer
    offsets = {}
    for layer in layers:
        name, padding, stride, kernel = layer
        offsets[name] = (padding - (kernel - 1) / 2) / stride if name != 'upscore' else ((kernel-1)/2 - padding) 
    #cal accum offset in reverse order
    accum_offsets = {}
    current_accum_offset = offsets['upscore']
    accum_offsets['upscore'] = offsets['upscore']
    for layer in list(reversed(layers))[1:]:
        name, _, _, _ = layer
        current_accum_offset += accum_scaling[name] * offsets[name]
        accum_offsets[name] = current_accum_offset
    #show results
    for layer in layers:
        name, _, _, _ = layer
        print('{} {} {}'.format(accum_scaling[name], offsets[name], accum_offsets[name]))
    
    return current_accum_offset


tiny_alexnet = [
    #name, padding, stride, kernel
    ['conv1', 36, 4, 11],
    ['maxpool1', 0, 2, 3],
    ['conv2', 2, 1, 5],
    ['maxpool2', 0, 2, 3],
    ['conv3', 1, 1, 3],
    ['conv4', 1, 1, 3],
    ['conv5', 1, 1, 3],
    ['maxpool3', 0, 2, 3],
    ['fc6', 0, 1, 3],
    ['fc7', 0, 1, 1],
    ['score_fr', 0, 1, 1],
    ['upscore', 0, 32, 63],
]

print('original alexnet')
check_alexnet_output_shape(224, original_alexnet)
print('----------------')
calculate_offsets(original_alexnet)
print('----------------')
print('tiny alexnet')
check_alexnet_output_shape(227, tiny_alexnet)
print('----------------')
calculate_offsets(tiny_alexnet)
