import numpy as np

def generate_random_data(input_shape, num_samples=10):
    if len(input_shape) == 3:  # 图像数据
        data = np.random.rand(num_samples, *input_shape).astype(np.float32)
    else:
        data = np.random.rand(num_samples, *input_shape).astype(np.float32)
    return data

#vgg16 10,3,224,224