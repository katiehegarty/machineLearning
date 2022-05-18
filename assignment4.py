import numpy as np
from PIL import Image


def part_i_a(array, kernel):
    k = len(kernel)
    n = len(array)

    output_array = np.zeros((n, n))
    for xn in range(k, n-k):
        for yn in range(k, n-k):
            result = 0
            for xk in range(k):
                for yk in range(k):
                    x = xk - (k // 2)
                    y = yk - (k // 2)
                    result += array[yn + y][xn + x] * kernel[yk][xk]
                output_array[yn][xn] = result
    return output_array


# part i b
image = Image.open('smile.jpg')
image.show()
rgb = np.array(image.convert('RGB'))
r = rgb[:, :, 0]
Image.fromarray(np.uint8(r)).show()
kernel1 = np.array([[-1, -1, -1,], [-1, 8, -1], [-1, -1, -1]])
kernel2 = np.array([[0, -1, 0], [-1, 8, -1], [0, -1, 0]])
convolution1 = part_i_a(r, kernel1)
convolution2 = part_i_a(r, kernel2)
image1 = Image.fromarray(np.uint8(convolution1))
image2 = Image.fromarray(np.uint8(convolution2))
image1.show()
image2.show()