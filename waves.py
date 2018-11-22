import matplotlib.pyplot as plt

import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

r_min = -2.5
r_max = 1
i_min = -1
i_max = 1

x_res = 1024


KERNEL = """
__global__ void mandel(float iterations[32*64][32*64]){
    const int gid_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int gid_y = threadIdx.y + blockIdx.y * blockDim.y;

    const double c_im = __int2double_rn(gid_x)/(32.*512.*5120000000000)-0.099871398;
    const double c_re = __int2double_rn(gid_y)/(32.*512.*5120000000000)-0.749171400101;
    
    float iteration = 0;
        
    double z_im = 0.;
    double z_re = 0.;    
    double z_im2 = 0.;
    double z_re2 = 0.;    
    
    for(unsigned i=0; i<1000; ++i){
        if((z_re2 + z_im2) < 400){
        
            z_im2 = z_im*z_im;
            z_re2 = z_re*z_re;
            
            iteration += 1;
        
            z_im = 2 * z_re * z_im + c_im;
            z_re = z_re2 - z_im2 + c_re;
        }
    }
    
    if((z_re2 + z_im2) > 400){
        iterations[gid_x][gid_y] = iteration + 1/log(2.) * log(log(400.)/(0.5*log(z_re2 + z_im2)));
    } else {
        iterations[gid_x][gid_y] = __int_as_float(0x7fffffff);
    }
}
"""

mandel = SourceModule(KERNEL).get_function("mandel")

iterations = np.zeros((32*64, 32*64), dtype=np.float32)
print(iterations.nbytes)

mandel(cuda.InOut(iterations), block=(32, 32, 1), grid=(64, 64, 1))

iterations = iterations

plt.matshow(iterations, cmap='flag')

# plt.figure()
# plt.hist(iterations.ravel(), bins=100)
# plt.yscale('log', nonposy='clip')

plt.show()
