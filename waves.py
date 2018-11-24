import matplotlib.pyplot as plt

import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# TODO:
#  mandel(center & zoom)
#  interior_distance
#  exterior_distance

def mandel(center, zoom, resolution):
    grid_size = round(resolution / 32 + 0.5)

    KERNEL = f"""
    __global__ void mandel(float iterations[32*{grid_size}][32*{grid_size}]){{
        const int gid_x = threadIdx.x + blockIdx.x * blockDim.x;
        const int gid_y = threadIdx.y + blockIdx.y * blockDim.y;
    
        const float c_im = {zoom:.16f}*(__int2float_rn(gid_x)/(32*{grid_size})-0.5) + {center[1]:.16f};
        const float c_re = {zoom:.16f}*(__int2float_rn(gid_y)/(32*{grid_size})-0.5) + {center[0]:.16f};
        
        float iteration = 0;
            
        float z_im = 0.;
        float z_re = 0.;    
        float z_im2 = 0.;
        float z_re2 = 0.;    
        
        #pragma unroll
        for(unsigned i=0; i<10000; ++i){{
            if((z_re2 + z_im2) < 100){{
            
                z_im2 = z_im*z_im;
                z_re2 = z_re*z_re;
                
                iteration += 1;
            
                z_im = 2 * z_re * z_im + c_im;
                z_re = z_re2 - z_im2 + c_re;
            }}
        }}
        
        if((z_re2 + z_im2) > 4){{
            iterations[gid_x][gid_y] = sin(log(iteration + 1/log(2.) * log(log(100.)/(0.5*log(z_re2 + z_im2))))*5);
        }} else {{
            iterations[gid_x][gid_y] = __int_as_float(0x7fffffff);
        }}
    }}
    """

    iterations = np.zeros((32 * grid_size, 32 * grid_size), dtype=np.float32)
    func = SourceModule(KERNEL).get_function("mandel")
    func(cuda.InOut(iterations), block=(32, 32, 1), grid=(grid_size, grid_size, 1))

    return iterations


center = np.array([0.3113, 0.03528])
zoom = 0.001
iterations = mandel(center, zoom, 16*1024)


plt.matshow(iterations, cmap='hsv', extent=[center[0]-zoom/2, center[0]+zoom/2, center[1]-zoom/2, center[1]+zoom/2], origin='lower')

# plt.figure()
# plt.hist(iterations.ravel(), bins=100)
# plt.yscale('log', nonposy='clip')
plt.show()
