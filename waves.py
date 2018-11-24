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
    grid_size = int(resolution / 32 + 1)

    KERNEL = f"""
    __global__ void mandel(float iterations[32*{grid_size}][32*{grid_size}]){{
        const int gid_x = threadIdx.x + blockIdx.x * blockDim.x;
        const int gid_y = threadIdx.y + blockIdx.y * blockDim.y;
    
        const double c_im = {zoom}*(__int2double_rn(gid_x)/(32*{grid_size})-0.5) + {center[1]};
        const double c_re = {zoom}*(__int2double_rn(gid_y)/(32*{grid_size})-0.5) + {center[0]};
        
        double iteration = 0;
            
        double z_im = 0.;
        double z_re = 0.;    
        double z_im2 = 0.;
        double z_re2 = 0.;    
        
        for(unsigned i=0; i<1000; ++i){{
            if((z_re2 + z_im2) < 400){{
            
                z_im2 = z_im*z_im;
                z_re2 = z_re*z_re;
                
                iteration += 1;
            
                z_im = 2 * z_re * z_im + c_im;
                z_re = z_re2 - z_im2 + c_re;
            }}
        }}
        
        if((z_re2 + z_im2) > 4){{
            iterations[gid_x][gid_y] = iteration + 1/log(2.) * log(log(400.)/(0.5*log(z_re2 + z_im2)));
        }} else {{
            iterations[gid_x][gid_y] = __int_as_float(0x7fffffff);
        }}
    }}
    """

    iterations = np.zeros((32 * grid_size, 32 * grid_size), dtype=np.float32)
    func = SourceModule(KERNEL).get_function("mandel")
    func(cuda.InOut(iterations), block=(32, 32, 1), grid=(grid_size, grid_size, 1))

    return iterations


center = np.array([-1, 0])
zoom = 1
iterations = mandel(center, zoom, 2048)


plt.matshow(iterations, cmap='seismic', extent=[center[0]-zoom, center[0]+zoom, center[1]-zoom, center[1]+zoom])

# plt.figure()
# plt.hist(iterations.ravel(), bins=100)
# plt.yscale('log', nonposy='clip')

plt.show()
