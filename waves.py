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

        const double c_im = {zoom:.16f}*(__int2float_rn(gid_x)/(32*{grid_size})-0.5) + {center[1]:.16f};
        const double c_re = {zoom:.16f}*(__int2float_rn(gid_y)/(32*{grid_size})-0.5) + {center[0]:.16f};

        float iteration = 0;

        double z_im = 0.;
        double z_re = 0.;    
        double z_im2 = 0.;
        double z_re2 = 0.;    

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


def mandel_distance(center, zoom, resolution):
    grid_size = round(resolution / 32 + 0.5)
    KERNEL = f"""

    __device__ double cabs(double re, double im){{
        return sqrt(re*re + im*im);
    }}

    __global__ void mandel(float b[32*{grid_size}][32*{grid_size}]){{
        const int gid_x = threadIdx.x + blockIdx.x * blockDim.x;
        const int gid_y = threadIdx.y + blockIdx.y * blockDim.y;

        const double c_im = {zoom:.16f}*(__int2float_rn(gid_x)/(32*{grid_size})-0.5) + {center[1]:.16f};
        const double c_re = {zoom:.16f}*(__int2float_rn(gid_y)/(32*{grid_size})-0.5) + {center[0]:.16f};

        double dz_im = 0;
        double dz_re = 1.;

        double _dz_im = 0;
        double _dz_re = 1.;
          
        double z_im = 0;
        double z_re = 0;    
        double z_im2 = 0.;
        double z_re2 = 0.;    

        for(unsigned i=0; i<1000; ++i){{
            if(z_im2 + z_re2 < 1000){{
                _dz_im = 2*(z_im*dz_re + z_re*dz_im);
                _dz_re = 2*(z_re*dz_re - z_im*dz_im)+1;
                
                dz_re = _dz_re;
                dz_im = _dz_im;
    
                z_im2 = z_im*z_im;
                z_re2 = z_re*z_re;
    
                z_im = 2 * z_re * z_im + c_im;
                z_re = z_re2 - z_im2 + c_re;
            }}
        }}
        if(z_im2 + z_re2 > 1000){{
            b[gid_x][gid_y] = log(2 * cabs(z_re, z_im)*log(cabs(z_re, z_im))/cabs(dz_re, dz_im));
        }} else {{
            b[gid_x][gid_y] = __int_as_float(0x7fffffff);
        }}
    }}
    """


    b = np.zeros((32 * grid_size, 32 * grid_size), dtype=np.float32)
    func = SourceModule(KERNEL).get_function("mandel")
    func(cuda.InOut(b), block=(32, 32, 1), grid=(grid_size, grid_size, 1))

    return b


zoom = 0.01
center = np.array([-1.1797, 0.31])
iterations = mandel_distance(center, zoom, 4*1024)


plt.matshow(iterations, cmap='prism', extent=[center[0]-zoom/2, center[0]+zoom/2, center[1]-zoom/2, center[1]+zoom/2], origin='lower')

# plt.figure()
# plt.hist(iterations.ravel(), bins=100)
# plt.yscale('log', nonposy='clip')
plt.show()
