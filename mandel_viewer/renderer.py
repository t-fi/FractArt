import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

RESOLUTION = 16*1024

KERNELS = """
#define precision double
#define resolution 16*1024

__device__ precision cabs(precision re, precision im){
        return sqrt(re*re + im*im);
}

__global__ void exterior_distance(unsigned char image[resolution][resolution][3],
                                  const double center_im, const double center_re, const double zoom){
    const int gid_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int gid_y = threadIdx.y + blockIdx.y * blockDim.y;

    const precision c_im = zoom*(__int2float_rn(gid_x)/resolution-0.5) + center_im;
    const precision c_re = zoom*(__int2float_rn(gid_y)/resolution-0.5) + center_re;

    precision dz_im = 0;
    precision dz_re = 1.;

    precision _dz_im = 0;
    precision _dz_re = 1.;
      
    precision z_im = 0;
    precision z_re = 0;    
    precision z_im2 = 0.;
    precision z_re2 = 0.;    

    for(unsigned i=0; i<10000; ++i){
        if(z_im2 + z_re2 < 1000){
            _dz_im = 2*(z_im*dz_re + z_re*dz_im);
            _dz_re = 2*(z_re*dz_re - z_im*dz_im)+1;
            
            dz_re = _dz_re;
            dz_im = _dz_im;

            z_im2 = z_im*z_im;
            z_re2 = z_re*z_re;

            z_im = 2 * z_re * z_im + c_im;
            z_re = z_re2 - z_im2 + c_re;
        }
    }
    if(z_im2 + z_re2 > 1000){
        precision b = log(2 * cabs(z_re, z_im)*log(cabs(z_re, z_im))/cabs(dz_re, dz_im));
        image[gid_y][gid_x][0] = (unsigned char)(126*(1+sin(2*b+1)));
        image[gid_y][gid_x][1] = (unsigned char)(126*(1+sin(3*b+2)));
        image[gid_y][gid_x][2] = (unsigned char)(126*(1+sin(4*b+3)));
    } else {
        image[gid_y][gid_x][0] = 0;
        image[gid_y][gid_x][1] = 0;
        image[gid_y][gid_x][2] = 0;
    }
}

__global__ void escape_time(unsigned char image[resolution][resolution][3],
                            const double center_im, const double center_re, const double zoom){
    const int gid_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int gid_y = threadIdx.y + blockIdx.y * blockDim.y;

    const precision c_im = zoom*(__int2float_rn(gid_x)/resolution-0.5) + center_im;
    const precision c_re = zoom*(__int2float_rn(gid_y)/resolution-0.5) + center_re;

    float iteration = 0;

    precision z_im = 0.;
    precision z_re = 0.;    
    precision z_im2 = 0.;
    precision z_re2 = 0.;    

    for(unsigned i=0; i<50000; ++i){
        if((z_re2 + z_im2) < 100){

            z_im2 = z_im*z_im;
            z_re2 = z_re*z_re;

            iteration += 1;

            z_im = 2 * z_re * z_im + c_im;
            z_re = z_re2 - z_im2 + c_re;
        }
    }

    if((z_re2 + z_im2) > 4){
        iteration = sqrt(iteration + 1/log(2.) * log(log(100.)/(0.5*log(z_re2 + z_im2))));
        image[gid_y][gid_x][0] = (unsigned char)(255/1.5*(0.5+sin(iteration)));
        image[gid_y][gid_x][1] = (unsigned char)(255/1.5*(0.5+sin(iteration+1)));
        image[gid_y][gid_x][2] = (unsigned char)(255/1.5*(0.5+sin(iteration+2)));
    } else {
        image[gid_y][gid_x][0] = 0;
        image[gid_y][gid_x][1] = 0;
        image[gid_y][gid_x][2] = 0;
    }
}
"""

mod = SourceModule(KERNELS)
escape_time_gpu = mod.get_function('escape_time')
exterior_distance_gpu = mod.get_function('exterior_distance')


def mandel(center, zoom):
    image = np.zeros((RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
    escape_time_gpu(cuda.InOut(image), np.float64(center[1]), np.float64(center[0]), np.float64(zoom),
                    block=(32, 32, 1), grid=(RESOLUTION//32, RESOLUTION//32, 1))
    return image


def mandel_distance(center, zoom):
    image = np.zeros((RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
    exterior_distance_gpu(cuda.InOut(image), np.float64(center[1]), np.float64(center[0]), np.float64(zoom),
                          block=(32, 32, 1), grid=(RESOLUTION//32, RESOLUTION//32, 1))
    return image
