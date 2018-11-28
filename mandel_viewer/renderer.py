import matplotlib.pyplot as plt

import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# TODO:
#   choose random point in [-2..1 + -1..1i]
#   calculate exterior distance D => throw away if NaN
#   Render img around point with zoom D
#   Subsample img


KERNELS = """
__device__ double cabs(double re, double im){
        return sqrt(re*re + im*im);
}

__global__ void exterior_distance(float b[1024][1024], const double center_im, const double center_re, const double zoom){
    const int gid_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int gid_y = threadIdx.y + blockIdx.y * blockDim.y;

    const double c_im = zoom*(__int2float_rn(gid_x)/1024-0.5) + center_im;
    const double c_re = zoom*(__int2float_rn(gid_y)/1024-0.5) + center_re;

    double dz_im = 0;
    double dz_re = 1.;

    double _dz_im = 0;
    double _dz_re = 1.;
      
    double z_im = 0;
    double z_re = 0;    
    double z_im2 = 0.;
    double z_re2 = 0.;    

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
        b[gid_x][gid_y] = log(2 * cabs(z_re, z_im)*log(cabs(z_re, z_im))/cabs(dz_re, dz_im));
    } else {
        b[gid_x][gid_y] = __int_as_float(0x7fffffff);
    }
}

__global__ void escape_time(unsigned char image[1024][1024][3], const double center_im, const double center_re, const double zoom){
    const int gid_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int gid_y = threadIdx.y + blockIdx.y * blockDim.y;

    const double c_im = zoom*(__int2float_rn(gid_x)/1024-0.5) + center_im;
    const double c_re = zoom*(__int2float_rn(gid_y)/1024-0.5) + center_re;

    double iteration = 0;

    double z_im = 0.;
    double z_re = 0.;    
    double z_im2 = 0.;
    double z_re2 = 0.;    

    for(unsigned i=0; i<10000; ++i){
        if((z_re2 + z_im2) < 100){

            z_im2 = z_im*z_im;
            z_re2 = z_re*z_re;

            iteration += 1;

            z_im = 2 * z_re * z_im + c_im;
            z_re = z_re2 - z_im2 + c_re;
        }
    }

    if((z_re2 + z_im2) > 4){
        iteration = log(iteration + 1/log(2.) * log(log(100.)/(0.5*log(z_re2 + z_im2))));
        image[gid_x][gid_y][0] = (unsigned char)(255*sin(iteration));
        image[gid_x][gid_y][1] = (unsigned char)(255*sin(iteration+1));
        image[gid_x][gid_y][2] = (unsigned char)(255*sin(iteration+2));
    } else {
        image[gid_x][gid_y][0] = 0;
        image[gid_x][gid_y][1] = 0;
        image[gid_x][gid_y][2] = 0;
    }
}
"""

mod = SourceModule(KERNELS)
escape_time_gpu = mod.get_function('escape_time')
exterior_distance_gpu = mod.get_function('exterior_distance')


def mandel(center, zoom):
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    escape_time_gpu(cuda.InOut(image), np.float64(center[1]), np.float64(center[0]), np.float64(zoom),
                    block=(32, 32, 1), grid=(32, 32, 1))
    return image


def mandel_distance(center, zoom):
    b = np.zeros((1024, 1024), dtype=np.float32)
    exterior_distance_gpu(cuda.InOut(b), np.float64(center[1]), np.float64(center[0]), np.float64(zoom),
                          block=(32, 32, 1), grid=(32, 32, 1))
    return b


def test_run():
    # @2048*2048:
    # for float min~0.0001
    # for double min~0.0000000000001
    zoom = 3
    center = np.array([0.34953738586551997, 0.06642601431185])
    iterations = mandel(center, zoom)[:, :, 0]
    plt.matshow(iterations, cmap='prism',
                extent=[center[0] - zoom / 2, center[0] + zoom / 2, center[1] - zoom / 2, center[1] + zoom / 2],
                origin='lower')
    # plt.figure()
    # plt.hist(iterations.ravel(), bins=100)
    # plt.yscale('log', nonposy='clip')
    plt.show()


if __name__ == '__main__':
    test_run()
