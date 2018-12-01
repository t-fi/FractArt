import matplotlib.pyplot as plt

import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from tqdm import tqdm

from mandel_viewer.renderer import mandel

import cv2
import blosc
import multiprocessing as mp

KERNELS = """
#define precision double

__device__ precision cabs(precision re, precision im){
        return sqrt(re*re + im*im);
}

__global__ void exterior_distance(const precision* __restrict__ c_im_arr, const precision* __restrict__ c_re_arr,
                                  precision* __restrict__ b){
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;

    const precision c_im = c_im_arr[gid];
    const precision c_re = c_re_arr[gid];

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
        b[gid] = log(2 * cabs(z_re, z_im)*log(cabs(z_re, z_im))/cabs(dz_re, dz_im));
    } else {
        b[gid] = 0;
    }
}
"""

mod = SourceModule(KERNELS)
exterior_distance_gpu = mod.get_function('exterior_distance')


def exterior_distances(im, re):
    b = np.zeros_like(im)
    exterior_distance_gpu(cuda.In(im), cuda.In(re), cuda.InOut(b),
                          grid=(b.size // 1024, 1, 1), block=(1024, 1, 1))
    return b


def process_and_store_image(image, re, im, zoom):
    image = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'/home/oba/Dropbox/Kunst/generated_fractals/{re:.16f}_{im:.16f}_{zoom:.16f}.png', image)
    # c = blosc.compress_ptr(image.__array_interface__['data'][0], image.size, image.dtype.itemsize, 9, True)
    # np.save(f'/home/oba/Dropbox/Kunst/generated_fractals/{re:.16f}_{im:.16f}_{zoom:.16f}', c, False)


def gather_samples(num_blocks):
    re = np.random.uniform(-2, 0.48, 32 * 32 * num_blocks).astype(np.float64)
    im = np.random.uniform(-2, 0.48, 32 * 32 * num_blocks).astype(np.float64)
    b = exterior_distances(im, re)
    valid_indices = np.logical_and(b > -35, b < -25)
    re = re[valid_indices]
    im = im[valid_indices]
    b = np.exp(b[valid_indices])
    p = []
    for im_, re_, b_ in tqdm(zip(im, re, b), total=b.size):
        zoom = b_ * 500
        image = mandel([re_, im_], zoom)
        p.append(mp.Process(target=process_and_store_image, args=[image, re_, im_, zoom]))
        p[-1].start()
        # plt.imshow(image, origin='lower')
        # plt.show()


gather_samples(128)
