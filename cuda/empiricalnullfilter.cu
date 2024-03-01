// MIT License
// Copyright (c) 2020-2024 Sherman Lo

#include <cuda.h>
#include <curand_kernel.h>

// See EmpiricalNullFilter - this is the main kernel and the main point of entry
// Notes: row major
// Notes: the image to filter can be commonly referred to as the cache
// Notes: __constant__ are to be set before running the kernel
// Notes: shared memory is used to store the empirical null mean and std. IF big
//   enough, also the image. Size becomes a problem if the kernel radius becomes
//   too big, in this case, the image lives in global memory and hopefully may
//   be picked up in L1 and L2
__constant__ int kRoiWidth;      // region of interest width
__constant__ int kRoiHeight;     // region of interest height
__constant__ int kCacheWidth;    // width of the image (including padding)
__constant__ int kKernelRadius;  // the radius of the kernel
__constant__ int kKernelHeight;  // the number of rows in the kernel
__constant__ int kNInitial;      // number of initial values for Newton-Raphson
__constant__ int kNStep;         // number of steps for Newton-Raphson
__constant__ int kIsCopyImageToShared;  // indicate to copy image to shared mem

/**
 * Get derivative of the log density
 *
 * Set dx_lnf to contain derivatives of the density estimate evaluated at a
 * point
 *
 * @param cache_pointer the image to filter, can either be in global or shared
 *   memory, positioned at the centre of the kernel
 * @param cache_width the width of the image in cache_pointer
 * @param bandwidth parameter for the density estimate
 * @param kernel_pointers array (even number of elements, size 2*kKernelHeight)
 *   containing pairs of integers, indicates for each row the starting and
 *   ending column position from the centre of the kernel
 * @param value where the density estimate is evaluated at
 * @param dx_lnf MODIFIED 3-element array, to store results. The elements are:
 *   <ol>
 *     <li>the density (ignore any constant multiplied to it) (NOT THE LOG)</li>
 *     <li>the first derivative of the log density</li>
 *     <li>the second derivative of the log density</li>
 *   </ol>
 */
__device__ void GetDLnDensity(float* cache_pointer, int cache_width,
                              float bandwidth, int* kernel_pointers,
                              float* value, float* dx_lnf) {
  // variables when going through all pixels in the kernel
  float z;                       // value of a pixel when looping through kernel
  float sum_kernel[3] = {0.0f};  // store sums of weights
  float phi_z;                   // weight, use Gaussian kernel

  // pointer for the image
  // point to the top of the kernel
  cache_pointer -= kKernelRadius * cache_width;

  // for each row in the kernel
  for (int i = 0; i < 2 * kKernelHeight; i++) {
    // for each column for this row
    for (int dx = kernel_pointers[i++]; dx <= kernel_pointers[i]; dx++) {
      // append to sum if the value in cache_pointer is finite
      z = *(cache_pointer + dx);
      if (isfinite(z)) {
        z -= *value;
        z /= bandwidth;
        phi_z = expf(-z * z / 2);
        sum_kernel[0] += phi_z;
        sum_kernel[1] += phi_z * z;
        sum_kernel[2] += phi_z * z * z;
      }
    }
    cache_pointer += cache_width;
  }

  // work out derivatives
  float normaliser = bandwidth * sum_kernel[0];
  dx_lnf[0] = sum_kernel[0];
  dx_lnf[1] = sum_kernel[1] / normaliser;
  dx_lnf[2] = (sum_kernel[0] * (sum_kernel[2] - sum_kernel[0]) -
               sum_kernel[1] * sum_kernel[1]) /
              (normaliser * normaliser);
}

/**
 * Find mode
 *
 * Use Newton-Raphson to find the maximum value of the density estimate. Uses
 * the passed null_mean as the initial value and modifies it at each step,
 * ending up with a final answer.
 *
 * The second derivative of the log density and the density (up to a constant)
 * at the final answer is stored in second_diff_ln and density_at_mode.
 *
 * @param cache_pointer the image to filter, can either be in global or shared
 *   memory, positioned at the centre of the kernel
 * @param cache_width the width of the image in cache_pointer
 * @param bandwidth bandwidth for the density estimate
 * @param kernel_pointers array (even number of elements, size 2*kKernelHeight)
 *   containing pairs of integers, indicates for each row the starting and
 *   ending column position from the centre of the kernel
 * @param null_mean MODIFIED initial value for the Newton-Raphson method,
 *   modified to contain the final answer
 * @param second_diff_ln MODIFIED second derivative of the log density at the
 *   mode
 * @param density_at_mode MODIFIED contains the density (up to a constant) at
 *   the mode
 * @returns true if sucessful, false otherwise
 */
__device__ bool FindMode(float* cache_pointer, int cache_width, float bandwidth,
                         int* kernel_pointers, float* null_mean,
                         float* second_diff_ln, float* density_at_mode) {
  float dx_lnf[3];
  // kNStep of Newton-Raphson
  for (int i = 0; i < kNStep; i++) {
    GetDLnDensity(cache_pointer, cache_width, bandwidth, kernel_pointers,
                  null_mean, dx_lnf);
    *null_mean -= dx_lnf[1] / dx_lnf[2];
  }
  GetDLnDensity(cache_pointer, cache_width, bandwidth, kernel_pointers,
                null_mean, dx_lnf);
  // need to check if answer is valid
  if (isfinite(*null_mean) && isfinite(dx_lnf[0]) && isfinite(dx_lnf[1]) &&
      isfinite(dx_lnf[2]) && (dx_lnf[2] < 0)) {
    *density_at_mode = dx_lnf[0];
    *second_diff_ln = dx_lnf[2];
    return true;
  }
  return false;
}

/**
 * Copy image to shared memory
 *
 * @param dest pointer to shared memory
 * @param cache_in_block_width the size of the cache captured by the block,
 *   including the padding
 * @param source pointer to image
 * @param kernel_pointers array (even number of elements, size 2*kKernelHeight)
 *   containing pairs of integers, indicates for each row the starting and
 *   ending column position from the centre of the kernel
 */
__device__ void CopyImageToSharedMemory(float* dest, int cache_in_block_width,
                                        float* source, int* kernel_pointers) {
  // point to top left
  dest -= kKernelRadius * cache_in_block_width;
  source -= kKernelRadius * kCacheWidth;
  // for each row in the kernel
  for (int i = 0; i < 2 * kKernelHeight; i++) {
    // for each column for this row
    for (int dx = kernel_pointers[i++]; dx <= kernel_pointers[i]; dx++) {
      *(dest + dx) = *(source + dx);
    }
    source += kCacheWidth;
    dest += cache_in_block_width;
  }
}

/**
 * Main kernel: Empirical Null Filter
 *
 * Does the empirical null filter on the pixels in image, giving the empirical
 * null mean (aka mode) and the empirical null std.
 *
 * @param cache array of pixels to filter
 * @param initial_sigma_roi: array of pixels (same size as the ROI) containing
 *   standard deviations, used for producing random initial values for
 *   Newton-Raphson
 * @param bandwidth_roi array of pixels (same size as the ROI) containing the
 *   bandwidth for the density estimate
 * @param kernel_pointers: array (even number of elements, size 2*kKernelHeight)
 *   containing pairs of integers, indicates for each row the starting and
 *   ending column position from the centre of the kernel
 * @param null_mean_roi MODIFIED array of pixels (same size as ROI), pass
 *   results of median filter here to be used as initial values. Modified to
 *   contain the empricial null mean afterwards.
 * @param null_std_roi MODIFIED array of pixels (same size as ROI) to contain
 *   the empirical null std
 * @param progress_roi MODIFIED array of pixels (same size as ROI) initally
 *   contains all zeros. A filtered pixel will change it to a one.
 */
extern "C" __global__ void EmpiricalNullFilter(
    float* cache, float* initial_sigma_roi, float* bandwidth_roi,
    int* kernel_pointers, float* null_mean_roi, float* null_std_roi,
    int* progress_roi) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;
  // adjust pointer to the corresponding x y coordinates
  cache += (y0 + kKernelRadius) * kCacheWidth + x0 + kKernelRadius;
  // check if in roi
  // &&isfinite(*cache) is not required as accessing the image from this
  // pixel is within bounds
  bool is_in_roi = x0 < kRoiWidth && y0 < kRoiHeight;

  // get shared memory
  extern __shared__ float shared_memory[];
  float* null_mean_shared_pointer = shared_memory;
  float* second_diff_shared_pointer =
      null_mean_shared_pointer + blockDim.x * blockDim.y;

  // offset by the x and y coordinates
  int roi_index = y0 * kRoiWidth + x0;
  int null_shared_index = threadIdx.y * blockDim.x + threadIdx.x;

  // cache_pointer points to the image to filter (including padding)
  // cache_pointer may either points to global or shared memory
  // cache_width will specify the width of the cache according to if the cache
  // is in global or shared memory
  float* cache_pointer;
  int cache_width;

  // if the shared memory is big enough, copy the image
  // cache_pointer points to shared memory if shared memory allows it, otherwise
  // points to global memory
  if (kIsCopyImageToShared) {
    // width of the cache captured by a block, including the padding
    // padding is of kKernelRadius size, on left and right
    cache_width = blockDim.x + 2 * kKernelRadius;
    cache_pointer = second_diff_shared_pointer + blockDim.x * blockDim.y;
    cache_pointer += (threadIdx.y + kKernelRadius) * cache_width + threadIdx.x +
                     kKernelRadius;
    // copy image to shared memory
    if (is_in_roi) {
      CopyImageToSharedMemory(cache_pointer, cache_width, cache,
                              kernel_pointers);
    }
  } else {
    // else keep the cache in global memory
    cache_width = kCacheWidth;
    cache_pointer = cache;
  }

  __syncthreads();

  // adjust pointer to the corresponding x y coordinates
  null_mean_shared_pointer += null_shared_index;
  second_diff_shared_pointer += null_shared_index;

  // for rng
  curandState_t state;
  curand_init(0, roi_index, 0, &state);
  // null_mean used to store mode for each initial value
  float null_mean;
  float median;
  float sigma;      // how much noise to add
  float bandwidth;  // bandwidth for density estimate

  if (is_in_roi) {
    null_mean = null_mean_roi[roi_index];  // use median as first initial
    median = null_mean;
    // modes with highest densities are stored in shared memory
    *null_mean_shared_pointer = null_mean;
    sigma = initial_sigma_roi[roi_index];  // how much noise to add
    bandwidth = bandwidth_roi[roi_index];  // bandwidth for density estimate
  }

  bool is_success;        // indicate if newton-raphson was sucessful
  float density_at_mode;  // density for this particular mode
  // second derivative of the log density, to set empirical null std
  float second_diff_ln;
  // keep solution with the highest density
  float max_density_at_mode = -INFINITY;

  // try different initial values, the first one is the median, then for
  // additional initial values, add normal noise to neighbouring null_mean
  // solutions in shared memory, neighbours rotate from -1, itself and +1 from
  // current pointer
  int min;
  int n_neighbour;
  float initial0;
  if (null_shared_index == 0) {
    min = 0;
  } else {
    min = -1;
  }
  if (null_shared_index == blockDim.x * blockDim.y - 1) {
    n_neighbour = 1 - min;
  } else {
    n_neighbour = 2 - min;
  }

  for (int i = 0; i < kNInitial; i++) {
    if (is_in_roi) {
      is_success =
          FindMode(cache_pointer, cache_width, bandwidth, kernel_pointers,
                   &null_mean, &second_diff_ln, &density_at_mode);
      // keep null_mean and nullStd with the highest density
      if (is_success) {
        if (density_at_mode > max_density_at_mode) {
          max_density_at_mode = density_at_mode;
          *null_mean_shared_pointer = null_mean;
          *second_diff_shared_pointer = second_diff_ln;
        }
      }
    }

    // try different initial value
    __syncthreads();

    if (is_in_roi) {
      // try an initial value using its neighbour in shared memory
      initial0 = *(null_mean_shared_pointer + i % n_neighbour + min);
      // ensure the initial value is finite, otherwise use previous solution
      if (!isfinite(initial0)) {
        initial0 = null_mean;
      }
      // add normal noise and add bias towards median
      null_mean = (initial0 + median) / 2 + sigma * curand_normal(&state);
    }
  }

  // store final results
  if (is_in_roi) {
    null_mean_roi[roi_index] = *null_mean_shared_pointer;
    null_std_roi[roi_index] = powf(-*second_diff_shared_pointer, -0.5f);
    progress_roi[roi_index] = 1;
  }
}

/**
 * Get the (local) count, mean and std in a kernel
 *
 * @param cache_pointer the image to filter on global memory, positioned at the
 *   centre of the kernel
 * @param kernel_pointers array (even number of elements, size 2*kKernelHeight)
 *   containing pairs of integers, indicates for each row the starting and
 *   ending column position from the centre of the kernel
 * @param count MODIFIED the resulting local count, ie number of finite elements
 *   in the kernel
 * @param mean MODIFIED the resulting local mean
 * @param std MODIFIED the resulting local std
 */
__device__ void GetMeanStd(float* cache_pointer, int* kernel_pointers,
                           int* count, float* mean, float* std) {
  float z;  // value of a pixel when looping through kernel

  // initial values
  *count = 0;
  *mean = {0.0f};
  *std = {0.0f};

  // pointer for the image
  // point to the top of the kernel
  float* cache_start = cache_pointer - kKernelRadius * kCacheWidth;

  cache_pointer = cache_start;

  // calculate count and mean here
  // for each row in the kernel
  for (int i = 0; i < 2 * kKernelHeight; i++) {
    // for each column for this row
    for (int dx = kernel_pointers[i++]; dx <= kernel_pointers[i]; dx++) {
      z = *(cache_pointer + dx);
      if (isfinite(z)) {
        ++(*count);
        *mean += z;
      }
    }
    cache_pointer += kCacheWidth;
  }
  *mean /= (float)*count;

  // given mean, calculate std
  cache_pointer = cache_start;
  // for each row in the kernel
  for (int i = 0; i < 2 * kKernelHeight; i++) {
    // for each column for this row
    for (int dx = kernel_pointers[i++]; dx <= kernel_pointers[i]; dx++) {
      z = *(cache_pointer + dx);
      if (isfinite(z)) {
        *std += (z - *mean) * (z - *mean);
      }
    }
    cache_pointer += kCacheWidth;
  }
  *std /= (float)(*count - 1);
  *std = sqrtf(*std);
}

/**
 * Kernel: Mean and Standard Deviation Filter
 *
 * Does the mean and standard deviation filter on an image. It ignore non-finite
 * elements. Also returns the local number of finite elements in the kernel.
 * Non-finite elements occur at the padding.
 *
 * @param cache array of pixels to filter
 * @param kernel_pointers: array (even number of elements, size 2*kKernelHeight)
 *   containing pairs of integers, indicates for each row the starting and
 *   ending column position from the centre of the kernel
 * @param count_roi MODIFIED array of pixels (same size as ROI), pass results
 *   with the local number of finite elements
 * @param mean_roi MODIFIED array of pixels (same size as ROI), pass results of
 *   the mean filter
 * @param std_roi MODIFIED array of pixels (same size as ROI), pass results of
 *   the std filter
 */
extern "C" __global__ void MeanStdFilter(float* cache, int* kernel_pointers,
                                         int* count_roi, float* mean_roi,
                                         float* std_roi) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;
  // adjust pointer to the corresponding x y coordinates
  cache += (y0 + kKernelRadius) * kCacheWidth + x0 + kKernelRadius;
  // check if in roi
  bool is_in_roi = x0 < kRoiWidth && y0 < kRoiHeight;

  // offset by the x and y coordinates
  int roi_index = y0 * kRoiWidth + x0;

  if (is_in_roi) {
    GetMeanStd(cache, kernel_pointers, count_roi + roi_index,
               mean_roi + roi_index, std_roi + roi_index);
  }
}
