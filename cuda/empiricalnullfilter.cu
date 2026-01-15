// MIT License
// Copyright (c) 2020-2026 Sherman Lo

/**
 * @file empiricalnullfilter.cu
 *
 * Empirical null filter - consisting of the null mean filter (aka mode filter)
 * and the null std filter. The filter, using the pixels in the kernel, creates
 * a local empirical density and the mode and standard deviation are solved
 * using the Newton-Raphson method
 *
 * The CUDA kernel is <code>EmpiricalNullFilter<<<>>>()</code>
 *
 * The parameters of the CUDA kernel requires the median filtered image,
 * standard deviation filtered image and bandwidth parameter image. The CUDA
 * kernel <code>MeanStdFilter<<<>>>()</code> is provided and may be used to
 * calculate the standard deviation filtered image and bandwidth parameter image
 *
 * Before running any of the GPU kernels, ensure the
 * <code>\_\_constant\_\_</code> variables are set
 *
 * Notes:
 * <ul>
 *   <li>Row major</li>
 *   <li>
 *     The word <i>kernel</i> on its own refers to the circular boundary,
 *     centred on a pixel, which captures pixels within it and used for
 *     calcaultions. Each thread will work on a kernel centred on the pixel the
 *     thread is working on. To avoid confusion, <i>CUDA kernel</i> is used to
 *     refer to a <code>\_\_global\_\_</code> function
 *   </li>
 *   <li>
 *     The image to filter is called the region of interest or ROI. The cache
 *     refers to the image which contains the ROI and NaN padding around it.
 *   </li>
 *   <li>
 *     Shared memory is used to store the empirical null mean and std. <b>If</b>
 *     big enough, it also stores the cache. Size becomes a problem if the
 *     kernel radius becomes too big, in this case, the cache lives in global
 *     memory
 *   </li>
 * </ul>
 */

#include <cuda.h>
#include <curand_kernel.h>

/** Width of the region of interest */
__constant__ int kRoiWidth;
/** Height of the region of interest */
__constant__ int kRoiHeight;
/** Width of the image (including padding) */
__constant__ int kCacheWidth;
/** Radius of the kernel */
__constant__ int kKernelRadius;
/** Number of rows in the kernel */
__constant__ int kKernelHeight;
/** Number of initial values for Newton-Raphson */
__constant__ int kNInitial;
/** Number of steps for Newton-Raphson */
__constant__ int kNStep;
/** Indicate to copy image to shared memory or not */
__constant__ int kIsCopyImageToShared;

struct GridInfo {
  int x0;         /** <code>x</code> coordinate of this thread */
  int y0;         /** <code>y</code> coordinate of this thread */
  bool is_in_roi; /** Indicate if this thread is in the region of interest */
  int roi_index;  /** Index of this thread in the region of interest */
};

/**
 * Get information about this thread
 *
 * @return GridInfo
 */
__device__ GridInfo GetGridInfo() {
  GridInfo grid_info;
  grid_info.x0 = threadIdx.x + blockIdx.x * blockDim.x;
  grid_info.y0 = threadIdx.y + blockIdx.y * blockDim.y;
  grid_info.is_in_roi = grid_info.x0 < kRoiWidth && grid_info.y0 < kRoiHeight;
  grid_info.roi_index = grid_info.y0 * kRoiWidth + grid_info.x0;
  return grid_info;
}

/**
 * Get the width of the cache in shared memory
 *
 * Return the width of the cache in shared memory. The cache in shared memory is
 * the ROI and padding captured by the block
 *
 * @return Width of the cache in shared memory
 */
__device__ int GetSharedMemCacheWidth() {
  return blockDim.x + 2 * kKernelRadius;
}

/**
 * Get derivative of the log density
 *
 * Capture all pixels by the kernel and draw an empirical density from it. This
 * function returns the density evaluated at a point, as well as the log density
 * and the second derivative of the log density. They are outputted in the
 * parameter <code>dx_lnf</code>
 *
 * @param cache Image to filter (the ROI and padding), this can either be in
 *   global or shared memory. The pointer is to be positioned at the centre of
 *   the kernel
 * @param cache_width Width of the image in <code>cache</code>
 * @param bandwidth Bandwidth parameter for the density estimate
 * @param kernel_pointers Array (even number of elements, size
 *   <code>2 * kKernelHeight</code>) containing pairs of integers, indicates for
 *   each row the starting and ending column position from the centre of the
 *   kernel
 * @param value Where the density estimate is to be evaluated at
 * @param dx_lnf <b>Modified</b> 3-element array - To store results where the
 * elements are the following:
 *   <ol>
 *     <li>The density (ignore any constant multiplied to it) <b>not the
 *         log</b></li>
 *     <li>The first derivative of the log density</li>
 *     <li>The second derivative of the log density</li>
 *   </ol>
 */
__device__ void GetDLnDensity(float* cache, int cache_width, float bandwidth,
                              int* kernel_pointers, float* value,
                              float* dx_lnf) {
  // variables when going through all pixels in the kernel
  float z;                       // value of a pixel when looping through kernel
  float sum_kernel[3] = {0.0f};  // store sums of weights
  float phi_z;                   // weight, use Gaussian kernel

  // pointer for the image
  // point to the top of the kernel
  cache -= kKernelRadius * cache_width;

  // for each row in the kernel
  for (int i = 0; i < 2 * kKernelHeight; i++) {
    // for each column for this row
    for (int dx = kernel_pointers[i++]; dx <= kernel_pointers[i]; dx++) {
      // append to sum if the value in cache is finite
      z = *(cache + dx);
      if (isfinite(z)) {
        z -= *value;
        z /= bandwidth;
        phi_z = expf(-z * z / 2);
        sum_kernel[0] += phi_z;
        sum_kernel[1] += phi_z * z;
        sum_kernel[2] += phi_z * z * z;
      }
    }
    cache += cache_width;
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
 * the passed <code>null_mean</code> as the initial value and modifies it at
 * each step, ending up with a final answer
 *
 * The second derivative of the log density and the density (up to a constant)
 * at the final answer are stored in <code>second_diff_ln</code> and
 * <code>density_at_mode</code> if the algoirthm is successful. It is deemed
 * unsuccessful if any of the values are not finite or if the second derivative
 * of the log density is non-negative
 *
 * @param cache Image to filter (the ROI and padding), this can either be in
 *   global or shared memory. The pointer is to be positioned at the centre of
 *   the kernel
 * @param cache_width Width of the image in <code>cache</code>
 * @param bandwidth Bandwidth parameter for the density estimate
 * @param kernel_pointers Array (even number of elements, size
 *   <code>2 * kKernelHeight</code>) containing pairs of integers, indicates for
 *   each row the starting and ending column position from the centre of the
 *   kernel
 * @param null_mean <b>Modified</b> Initial value for the Newton-Raphson method.
 *   Modified after each step to contain the mode during the algorithm
 * @param second_diff_ln <b>Modified</b> To contain the second derivative of the
 *   log density at the mode if successful
 * @param density_at_mode <b>Modified</b> To contains the density (up to a
 *   constant) at the mode if successful
 * @returns <code>true</code> if sucessful, <code>false</code> otherwise
 */
__device__ bool FindMode(float* cache, int cache_width, float bandwidth,
                         int* kernel_pointers, float* null_mean,
                         float* second_diff_ln, float* density_at_mode) {
  float dx_lnf[3];
  // kNStep of Newton-Raphson
  for (int i = 0; i < kNStep; i++) {
    GetDLnDensity(cache, cache_width, bandwidth, kernel_pointers, null_mean,
                  dx_lnf);
    *null_mean -= dx_lnf[1] / dx_lnf[2];
  }
  GetDLnDensity(cache, cache_width, bandwidth, kernel_pointers, null_mean,
                dx_lnf);
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
 * Copy all pixels captured by this thread's kernel from global memory to shared
 * memory
 *
 * @param source Pointer to image at the centre of the kernel
 * @param kernel_pointers Array (even number of elements, size
 *   <code>2 * kKernelHeight</code>) containing pairs of integers, indicates for
 *   each row the starting and ending column position from the centre of the
 *   kernel
 * @param dest <b>Modified</b> Pointer to shared memory for this thread
 */
__device__ void CopyImageToSharedMemory(float* source, int* kernel_pointers,
                                        float* dest) {
  // point to top left
  dest -= kKernelRadius * GetSharedMemCacheWidth();
  source -= kKernelRadius * kCacheWidth;
  // for each row in the kernel
  for (int i = 0; i < 2 * kKernelHeight; i++) {
    // for each column for this row
    for (int dx = kernel_pointers[i++]; dx <= kernel_pointers[i]; dx++) {
      *(dest + dx) = *(source + dx);
    }
    source += kCacheWidth;
    dest += GetSharedMemCacheWidth();
  }
}

/**
 * Get shared memory pointers
 *
 * Get pointers to shared memory for the cache, that is the ROI and padding.
 * Depending on <code>kIsCopyImageToShared</code>, the cache can either be in
 * global memory or in shared memory (where a block of the image is copied from
 * global memory to shared memory)
 *
 * If <code>kIsCopyImageToShared</code> is <code>true</code>, the cache is
 * copied to shared memory, else the cache is left in global memory
 *
 * Also get pointers to shared memory for the null mean and the second diff of
 * the log density. They only capture the ROI
 *
 * @param kernel_pointers Array (even number of elements, size
 *   <code>2 * kKernelHeight</code>) containing pairs of integers, indicates for
 *   each row the starting and ending column position from the centre of the
 *   kernel
 * @param cache <b>Modified</b> Image to filter (the ROI and padding) in global
 *   memory. The pointer is at the start of the image. This is then modified so
 *   that the pointer is at the centre of the kernel for this thread. In
 *   addition, the memory pointed to may either be in global or shared memory
 * @param cache_width <b>Modified</b> To contain the width of the image in
 *   <code>cache</code>. If <code>cache</code> is in global memory, the width
 *   is of the entire image (ROI and padding). Otherwise in shared memory, the
 *   width is of the block plus padding
 * @param null_mean_shared <b>Modified</b> To point to shared memory for
 *   storing the null mean for this thread
 * @param second_diff_ln_shared <b>Modified</b> To point to the shared memory
 *   for storing the second derivative of the log density for this thread
 */
__device__ void GetSharedMemPointers(int* kernel_pointers, float** cache,
                                     int* cache_width, float** null_mean_shared,
                                     float** second_diff_ln_shared) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;
  // adjust pointer to the corresponding x y coordinates
  *cache += (y0 + kKernelRadius) * kCacheWidth + x0 + kKernelRadius;
  // check if in roi
  // &&isfinite(*cache) is not required as accessing the image from this
  // pixel is within bounds
  bool is_in_roi = x0 < kRoiWidth && y0 < kRoiHeight;

  // get shared memory
  extern __shared__ float shared_memory[];
  *null_mean_shared = shared_memory;
  *second_diff_ln_shared = *null_mean_shared + blockDim.x * blockDim.y;

  // cache_pointer points to the image to filter (including padding)
  // cache_pointer may either points to global or shared memory
  // cache_width will specify the width of the cache according to if the cache
  // is in global or shared memory

  // if the shared memory is big enough, copy the image
  // cache_pointer points to shared memory if shared memory allows it, otherwise
  // points to global memory
  if (kIsCopyImageToShared) {
    // width of the cache captured by a block, including the padding
    // padding is of kKernelRadius size, on left and right
    *cache_width = GetSharedMemCacheWidth();
    float* cache_shared = *second_diff_ln_shared + blockDim.x * blockDim.y;
    cache_shared += (threadIdx.y + kKernelRadius) * *cache_width + threadIdx.x +
                    kKernelRadius;
    // copy image to shared memory
    if (is_in_roi) {
      CopyImageToSharedMemory(*cache, kernel_pointers, cache_shared);
    }
    // use the cache in shared memory by pointing to it
    *cache = cache_shared;
  } else {
    // else keep the cache in global memory
    *cache_width = kCacheWidth;
  }

  // adjust pointer to the corresponding x y coordinates
  int shared_memory_index = threadIdx.y * blockDim.x + threadIdx.x;
  *null_mean_shared += shared_memory_index;
  *second_diff_ln_shared += shared_memory_index;
}

/**
 * Find the mode using multiple initial values
 *
 * Use Newton-Raphson to find the maximum value of the density estimate with
 * different initial values. The result with the highest density is kept. The
 * initial values are a weighted sum of the median and the previous solution
 * from this thread or a thread adjacent to it with random noise added
 *
 * @param cache Image to filter (the ROI and padding), this can either be in
 *   global or shared memory. The pointer is to be positioned at the centre of
 *   the kernel
 * @param cache_width Width of the image in <code>cache</code>
 * @param median The median over the kernel
 * @param sigma The standard deviation of the added Normal noise
 * @param bandwidth Bandwidth parameter for the density estimate
 * @param kernel_pointers Array (even number of elements, size
 *   <code>2 * kKernelHeight</code>) containing pairs of integers, indicates for
 *   each row the starting and ending column position from the centre of the
 *   kernel
 * @param null_mean_shared <b>Modified</b> To contain the null mean for this
 *   thread
 * @param second_diff_ln_shared <b>Modified</b> To contain the second derivative
 *   of the log density for this thread
 */
__device__ void MultiFindMode(float* cache, int cache_width, float median,
                              float sigma, float bandwidth,
                              int* kernel_pointers, float* null_mean_shared,
                              float* second_diff_ln_shared) {
  GridInfo grid_info = GetGridInfo();

  float null_mean = median;

  if (grid_info.is_in_roi) {
    // initalise null_std with nan, this will stay nan if all repeats of
    // newton-raphson fails
    *null_mean_shared = median;
    *second_diff_ln_shared = NAN;
  }

  // try different initial values, the first one is the median, then for
  // additional initial values, add normal noise to neighbouring null_mean
  // solutions in shared memory, neighbours rotate from -1, itself and +1 from
  // current pointer
  // if on left edge or right edge of shared memory or block, do not use
  // initial value which goes beyond the boundary or edge
  int min;
  int n_neighbour;
  if (threadIdx.x == 0) {  // left edge
    // set to zero so that it does not go beyond left edge
    min = 0;
  } else {
    min = -1;
  }
  if (threadIdx.x == blockDim.x - 1 ||
      grid_info.x0 == kRoiWidth - 1) {  // right edge
    // reduce number of neighbours so that it does not go beyond right edge
    n_neighbour = 1 - min;
  } else {
    n_neighbour = 2 - min;
  }

  // for rng
  curandState_t state;
  curand_init(0, grid_info.roi_index, 0, &state);
  // keep solution with the highest density
  float max_density_at_mode = -INFINITY;

  for (int i = 0; i < kNInitial; i++) {
    if (grid_info.is_in_roi) {
      float density_at_mode;  // density for this particular mode
      // second derivative of the log density, to set empirical null std
      float second_diff_ln;
      // indicate if newton-raphson was sucessful
      bool is_success = FindMode(cache, cache_width, bandwidth, kernel_pointers,
                                 &null_mean, &second_diff_ln, &density_at_mode);
      // keep null_mean and nullStd with the highest density
      if (is_success) {
        if (density_at_mode > max_density_at_mode) {
          max_density_at_mode = density_at_mode;
          *null_mean_shared = null_mean;
          *second_diff_ln_shared = second_diff_ln;
        }
      }
    }

    // try different initial value
    __syncthreads();

    if (grid_info.is_in_roi) {
      // try an initial value using its neighbour in shared memory
      float initial0 = *(null_mean_shared + i % n_neighbour + min);
      // ensure the initial value is finite, otherwise use previous solution
      if (!isfinite(initial0)) {
        initial0 = null_mean;
      }
      // add normal noise and add bias towards median
      null_mean = (initial0 + median) / 2 + sigma * curand_normal(&state);
    }
  }
}

/**
 * Empirical Null Filter
 *
 * Does the empirical null filter on an image, giving the empirical null mean
 * (aka mode) and the empirical null std
 *
 * In this filter, all pixels in the kernel are gathered to create an empirical
 * density. The mode is solved using the Newton-Raphson method with different
 * initial values. The standard deviation can also be obatined from the
 * second derivative of the log density at the mode. They are called the null
 * mean and null std respectively
 *
 * To initalise the Newton-Raphson method, pass the resulting median filtered
 * image via the parameter <code>null_mean_roi</code> - this is later modified,
 * see below. This is used as the initial value. To produce futher random
 * initial value, pass the resulting std filter via the parameter
 * <code>initial_sigma_roi</code>
 *
 * The resulting null mean and null std images are returned via the
 * <code>null_mean_roi</code> and <code>null_std_roi</code> parameters
 * respectively
 *
 * @param cache Image to filter (the ROI and padding) in global memory. The
 *   pointer is at the start of the image
 * @param initial_sigma_roi: Image (same size as the ROI) containing standard
 *   deviations, used for producing random initial values for Newton-Raphson
 * @param bandwidth_roi Image (same size as the ROI) containing the bandwidth
 *   parameter for the density estimate
 * @param kernel_pointers Array (even number of elements, size
 *   <code>2 * kKernelHeight</code>) containing pairs of integers, indicates for
 *   each row the starting and ending column position from the centre of the
 *   kernel
 * @param null_mean_roi <b>Modified</b> Image same size as ROI. Pass results of
 *   median filter here to be used as initial values. This is then modified to
 *   contain the empricial null mean afterwards. If the Newton-Raphson method
 *   fails in a pixel, it will remain as the medium
 * @param null_std_roi <b>Modified</b> Image same size as ROI. To contain the
 *   resulting empirical null std. If the Newton-Raphson method fails in a
 *   pixel, it take a value of <code>NAN</code>
 * @param progress_roi <b>Modified</b>  Image same size as ROI. To contain
 *   initally contains all zeros. A filtered pixel will change it to a one
 */
extern "C" __global__ void EmpiricalNullFilter(
    float* cache, float* initial_sigma_roi, float* bandwidth_roi,
    int* kernel_pointers, float* null_mean_roi, float* null_std_roi,
    int* progress_roi) {
  GridInfo grid_info = GetGridInfo();

  int cache_width;  // width of the cache after calling GetSharedMemPointers()
  float* null_mean_shared;       // shared memory for null mean
  float* second_diff_ln_shared;  // shared memory for second diff of log density
  GetSharedMemPointers(kernel_pointers, &cache, &cache_width, &null_mean_shared,
                       &second_diff_ln_shared);

  float median;
  float sigma;
  float bandwidth;

  if (grid_info.is_in_roi) {
    // initalise values
    median = null_mean_roi[grid_info.roi_index];
    sigma = initial_sigma_roi[grid_info.roi_index];
    bandwidth = bandwidth_roi[grid_info.roi_index];
  }

  MultiFindMode(cache, cache_width, median, sigma, bandwidth, kernel_pointers,
                null_mean_shared, second_diff_ln_shared);

  // store final results
  if (grid_info.is_in_roi) {
    null_mean_roi[grid_info.roi_index] = *null_mean_shared;
    null_std_roi[grid_info.roi_index] = powf(-*second_diff_ln_shared, -0.5f);
    progress_roi[grid_info.roi_index] = 1;
  }
}

/**
 * Get the (local) count, mean and std in a kernel
 *
 * @param cache Image to filter (the ROI and padding) in global memory. The
 *   pointer is to be positioned at the centre of the kernel
 * @param kernel_pointers Array (even number of elements, size
 *   <code>2 * kKernelHeight</code>) containing pairs of integers, indicates for
 *   each row the starting and ending column position from the centre of the
 *   kernel
 * @param count <b>Modified</b> To contain the resulting local count, ie number
 *   of finite elements in the kernel
 * @param mean <b>Modified</b> To contain the resulting local mean
 * @param std <b>Modified</b> To containthe resulting local std
 */
__device__ void GetMeanStd(float* cache, int* kernel_pointers, int* count,
                           float* mean, float* std) {
  float z;  // value of a pixel when looping through kernel

  // initial values
  *count = 0;
  *mean = {0.0f};
  *std = {0.0f};

  // pointer for the image
  // point to the top of the kernel
  float* cache_start = cache - kKernelRadius * kCacheWidth;

  cache = cache_start;

  // calculate count and mean here
  // for each row in the kernel
  for (int i = 0; i < 2 * kKernelHeight; i++) {
    // for each column for this row
    for (int dx = kernel_pointers[i++]; dx <= kernel_pointers[i]; dx++) {
      z = *(cache + dx);
      if (isfinite(z)) {
        ++(*count);
        *mean += z;
      }
    }
    cache += kCacheWidth;
  }
  *mean /= (float)*count;

  // given mean, calculate std
  cache = cache_start;
  // for each row in the kernel
  for (int i = 0; i < 2 * kKernelHeight; i++) {
    // for each column for this row
    for (int dx = kernel_pointers[i++]; dx <= kernel_pointers[i]; dx++) {
      z = *(cache + dx);
      if (isfinite(z)) {
        *std += (z - *mean) * (z - *mean);
      }
    }
    cache += kCacheWidth;
  }
  *std /= (float)(*count - 1);
  *std = sqrtf(*std);
}

/**
 * Mean and Standard Deviation Filter
 *
 * Does the mean and standard deviation filter on an image. It ignores
 * non-finite elements. Also returns the local number of finite elements in the
 * kernel. Non-finite elements occur at the padding
 *
 * @param cache Image to filter (the ROI and padding) in global memory. The
 *   pointer is to be positioned at the centre of the kernel
 * @param kernel_pointers Array (even number of elements, size
 *   <code>2 * kKernelHeight</code>) containing pairs of integers, indicates for
 *   each row the starting and ending column position from the centre of the
 *   kernel
 * @param count_roi <b>Modified</b> Image same size as ROI. To contain the
 *   resulting local number of finite elements
 * @param mean_roi <b>Modified</b> Image same size as ROI. To contain the
 *   resulting mean filter
 * @param std_roi <b>Modified</b> Image same size as ROI. To contain the
 *   resulting std filter
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
