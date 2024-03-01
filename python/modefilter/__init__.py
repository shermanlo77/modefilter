import ctypes
from importlib.resources import files
import math

import cupy
import numpy as np
import scipy.ndimage


PTX_FILE_PATH = str(
    files("modefilter").joinpath("empiricalnullfilter.ptx"))


class EmpiricalNullFilter:
    """Empirical null filter using GPU

    Locally normalise an image using the empirical null mean (also known as the
    mode) and empirical null std.

    How to use:
        - construct the filter filter = EmpiricalNullFilter(radius)
        - set optional parameters, for example, filter.set_n_initial(100)
        - call filter.filter(image) which returns the filtered image
        - call filter.get_null_mean() or filter.get_null_std() to get the
              the null mean or null std image
    """

    def __init__(self, radius):
        self._radius = radius
        self._n_initial = 3
        self._n_step = 10
        self._bandwidth_parameter_a = 0.16
        self._bandwidth_parameter_b = 0.9
        self._block_dim_x = 16
        self._block_dim_y = 16
        self._std_for_zero = 0.289
        self._null_mean = None
        self._null_std = None
        self._kernel = _Kernel(self._radius)

    def set_n_initial(self, n_initial):
        """Set the number of initial points for the Newton method

        Args:
            n_initial (int): Number of initial points for Newton method
        """
        self._n_initial = n_initial

    def set_n_step(self, n_step):
        """Set the number of stepsfor the Newton method

        Args:
            n_step (int): Number of steps for Newton method
        """
        self._n_step = n_step

    def set_bandwidth_parameter_a(self, bandwidth_parameter_a):
        """Set the bandwidth parameter A for density estimate

        Args:
            bandwidth_parameter_a (float): bandwidth parameter A
        """
        self._bandwidth_parameter_a = bandwidth_parameter_a

    def set_bandwidth_parameter_b(self, bandwidth_parameter_b):
        """Set the bandwidth parameter B for density estimate

        Args:
            bandwidth_parameter_b (float): bandwidth parameter B
        """
        self._bandwidth_parameter_b = bandwidth_parameter_b

    def set_block_dim_x(self, block_dim_x):
        """Set the x block dimension for GPU

        Args:
            block_dim_x (int): x block dimension
        """
        self._block_dim_x = block_dim_x

    def set_block_dim_y(self, block_dim_y):
        """Set the y block dimension for GPU

        Args:
            block_dim_y (int): y block dimension
        """
        self._block_dim_y = block_dim_y

    def get_null_mean(self):
        """Get the resulting empirical null mean filter image

        The empirical null mean filter is also known as the mode filter

        Returns:
            numpy.ndarray: the null mean image
        """
        return self._null_mean

    def get_null_std(self):
        """Get the resulting empirical null std image

        Returns:
            numpy.ndarray: the null std image
        """
        return self._null_std

    def filter(self, image):
        """Filter the image using the empirical null filter

        Filter the image using the empirical null filter. For the empirical null
        mean image (or the mode image), use self._get_null_mean(). Similarly,
        for the empirical null std image, use self._get_null_std().

        Args:
            image (numpy.ndarray): The image to filter

        Returns:
            numpy.ndarray: Empirical null filtered image
        """
        with cupy.cuda.Device() as device:
            module = cupy.RawModule(path=PTX_FILE_PATH)
            self._null_mean, self._null_std = self._filter_image_on_gpu(
                device, module, image)
        return (image-self._null_mean) / self._null_std

    def _filter_image_on_gpu(self, device, module, image):
        """Run the empirical null gpu kernel

        Args:
            device (cupy.cuda.device.Device): device to run on
            module (cupy._core.raw.RawModule): contains the empirical null gpu
                kernel, as well as other constant/parameters
            image (numpy.ndarray): the image to filter on

        Returns:
            numpy.ndarray: null mean image on cpu
            numpy.ndarray: null std image on cpu
        """
        # cache is referred to the image with NaN padding
        cache = self._pad_image(image)
        # require some images before doing the empirical null filter
        null_mean_roi, initial_sigma_roi, bandwidth_roi = (
            self._get_prerequisite_images(image))
        # shared memory is used to store the results of the null mean and null
        # std
        # here, it determines if shared memory is used for the cache too, thus
        # the size of the shared memory
        shared_memory_size, is_copy_cache_to_shared = (
            self._get_shared_memory_size(device, cache))
        # set __constant__ variables here
        self._set_cuda_parameters(module, image.shape, cache,
                                  is_copy_cache_to_shared)

        d_null_mean_roi, d_null_std_roi = self._call_cuda_kernel(
            module, image.shape, cache, initial_sigma_roi, bandwidth_roi,
            null_mean_roi, shared_memory_size)

        # transfer from gpu to cpu
        h_null_mean_roi = cupy.asnumpy(d_null_mean_roi)
        h_null_std_roi = cupy.asnumpy(d_null_std_roi)

        return h_null_mean_roi, h_null_std_roi

    def _pad_image(self, image):
        """Pad the image with NaN

        Pad the image with NaN, the size of the padding is the kernel radius

        Args:
            image (numpy.ndarray): the image to pad

        Returns:
            numpy.ndarray: the image padded
        """
        image_shape = image.shape
        padded_image = np.full(
            (image_shape[0] + 2*self._radius, image_shape[1] + 2*self._radius),
            math.nan
        )
        padded_image[self._radius:(self._radius+image.shape[0]),
                     self._radius:(self._radius+image.shape[1])] = image
        return padded_image

    def _get_prerequisite_images(self, image):
        """Get prerequisite images for the empirical null filter

        Prerequisite images include the median filter, standard deviation filter
        and the bandwidth for each pixel in the image. The bandwidth is obtained
        from the inter-quartile filter.

        Args:
            image (numpy.ndarray): the image to filter

        Returns:
            numpy.ndarray: median filtered image
            numpy.ndarray: standard deviation filtered image (with some zero
                handling)
            numpy.ndarray: image of bandwidth
        """
        footprint = self._kernel.get_footprint()
        std = scipy.ndimage.generic_filter(image, np.std, footprint=footprint)
        std[np.isclose(std, 0)] = self._std_for_zero

        quartile_1 = scipy.ndimage.percentile_filter(
            image, 25, footprint=footprint)
        quartile_2 = scipy.ndimage.median_filter(
            image, footprint=footprint)
        quartile_3 = scipy.ndimage.percentile_filter(
            image, 75, footprint=footprint)

        iqr = (quartile_3 - quartile_1) / 1.34

        bandwidth = np.minimum(std, iqr)
        bandwidth[np.isclose(bandwidth, 0)] = self._std_for_zero
        bandwidth *= (
            self._bandwidth_parameter_b *
            math.pow(self._kernel.get_n_points(), -0.2)
            + self._bandwidth_parameter_a)

        return quartile_2, std, bandwidth

    def _get_shared_memory_size(self, device, cache):
        """Set the size of the shared memory

        Set the size of the shared memory to contain the null mean and null
        std. It will also contain the image and pixels outside the block
        captured by the kernel if the device allows it.

        Args:
            device (cupy.cuda.device.Device): device to run on
            cache (numpy.ndarray): the image padded

        Returns:
            int: the size of the shared memory (in bytes per block)
            int: boolean, True if shared memory is big enough to store the image
                plus pixels captured by the kernel in addition to the null mean
                and null std. False if shared memory is only big enough to store
                the null mean and null std
        """
        # set the shared memory to contain:
        #     - the null mean in the block
        #           size (self._block_dim_x * self._block_dim_y)
        #     - the null std in the block
        #           size (self._block_dim_x * self._block_dim_y)
        #     - the image to filter in the block and outside the block captured
        #           by the kernel
        #           size ((self._block_dim_x + 2 * self._kernel.get_radius())
        #                 * (self._block_dim_y + 2 * self._kernel.get_radius()))
        shared_memory_size = (
            2 * self._block_dim_x * self._block_dim_y
            + (self._block_dim_x + 2 * self._kernel.get_radius())
            * (self._block_dim_y + 2 * self._kernel.get_radius())
        )
        shared_memory_size *= ctypes.sizeof(ctypes.c_float)

        max_shared_size = device.attributes["MaxSharedMemoryPerBlock"]

        # if the requested shared memory is too large, then shared memory
        # only contains the null mean in the block and the null std in the block
        if shared_memory_size > max_shared_size:
            shared_memory_size = (2 * self._block_dim_x * self._block_dim_y
                                    * ctypes.sizeof(ctypes.c_float))
            is_copy_cache_to_shared = int(0)
        else:
            is_copy_cache_to_shared = int(1)

        return shared_memory_size, is_copy_cache_to_shared

    def _set_cuda_parameters(self, module, image_shape, cache,
                             is_copy_cache_to_shared):
        """Set the __constant__ variables in the cuda module

        Set the __constant__ variables in the cuda module. They act as
        parameters

        Args:
            module (cupy._core.raw.RawModule): contains the empirical null gpu
                kernel, as well as other constant/parameters
            image_shape (tuple): size two, the shape or size of the image
            cache (numpy.ndarray): the image padded
            is_copy_cache_to_shared (int): bool, return value of
                _get_shared_memory_size()
        """
        self._set_int_constant(module, "kRoiWidth", image_shape[1])
        self._set_int_constant(module, "kRoiHeight", image_shape[0])
        self._set_int_constant(module, "kCacheWidth",
                               cache.shape[1])
        self._set_int_constant(
            module, "kKernelRadius", self._kernel.get_radius())
        self._set_int_constant(
            module, "kKernelHeight", self._kernel.get_height())
        self._set_int_constant(module, "kNInitial", self._n_initial)
        self._set_int_constant(module, "kNStep", self._n_step)
        self._set_int_constant(
            module, "kIsCopyImageToShared", is_copy_cache_to_shared)

    def _set_int_constant(self, module, constant_name, value):
        """Set the __constant__ variables in the cuda module with an int

        Args:
            module (cupy._core.raw.RawModule): contains the empirical null gpu
                kernel, as well as other constant/parameters
            constant_name (str): the name of the __constant__ variable
            value (int): the value to transfer to the __constant__ variable
        """
        device_var = module.get_global(constant_name)
        host_var = ctypes.c_int32(value)
        device_var.copy_from_host(ctypes.addressof(host_var),
                                  ctypes.sizeof(host_var))

    def _call_cuda_kernel(self, module, image_shape, cache, initial_sigma_roi,
                          bandwidth_roi, null_mean_roi, shared_memory_size):
        """Call the cuda kernel for the empirical null filter

        Args:
            module (cupy._core.raw.RawModule): contains the empirical null gpu
                kernel, as well as other constant/parameters
            image_shape (tuple): size two, the shape or size of the image
            cache (numpy.ndarray): the image padded
            initial_sigma_roi (numpy.ndarray): return value of
                _get_prerequisite_images(), the standard deviation filter of the
                image
            bandwidth_roi (numpy.ndarray): return value of
                _get_prerequisite_images(), the bandwidth for the density
                estimate for each pixel
            null_mean_roi (numpy.ndarray): return value of
                _get_prerequisite_images(), the median filter of the image
            shared_memory_size (int): the size of the shared memory per block
                in bytes

        Returns:
            cupy.ndarray: the null mean image on device
            cupy.ndarray: the null std image on device
        """
        kernel = module.get_function("EmpiricalNullFilter")

        # transfer all parameters to gpu
        d_cache = cupy.asarray(cache, cupy.float32)
        d_initial_sigma_roi = cupy.asarray(initial_sigma_roi, cupy.float32)
        d_bandwidth_roi = cupy.asarray(bandwidth_roi, cupy.float32)
        d_kernel_pointers = cupy.asarray(
            self._kernel.get_pointer(), cupy.int32)
        d_null_mean_roi = cupy.asarray(null_mean_roi, cupy.float32)
        d_null_std_roi = cupy.empty_like(d_null_mean_roi, cupy.float32)
        d_progress_roi = cupy.zeros_like(d_null_mean_roi, cupy.int32)

        kernel_args = (
            d_cache, d_initial_sigma_roi, d_bandwidth_roi,
            d_kernel_pointers, d_null_mean_roi, d_null_std_roi, d_progress_roi
        )

        # get number of blocks to run
        n_block_x = (
            (image_shape[1] + self._block_dim_x - 1) // self._block_dim_x)
        n_block_y = (
            (image_shape[0] + self._block_dim_y - 1) // self._block_dim_y)

        # call cuda kernel and wait for results
        kernel((n_block_x, n_block_y), (self._block_dim_x, self._block_dim_x),
               kernel_args, shared_mem=shared_memory_size)
        cupy.cuda.runtime.deviceSynchronize()

        return d_null_mean_roi, d_null_std_roi


class _Kernel:
    """A circular kernel which captures the local pixels

    Attributes:
        _kernel_radius (int): the radius of the kernel as an integer
        _kernel_n_points (int): the number of pixels this kernel captures
        _kernel_pointer (numpy.ndarray): an array of integers, length
            2 * _kernel_radius, indicates for each row the starting and ending
            column position from the centre of the kernel
        _kernel_height (int): the height of the kernel
    """

    def __init__(self, radius):
        self._kernel_radius = None
        self._kernel_n_points = None
        self._kernel_pointer = None
        self._kernel_height = None

        # see Kernel.java for these special cases
        if (radius >= 1.5 and radius < 1.75):
            radius = 1.75
        elif (radius >= 2.5 and radius < 2.85):
            radius = 2.85

        radius_squared = math.floor((radius * radius) + 1)
        self._kernel_radius = math.floor(math.sqrt(radius_squared + 1e-10))
        self._kernel_height = 2 * self._kernel_radius + 1

        self._kernel_pointer = np.empty(
            2 * self._kernel_height, dtype=np.int32)

        self._kernel_pointer[2 * self._kernel_radius] = -self._kernel_radius
        self._kernel_pointer[2 * self._kernel_radius + 1] = self._kernel_radius

        self._kernel_n_points = 2 * self._kernel_radius + 1
        # lines above and below centre together
        for row_id in range(self._kernel_radius):
            row_id += 1
            dist = math.floor(
                math.sqrt(radius_squared - row_id * row_id + 1e-10))
            self._kernel_pointer[2 * (self._kernel_radius - row_id)] = -dist
            self._kernel_pointer[2 * (self._kernel_radius - row_id) + 1] = dist
            self._kernel_pointer[2 * (self._kernel_radius + row_id)] = -dist
            self._kernel_pointer[2 * (self._kernel_radius + row_id) + 1] = dist
            # 2*dx+1 for each line, above&below
            self._kernel_n_points += 4 * dist + 2

    def get_radius(self):
        return self._kernel_radius

    def get_n_points(self):
        return self._kernel_n_points

    def get_pointer(self):
        return self._kernel_pointer

    def get_height(self):
        return self._kernel_height

    def get_footprint(self):
        """Return the footprint of the kernel, used by scipy.ndimage functions

        Returns:
            numpy.ndarray: array of booleans, True if this pixel shall be
                considered in the filter, else False
        """
        footprint = np.zeros(
            (self._kernel_height, self._kernel_height), np.bool_)
        for i_row in range(self._kernel_height):
            j_start = self._kernel_pointer[2*i_row] + self._kernel_radius
            j_end = self._kernel_pointer[2*i_row + 1] + 1 + self._kernel_radius
            footprint[i_row, j_start:j_end] = True
        return footprint
