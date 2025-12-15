import pytest
import numpy as np
import skimage.data

import modefilter

FLOAT_BLACK = 0.0  # value of black pixel in np.float32 image
FLOAT_GREY = 0.5  # value of grey pixel in np.float32 image
FLOAT_WHITE = 1.0  # value of white pixel in np.float32 image


@pytest.fixture
def rng(seed):
    return np.random.default_rng(seed)


def white_value(dtype):
    """Value of white pixel"""
    if dtype == np.float32:
        return FLOAT_WHITE
    else:
        return np.iinfo(dtype).max


def grey_value(dtype):
    """Value of grey pixel"""
    if dtype == np.float32:
        return FLOAT_GREY
    else:
        return np.iinfo(dtype).max // 2


def black_value(dtype):
    """Value of black pixel"""
    if dtype == np.float32:
        return FLOAT_BLACK
    else:
        return np.iinfo(dtype).min


@pytest.fixture
def white(shape, dtype):
    """Image of just white pixels"""
    return np.full(shape, white_value(dtype), dtype)


@pytest.fixture
def black(shape, dtype):
    """Image of just black pixels"""
    return np.full(shape, black_value(dtype), dtype)


@pytest.fixture
def grey(shape, dtype):
    """Image of just grey pixels"""
    return np.full(shape, grey_value(dtype), dtype)


@pytest.fixture
def random(shape, dtype, rng):
    """Image of random value pixels"""
    if dtype == np.float32:
        return rng.standard_normal(shape).astype(dtype)
    else:
        return rng.integers(
            np.iinfo(dtype).min, np.iinfo(dtype).max + 1, shape, dtype
        )


@pytest.fixture
def coffee():
    """Sample RGB image

    Sample RGB image, dtype uint8
    """
    return skimage.data.coffee()[:, :, 0]


@pytest.fixture
def brain():
    """Sample uint16 image"""
    return skimage.data.brain()[6, :, :]


@pytest.fixture
def sheep():
    """Sample float64 image

    Sample float64 image, values [0.0, 1.0]
    """
    return skimage.data.shepp_logan_phantom()


def run_filter(
    filter_cls,
    image,
    radius,
    n_initial=None,
    n_step=None,
    bandwidth_a=None,
    bandwidth_b=None,
    block_dim_x=None,
    block_dim_y=None,
):
    """Run and test the filter

    Given a class, instantiate a filter and run it on the image. The following
    are tested:

    - The passed image remained unchanged
    - The returned images has the same shape as the passed image
    - There are no NaN in the null mean

    Optional parameters can be passed

    Args:
        filter_cls (type): Class to instantiate a filter from, eg
            EmpiricalNullFilter, ModeFilter
        image (numpy.ndarray): The image to filter
        radius (float): Radius of the kernel. Positive real number
        n_initial (int, optional): Number of initial points to try. Defaults to
            None.
        n_step (int, optional): Number of steps to take in the Newton-Raphson
            method. Defaults to None.
        bandwidth_a (float, optional): Bandwidth parameter A for density
            estimate. Defaults to None.
        bandwidth_b (float, optional): Bandwidth parameter B for density
            estimate. Defaults to None.
        block_dim_x (int, optional): Block x dimension for the GPU grid
            configuration. Defaults to None.
        block_dim_y (int, optional): Block y dimension for the GPU grid
            configuration. Defaults to None.

    Returns:
        np.ndarray: Filtered image
        np.ndarray: Null mean (or mode) image
        np.ndarray: Null std image
    """
    # make copy of image
    image_before = image.copy()

    shape = image.shape
    image_filter = filter_cls(radius)

    if n_initial is not None:
        image_filter.set_n_initial(n_initial)
    if n_step is not None:
        image_filter.set_n_step(n_step)
    if bandwidth_a is not None:
        image_filter.set_bandwidth_parameter_a(bandwidth_a)
    if bandwidth_b is not None:
        image_filter.set_bandwidth_parameter_b(bandwidth_b)
    if block_dim_x is not None:
        image_filter.set_block_dim_x(block_dim_x)
    if block_dim_y is not None:
        image_filter.set_block_dim_y(block_dim_y)

    image_after = image_filter.filter(image)
    null_mean = image_filter.get_null_mean()
    null_std = image_filter.get_null_std()

    # test if image input remains unchanged
    assert np.all(np.isclose(image, image_before))

    # test shape is correct
    assert image_after.shape == shape
    assert null_mean.shape == shape
    assert null_std.shape == shape

    # no nan in null mean
    assert np.sum(np.isnan(null_mean)) == 0

    return image_after, null_mean, null_std


class TestModeFilter:
    @pytest.mark.parametrize("shape", [(1, 1), (16, 16), (200, 200)])
    @pytest.mark.parametrize(
        "dtype", [np.float32, np.int32, np.uint32, np.uint8]
    )
    @pytest.mark.parametrize("radius", [0.01, 1, 5, 10])
    def test_modefilter_white(self, white, radius, dtype):
        """Test the mode filter on a white image

        On a flat image, the filter should have no effect
        """
        image, _, _ = run_filter(modefilter.ModeFilter, white, radius)
        assert np.all(np.isclose(image, white_value(dtype)))

    @pytest.mark.parametrize("shape", [(1, 1), (16, 16), (200, 200)])
    @pytest.mark.parametrize(
        "dtype", [np.float32, np.int32, np.uint32, np.uint8]
    )
    @pytest.mark.parametrize("radius", [0.01, 1, 5, 10])
    def test_modefilter_black(self, black, radius, dtype):
        """Test the mode filter on a black image

        On a flat image, the filter should have no effect
        """
        image, _, _ = run_filter(modefilter.ModeFilter, black, radius)
        assert np.all(np.isclose(image, black_value(dtype)))

    @pytest.mark.parametrize("shape", [(1, 1), (16, 16), (200, 200)])
    @pytest.mark.parametrize(
        "dtype", [np.float32, np.int32, np.uint32, np.uint8]
    )
    @pytest.mark.parametrize("radius", [0.01, 1, 5, 10])
    def test_modefilter_grey(self, grey, radius, dtype):
        """Test the mode filter on a grey image

        On a flat image, the filter should have no effect
        """
        image, _, _ = run_filter(modefilter.ModeFilter, grey, radius)
        assert np.all(np.isclose(image, grey_value(dtype)))

    @pytest.mark.parametrize("shape", [(1, 1), (16, 16), (200, 200)])
    @pytest.mark.parametrize(
        "dtype", [np.float32, np.int32, np.uint32, np.uint8]
    )
    @pytest.mark.parametrize("radius", [0.01, 1, 5, 10])
    @pytest.mark.parametrize("seed", [223776772019103709826684953708207922138])
    def test_modefilter_random(self, random, radius):
        """Test the mode filter on a random image"""
        run_filter(modefilter.ModeFilter, random, radius)

    @pytest.mark.parametrize("shape", [(1, 1), (16, 16), (200, 200)])
    @pytest.mark.parametrize(
        "dtype", [np.float32, np.int32, np.uint32, np.uint8]
    )
    @pytest.mark.parametrize("radius", [0.01, 1, 5, 10])
    @pytest.mark.parametrize("seed", [86214657484624831472773009300617983166])
    def test_modefilter_reproduce(self, random, radius):
        """Test if the mode filter can reproduce results

        Use the mode filter on the same image twice, should produce the same
        result
        """
        image0, _, _ = run_filter(modefilter.ModeFilter, random, radius)
        image1, _, _ = run_filter(modefilter.ModeFilter, random, radius)
        assert np.all(np.isclose(image0, image1))

    @pytest.mark.parametrize("radius", [0.01, 1, 5, 10])
    @pytest.mark.parametrize("n_initial", [50])
    def test_modefilter_coffee(self, coffee, radius, n_initial):
        """Test the mode filter on a sample RGB image

        Test if the mode filter at least modifies the image
        """
        filtered = run_filter(modefilter.ModeFilter, coffee, radius, n_initial)
        assert np.any(np.logical_not(np.isclose(coffee, filtered)))

    @pytest.mark.parametrize("radius", [0.01, 1, 5, 10])
    @pytest.mark.parametrize("n_initial", [50])
    def test_modefilter_brain(self, brain, radius, n_initial):
        """Test the mode filter on a sample unit16 image

        Test if the mode filter at least modifies the image
        """
        filtered = run_filter(modefilter.ModeFilter, brain, radius, n_initial)
        assert np.any(np.logical_not(np.isclose(brain, filtered)))

    @pytest.mark.parametrize("radius", [0.01, 1, 5, 10])
    @pytest.mark.parametrize("n_initial", [50])
    def test_modefilter_sheep(self, sheep, radius, n_initial):
        """Test the mode filter on a sample float32 image

        Test if the mode filter at least modifies the image
        """
        filtered = run_filter(modefilter.ModeFilter, sheep, radius, n_initial)
        assert np.any(np.logical_not(np.isclose(sheep, filtered)))

    @pytest.mark.parametrize("shape", [(16, 16)])
    @pytest.mark.parametrize(
        "dtype", [np.float32, np.int32, np.uint32, np.uint8]
    )
    @pytest.mark.parametrize("radius", [5])
    @pytest.mark.parametrize("n_initial", [None, 10])
    @pytest.mark.parametrize("n_step", [None, 20])
    @pytest.mark.parametrize("bandwidth_a", [None, 0.12])
    @pytest.mark.parametrize("bandwidth_b", [None, 0.1])
    @pytest.mark.parametrize("block_dim_x", [1, 2, 4, 8, 16, 32])
    @pytest.mark.parametrize("block_dim_y", [16, 32])
    @pytest.mark.parametrize("seed", [59728372820829031143570037554818138128])
    def test_modefilter_optional(
        self,
        random,
        radius,
        n_initial,
        n_step,
        bandwidth_a,
        bandwidth_b,
        block_dim_x,
        block_dim_y,
    ):
        """Test passing optional parameters to the mode filter"""
        run_filter(
            modefilter.ModeFilter,
            random,
            radius,
            n_initial,
            n_step,
            bandwidth_a,
            bandwidth_b,
            block_dim_x,
            block_dim_y,
        )


class TestEmpiricalNullFilter:
    @pytest.mark.parametrize("shape", [(1, 1), (16, 16), (200, 200)])
    @pytest.mark.parametrize("dtype", [np.float32])
    @pytest.mark.parametrize("radius", [0.01, 1, 5, 10])
    def test_empirical_null_filter_white(self, white, radius):
        """Test the empirical null filter on a white image"""
        run_filter(modefilter.EmpiricalNullFilter, white, radius)

    @pytest.mark.parametrize("shape", [(1, 1), (16, 16), (200, 200)])
    @pytest.mark.parametrize("dtype", [np.float32])
    @pytest.mark.parametrize("radius", [0.01, 1, 5, 10])
    def test_empirical_null_filter_black(self, black, radius):
        """Test the empirical null filter on a black image"""
        run_filter(modefilter.EmpiricalNullFilter, black, radius)

    @pytest.mark.parametrize("shape", [(1, 1), (16, 16), (200, 200)])
    @pytest.mark.parametrize("dtype", [np.float32])
    @pytest.mark.parametrize("radius", [0.01, 1, 5, 10])
    def test_empirical_null_filter_grey(self, grey, radius):
        """Test the empirical null filter on a grey image"""
        run_filter(modefilter.EmpiricalNullFilter, grey, radius)

    @pytest.mark.parametrize("shape", [(1, 1), (16, 16), (200, 200)])
    @pytest.mark.parametrize("dtype", [np.float32])
    @pytest.mark.parametrize("radius", [0.01, 1, 5, 10])
    @pytest.mark.parametrize("seed", [223776772019103709826684953708207922138])
    def test_empirical_null_filter_random(self, random, radius):
        """Test the empirical null filter on a random image"""
        run_filter(modefilter.EmpiricalNullFilter, random, radius)

    @pytest.mark.parametrize("shape", [(1, 1), (16, 16), (200, 200)])
    @pytest.mark.parametrize("dtype", [np.float32])
    @pytest.mark.parametrize("radius", [0.01, 1, 5, 10])
    @pytest.mark.parametrize("seed", [86214657484624831472773009300617983166])
    def test_empirical_null_filter_reproduce(self, random, radius):
        """Test if the empirical null filter can reproduce results

        Use the empirical null filter on the same image twice, should produce
        the same result, including the null mean and null std images
        """
        image0, mean0, std0 = run_filter(
            modefilter.EmpiricalNullFilter, random, radius
        )
        image1, mean1, std1 = run_filter(
            modefilter.EmpiricalNullFilter, random, radius
        )
        # nan can happen in null_std if the newton-raphson method fails
        # this carries forward to the empirical null filter
        assert np.all(np.isclose(image0, image1, equal_nan=True))
        assert np.all(np.isclose(mean0, mean1))
        assert np.all(np.isclose(std0, std1, equal_nan=True))

    @pytest.mark.parametrize("radius", [0.01, 1, 5, 10])
    @pytest.mark.parametrize("n_initial", [50])
    def test_empirical_null_filter_coffee(self, coffee, radius, n_initial):
        """Test the empirical null filter on a sample RGB image"""
        run_filter(modefilter.EmpiricalNullFilter, coffee, radius, n_initial)

    @pytest.mark.parametrize("radius", [0.01, 1, 5, 10])
    @pytest.mark.parametrize("n_initial", [50])
    def test_empirical_null_filter_brain(self, brain, radius, n_initial):
        """Test the empirical null filter on a sample unit16 image"""
        run_filter(modefilter.EmpiricalNullFilter, brain, radius, n_initial)

    @pytest.mark.parametrize("radius", [0.01, 1, 5, 10])
    @pytest.mark.parametrize("n_initial", [50])
    def test_empirical_null_filter_sheep(self, sheep, radius, n_initial):
        """Test the empirical null filter on a sample float32 image"""
        run_filter(modefilter.EmpiricalNullFilter, sheep, radius, n_initial)

    @pytest.mark.parametrize("shape", [(16, 16)])
    @pytest.mark.parametrize("dtype", [np.float32])
    @pytest.mark.parametrize("radius", [5])
    @pytest.mark.parametrize("n_initial", [None, 10])
    @pytest.mark.parametrize("n_step", [None, 20])
    @pytest.mark.parametrize("bandwidth_a", [None, 0.12])
    @pytest.mark.parametrize("bandwidth_b", [None, 0.1])
    @pytest.mark.parametrize("block_dim_x", [1, 2, 4, 8, 16, 32])
    @pytest.mark.parametrize("block_dim_y", [16, 32])
    @pytest.mark.parametrize("seed", [59728372820829031143570037554818138128])
    def test_empirical_null_filter_optional(
        self,
        random,
        radius,
        n_initial,
        n_step,
        bandwidth_a,
        bandwidth_b,
        block_dim_x,
        block_dim_y,
    ):
        """Test passing optional parameters to the empirical null filter"""
        run_filter(
            modefilter.EmpiricalNullFilter,
            random,
            radius,
            n_initial,
            n_step,
            bandwidth_a,
            bandwidth_b,
            block_dim_x,
            block_dim_y,
        )


@pytest.mark.parametrize("shape", [(1, 1), (16, 16), (200, 200)])
@pytest.mark.parametrize("dtype", [np.float32, np.int32, np.uint32, np.uint8])
@pytest.mark.parametrize("radius", [0.01, 1, 5, 10])
@pytest.mark.parametrize("seed", [339183785153159183696937800705473139952])
def test_mode_in_empirical_null(random, radius):
    """Test mode filter in EmpiricalNullFilter

    Test if ModeFilter gives the same result as the mode filtered image in
    EmpiricalNullFilter
    """
    image0, _, _ = run_filter(modefilter.ModeFilter, random, radius)
    _, image1, _ = run_filter(modefilter.EmpiricalNullFilter, random, radius)
    assert np.all(np.isclose(image0, image1))
