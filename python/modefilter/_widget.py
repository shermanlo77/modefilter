"""
Containers for napari plugins

Containers for napari plugins. The mode filter is implemented in
`ModeFilterContainer` and the empirical null filter in
`EmpiricalNullFilterContainer`

For these containers to be usable in napari, see the file `napari.yaml`

The following images are supported:

    - Grayscale image (int and float up to 32 bit)
    - Grayscale image stack (int and float up to 32 bit), each slice is filtered
      independently
    - RGB image, each channel is treated as a slice or a separate image

`ModeFilterContainer` will return an image of the same type as the input, while
`EmpiricalNullFilterContainer` will return a float32 image

The class `BaseContainer` is an abstract class. All widgets are created here
and the subclasses can extend this class to retrive user inputs from those
widgets
"""

import math
from typing import TYPE_CHECKING

from magicgui import widgets
import numpy as np

import modefilter

if TYPE_CHECKING:
    import napari


class BaseContainer(widgets.Container):
    """Abstract class for empirical null filter napari containers

    Abstract class for empirical null filter napari containers. It filters an
    image, or a stack of them, and outputs the filtered image. It can also
    output additional images, such as the empirical null mean and std if the
    filter supports it

    The following images are supported:

    - Grayscale image (int and float up to 32 bit)
    - Grayscale image stack (int and float up to 32 bit), each slice is filtered
      independently
    - RGB image, each channel is treated as a slice or a separate image

    In the constructor, all widgets are created and saved as member variables.
    Implementations of this class can extend itself to include those widgets

    Subclasses will need to implement the following methods:

    - _output_results()
    - _init_filter()

    For this container to be usable in napari, see the file `napari.yaml`

    Attributes:
        _viewer: user's interface for viewing images
        _image_layer_combo: widget for selecting the image layer
        _radius_widget: widget for specifying the radius of the filter
        _output_null_mean: widget for specifying whether to output the null mean
        _output_null_std: widget for specifying whether to output the null std
        _n_initial_widget: widget for specifying the number of initial values
        _n_step_widget: widget for specifying the number of steps
        _x_block_widget: widget for specifying the x block dimension
        _y_block_widget: widget for specifying the y block dimension
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Abstract class for empirical null filter napari containers

        Args:
            viewer (napari.viewer.Viewer): user's interface for viewing images
        """
        super().__init__()
        self._viewer = viewer
        self._image_layer_combo = widgets.create_widget(
            label="Image", annotation="napari.layers.Image"
        )

        self._radius_widget = widgets.create_widget(
            label="Radius (px)",
            annotation=float,
            widget_type="FloatSpinBox",
            value=2,
        )
        self._radius_widget.step = 1

        self._output_null_mean = widgets.create_widget(
            label="Output null mean",
            annotation=bool,
            widget_type="CheckBox",
            value=False,
        )

        self._output_null_std = widgets.create_widget(
            label="Output null std",
            annotation=bool,
            widget_type="CheckBox",
            value=False,
        )

        self._n_initial_widget = widgets.create_widget(
            label="Number of initial values",
            annotation=int,
            widget_type="SpinBox",
            value=3,
        )
        self._n_initial_widget.min = 1

        self._n_step_widget = widgets.create_widget(
            label="Number of steps",
            annotation=int,
            widget_type="SpinBox",
            value=10,
        )
        self._n_step_widget.min = 1

        self._x_block_widget = widgets.create_widget(
            label="Block dim x", annotation=int, widget_type="SpinBox", value=16
        )
        self._x_block_widget.min = 1

        self._y_block_widget = widgets.create_widget(
            label="Block dim y", annotation=int, widget_type="SpinBox", value=16
        )
        self._y_block_widget.min = 1

        self._run = widgets.create_widget(
            label="Run",
            annotation="napari.types.LabelsData",
            widget_type="PushButton",
        )
        self._run.clicked.connect(self.filter)

    def filter(self):
        """Filter the image

        Filter the image, or images, in the viewer using the provided filter in
        _get_filter(). They are then outputted using _output_results()

        Raises:
            ValueError: if the image type is not supported, only grayscale,
            grayscale stack or an RGB image are supported
        """
        image_layer = self._image_layer_combo.value
        if image_layer is None:
            return

        # the empirical null filter and the mode filter works on float32
        image = image_layer.data.copy()
        image = image.astype(np.float32)

        # declare and allocate null_mean and null_std when requested
        null_mean = None
        null_std = None
        if self._output_null_mean.value:
            null_mean = np.zeros(image.shape, dtype=np.float32)
        if self._output_null_std.value:
            null_std = np.zeros(image.shape, dtype=np.float32)

        image_filter = self._get_filter()

        if image_layer.rgb:
            # in rgb, the last dimension is the colour channel
            for i in range(3):
                self._filter_slice(
                    image_filter,
                    (slice(None), slice(None), i),
                    image,
                    null_mean,
                    null_std,
                )
        elif image.ndim == 2:
            # one grayscale image
            self._filter_slice(
                image_filter,
                (slice(None), slice(None)),
                image,
                null_mean,
                null_std,
            )
        elif image.ndim == 3:
            # stack of grayscale images
            for i in range(image.shape[0]):
                self._filter_slice(
                    image_filter,
                    (i, slice(None), slice(None)),
                    image,
                    null_mean,
                    null_std,
                )
        else:
            raise ValueError("Image type not supported")

        self._output_results(image_layer, image, null_mean, null_std)

    def _filter_slice(
        self, image_filter, slice_index_args, image, null_mean, null_std
    ):
        """Filter an image (or a slice) from a stack of images

        Filter an image (or a slice) from a stack of images using the provided
        filter.

        This method is required as the dimension of the stack can vary from
        image type to image type. For example, for a stack of grayscale images,
        the zeroth dimension is the image index. For a RGB image, the last
        dimension is the colour channel

        Args:
            image_filter (modefilter.EmpiricalNullFilter): an object which can
                filter an image and also output the empirical null mean and std
                when if requested
            slice_index_args (list): a list of slice or ints, this specifies
                an image to filter from the stack
            image (numpy.ndarray): either an image (2d) or a stack of images
                (3d)
            null_mean (bool): if the user requests the null mean image as an
                output
            null_std (bool): if the user requests the null std image as an
                output
        """
        image[*slice_index_args] = image_filter.filter(image[*slice_index_args])
        if null_mean is not None:
            null_mean[*slice_index_args] = image_filter.get_null_mean()
        if null_std is not None:
            null_std[*slice_index_args] = image_filter.get_null_std()

    def _output_results(self, image_layer, image, null_mean, null_std):
        """Output requested images after filtering

        Output requested images after filtering. Different filters may output
        different images

        Args:
            image_layer (napari.layers.image.image.Image): image before
                filtering
            image (numpy.ndarray): image after filtering
            null_mean (numpy.ndarray): The null mean image. Can be None if not
                requested
            null_std (numpy.ndarray): The null std image. Can be None if not
                requested

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def _get_filter(self):
        """Instantiates and sets up a filter

        Instantiates and sets up a filter, using the values the user provided
        in the widgets

        Raises:
            ValueError: if the radius is 0

        Returns:
            modefilter.EmpiricalNullFilter: a filter
        """
        radius = self._radius_widget.value
        n_initial = self._n_initial_widget.value
        n_step = self._n_step_widget.value
        x_block = self._x_block_widget.value
        y_block = self._y_block_widget.value

        if math.isclose(radius, 0):
            raise ValueError("Radius must be greater than 0")

        image_filter = self._init_filter(radius)
        image_filter.set_n_initial(n_initial)
        image_filter.set_n_step(n_step)
        image_filter.set_block_dim_x(x_block)
        image_filter.set_block_dim_y(y_block)

        return image_filter

    def _init_filter(self, *args, **kwargs):
        """Instantiates a filter

        To be implemented, different subclasses may instantiate different
        filters

        Raises:
            NotImplementedError
        """
        raise NotImplementedError


class ModeFilterContainer(BaseContainer):
    """Container for the mode filter

    Container for the mode filter, aka the null mean filter.
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Container for the mode filter

        Args:
            viewer (napari.viewer.Viewer): user's interface for viewing images
        """
        super().__init__(viewer)
        self.extend(
            [
                self._image_layer_combo,
                self._radius_widget,
                widgets.create_widget(
                    label="Advanced options", widget_type="Label"
                ),
                self._n_initial_widget,
                self._n_step_widget,
                widgets.create_widget(label="GPU options", widget_type="Label"),
                self._x_block_widget,
                self._y_block_widget,
                self._run,
            ]
        )

    def _output_results(self, image_layer, image, null_mean, null_std):
        """Output the null mean only

        Output the null mean only, which is the resulting filter. It will
        return an image of the same type as the input

        Args:
            image_layer (napari.layers.image.image.Image): image before
                filtering
            image (numpy.ndarray): image after filtering
            null_mean: Not used
            null_std: Not used
        """
        image_type = image_layer.data.dtype
        image = image.astype(image_type)
        name = image_layer.name + "_modefilter"

        if name in self._viewer.layers:
            self._viewer.layers[name].data = image
        else:
            self._viewer.add_image(image, name=name)

    def _init_filter(self, *args, **kwargs):
        return modefilter.ModeFilter(*args, **kwargs)


class EmpiricalNullFilterContainer(BaseContainer):
    """Container for the empirical null filter

    Container for the empirical null filter. It also as the option to output
    the null mean image and the null std image
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Container for the empirical null filter

        Args:
            viewer (napari.viewer.Viewer): user's interface for viewing images
        """
        super().__init__(viewer)
        self.extend(
            [
                self._image_layer_combo,
                self._radius_widget,
                widgets.create_widget(
                    label="Additional outputs", widget_type="Label"
                ),
                self._output_null_mean,
                self._output_null_std,
                widgets.create_widget(
                    label="Advanced options", widget_type="Label"
                ),
                self._n_initial_widget,
                self._n_step_widget,
                widgets.create_widget(label="GPU options", widget_type="Label"),
                self._x_block_widget,
                self._y_block_widget,
                self._run,
            ]
        )

    def _output_results(self, image_layer, image, null_mean, null_std):
        """Output the filtered image as well as optional ones

        Output the filtered image as well as, if requested, the null mean image
        and the null std image. The output images shall be left as float32

        Args:
            image_layer (napari.layers.image.image.Image): image before
                filtering
            image (numpy.ndarray): image after filtering
            null_mean (numpy.ndarray): The null mean image. Can be None if not
                requested
            null_std (numpy.ndarray): The null std image. Can be None if not
                requested
        """
        output_list = [[image, "_empirical_null_filter"]]
        if null_mean is not None:
            output_list.append([null_mean, "_null_mean"])
        if null_std is not None:
            output_list.append([null_std, "_null_std"])

        for output in output_list:
            name = f"{image_layer.name}{output[1]}"
            if name in self._viewer.layers:
                self._viewer.layers[name].data = output[0]
            else:
                self._viewer.add_image(output[0], name=name)

    def _init_filter(self, *args, **kwargs):
        return modefilter.EmpiricalNullFilter(*args, **kwargs)
