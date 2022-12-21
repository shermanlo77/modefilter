# Mode Filter and Empirical Null Filter

* MIT License - all source code
* Copyright (c) 2020-2023 Sherman Lo

*ImageJ* plugins for the mode filter and empirical null filter. The mode filter
is an edge preserving smoothing filter by taking the mode of the empirical
density. This may have applications in image processing such as image
segmentation. The filters were also implemented on a *Nvidia* GPU using *CUDA*
and *JCuda*. This speeds up the filtering by a huge margin.

Where appropriate, please cite the thesis Lo, S.E. (2020). *Characterisation of
Computed Tomography Noise in Projection Space with Applications to Additive
Manufacturing*. PhD thesis, University of Warwick, Department of Statistics.

![images of a Mandrill with a mode filter, of varying radius kernel,
applied](mandrillExample.jpg)
The mode filter applied on the
[Mandrill test image](http://sipi.usc.edu/database/database.php?volume=misc).
Top left to top right, bottom left to bottom right: mandrill test image, then
the mode filter with radius of 2, 4, 8, 16, 32, 64, 128 applied.

## How to Compile (Linux recommended)

Requires *Maven* as well as *Java Runtime Environment* and *Java Development
Kit*. For the use of a *Nvidia* GPU, requires *CUDA Development Kit* which
should include a *nvcc* compiler.

Clone this repository.

Optional for GPU: go to `cuda/` and run `make` to compile the *CUDA* code into
a `.ptx` file.

At `pom.xml`, run

```shell
mvn package
```

to compile the *Java* code. The compiled `.jar` file is
`target/Empirical_Null_Filter-X.X.X.jar` and can be used as an *ImageJ* plugin.
Copies of required libraries are stored in `target/libs/` and would need to be
installed in *ImageJ* as well.

The required `.jar` files shall be provided in the releases.

## How to Install (*Fiji* recommended)

The required `.jar` files can be obtained by either compiling (CPU and GPU) or
downloading from the releases (CPU only).

Installation of `Empirical_Null_Filter-X.X.X.jar` can be done by copying the
file into *Fiji*'s `plugins/` directory or, in *Fiji*, using the *Plugins* menu
followed by *Install...* (or Ctrl + Shift + M).

The required `.jar` libraries are to be copied into *Fiji*'s `jars/` directory.
They are:

* `commons-math3-3.6.1.jar` (may already be provided)
* `jcuda-10.1.0.jar` (for GPU usage)
* `jcuda-natives-10.1.0-linux-x86_64.jar` (or similar for GPU usage)

## Options

![Screenshot of the GUI](filter_gui.png)

* Number of initial values
  * Number of initial values for the Newton-Raphson method. Increase this for a
    more accurate filtering at a price of more computational time. Compared to
    other options, this has a big effort on the resulting image. The default
    value is 3 but should be in the order of 50-100 if this filter is to be
    applied to (non-Gaussian) images.
* Number of steps
  * Number of iterations in the Newton-Raphson method. Increase this for a more
    accurate filtering at a price of more computational time.
* Log tolerance (CPU version only)
  * The tolerance allowed for the Newton-Raphson method to accept the solution.
    Decrease this for a more accurate filtering at a price of more computational
    time.
* Block dim x and y (GPU version only)
  * Sets the dimensions of the block of threads on the GPU. This affects the
    performance of the filter. Good suggestions are 16 and 32. Solutions are
    shared between neighbours within blocks.

## Using the Mode Filter via Terminal

The mode filter can be used via the terminal. Go to `target/` and run the
`.jar` file. To use a GUI for parameter selection

```shell
java -jar Empirical_Null_Filter-x.x.x.jar gui ['cpu' or 'gpu'] <loc of image to filter>
```

This will make a GUI appear to select your parameters. Once selected, click OK
to filter the image. A dialogue box will appear to save the resulting image in
`.png` format.

To run the mode filter without a GUI

```shell
java -jar Empirical_Null_Filter-x.x.x.jar run ['cpu' or 'gpu'] \
<loc of image to filter> <loc to save resulting .png> [options]
```

where the options are

* `-r` radius of kernel
* `-i` number of initial points for Newton-Raphson
* `-s` number of steps for Newton-Raphson
* `-t` stopping condition tolerance for Newton-Raphson (recommend negative
  number), only for CPU]
* `-x` x block dimension, only for GPU
* `-y` y block dimension, only for GPU

## About the Mode Filter

The mode filter is an image filter much like the mean filter and median filter.
They process each pixel in an image. For a given pixel, the value of the pixel
is replaced by the mean or median over all pixels within a distance *r* away.
The mean and median filter can be used in *ImageJ*, it results in a smoothing of
the image.

![Mean, median and mode filter applied to an image of a Mandrill](filters.jpg)
Top left:
[Mandrill test image](http://sipi.usc.edu/database/database.php?volume=misc).
Top right: Mean filter with radius 32. Bottom left: Median filter with
radius 32. Bottom right: Mode filter with radius 32.

The mode filter is a by-product of the empirical null filter. Instead of taking
the mean or median, the mode is taken, more specifically, the argmax of the
empirical density. The optimisation problem was solved using the Newton-Raphson
method. Various random initial values were tried to home in on the global
maximum. Because the filtered image is expected to be smooth, the different
initial values were influenced by neighbouring pixels to aid in the optimisation
problem.

The resulting mode filtered image gives a smoothed image which has an impasto
effect and preserved edges. This may have applications in noise removal or image
segmentation.

The mode filtered was implemented on the CPU by modifying existing *Java* code
from *ImageJ*. Each thread filters a row of the image in parallel from left to
right. The solution to one pixel is passed to the pixel to the right. The filter
was also implemented on the GPU by writing *CUDA* code which can be compiled and
read by the *JCuda* package. The image is split into blocks. Within a block,
each thread filters a pixel and share its answer to neighbouring pixels within
that block.

One difficulty is that with the introduction of *CUDA* code, the ability to
"compile once, run anywhere" is difficult to keep hold of. A design choice was
that the user is to compile the *CUDA* code into a `.ptx` file. This is then
followed by compiling the *Java* code with the `.ptx` file into a `.jar` file
which can be installed as a Plugin in *ImageJ* or *Fiji*. The compiled `.jar`
file can be used by *MATLAB* as well.

## Further Reading and References

* Lo, S.E. (2020). *Characterisation of Computed Tomography Noise in Projection
  Space with Applications to Additive Manufacturing*. PhD thesis, University of
  Warwick, Department of Statistics.
* Efron, B. (2004). Large-scale simultaneous hypothesis testing: The choice of a
  null hypothesis. *Journal of the American Statistical Association*,
  99(465):96.
* Griffin, L. D. (2000). Mean, median and mode filtering of images. *Proceedings
  of the Royal Society of London A: Mathematical, Physical and Engineering
  Sciences*, 456(2004):2995–3004.
* Charles, D. and Davies, E. R. (2003). Properties of the mode filter when
  applied to colour images. *International Conference on Visual Information
  Engineering VIE 2003*, pp. 101-104.
