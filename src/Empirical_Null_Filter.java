// MIT License
// Copyright (c) 2019-2023 Sherman Lo

import ij.ImagePlus;
import ij.io.Opener;
import ij.io.FileSaver;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.plugin.filter.PlugInFilterRunner;

import uk.ac.warwick.sip.empiricalnullfilter.EmpiricalNullFilter;
import uk.ac.warwick.sip.empiricalnullfilter.ModeFilter;
import uk.ac.warwick.sip.empiricalnullfilter.ModeFilterGpu;

/**
 * Wrapper entry point for the mode filter and class usable by ImageJ
 *
 * The class name has underscores so that it can be used by ImageJ.
 *
 * A main() is provided which calls the mode filter using either the CPU or GPU, using the command
 * interface or a GUI.
 *
 * To use the GUI, provide the following arguments:
 * <ul>
 * <li>"gui"</li>
 * <li>either "cpu" or "gpu"</li>
 * <li>image file to run filter on</li>
 * </ul>
 *
 * To use the command interface, provide the following arguments:
 * <ul>
 * <li>"run"</li>
 * <li>either "cpu" or "gpu"</li>
 * <li>image file to run filter on</li>
 * <li>path to save filtered image to in .png format</li>
 * <li>[-r radius of the kernel]</li>
 * <li>[-n number of threads]</li>
 * <li>[-i number of initial points for Newton-Raphson]</li>
 * <li>[-s number of steps for Newton-Raphson]</li>
 * <li>[-t stopping condition tolerance for Newton-Raphson (recommend negative number), only for
 * CPU]</li>
 * <li>[-x x block dimension, only for GPU]</li>
 * <li>[-y y block dimension, only for GPU]</li>
 * </ul>
 *
 * @author Sherman Lo
 */
public class Empirical_Null_Filter extends EmpiricalNullFilter {

  public Empirical_Null_Filter() {
    this.setProgress(true);
  }

  public static void main(String[] args) throws Exception {
    System.out.println("MIT License - please see LICENSE");
    System.out.println("Copyright (c) 2019-2024 Sherman Lo");
    System.out.println("Please see https://github.com/shermanlo77/modefilter or README.md");
    System.out.println();

    if (args.length > 0) {

      if (args[0].equals("--help")) {
        printManual();
      } else {
        String fileNameLoad = null;
        boolean isCpu = false;

        // extract required arguments
        // 0: run or gui
        // 1: cpu or gpu
        // 2: location of file
        try {
          if (args[1].equals("cpu")) {
            isCpu = true;
          } else if (args[1].equals("gpu")) {
            isCpu = false;
          } else {
            throw new Exception();
          }
          fileNameLoad = args[2];
        } catch (Exception exception) {
          printManual();
          throw new Exception("Incorrect usage");
        }

        if (args[0].equals("gui")) {
          runGui(fileNameLoad, isCpu);
        } else if (args[0].equals("run")) {
          String fileNameSave = args[3];
          run(args, 4, isCpu, fileNameLoad, fileNameSave);
        }
      }
    }
  }

  /**
   * Parse arguments further for run
   *
   * @param args array of strings provided by user
   * @param argsIndex index pointing to args, where to start processing the args
   * @param filter To be modified according to args
   * @param isCpu true if using CPU, else GPU, certain options only available for each
   * @throws Exception When caused by parsing args to integers or floats
   */
  private static void parseArg(String[] args, int argsIndex, EmpiricalNullFilter filter,
      boolean isCpu) throws Exception {
    ModeFilterGpu filterGpu;

    // process all args
    while (argsIndex < args.length) {
      switch (args[argsIndex++]) {
        case "-r": // radius
          float radius;
          radius = Float.parseFloat(args[argsIndex++]);
          filter.setRadius(radius);
          break;
        case "-n": // number of threads
          int numThreads;
          numThreads = Integer.parseInt(args[argsIndex++]);
          filter.setNumThreads(numThreads);
          break;
        case "-i": // number of initial values
          int nInitial;
          nInitial = Integer.parseInt(args[argsIndex++]);
          filter.setNInitial(nInitial);
          break;
        case "-s": // number of steps
          int nStep;
          nStep = Integer.parseInt(args[argsIndex++]);
          filter.setNStep(nStep);
          break;
        case "-t": // log tolerance
          float log10Tolerance;
          log10Tolerance = Float.parseFloat(args[argsIndex++]);
          if (isCpu) {
            filter.setLog10Tolerance(log10Tolerance);
          } else {
            throw new Exception("-t only available for cpu");
          }
          break;
        case "-x": // block dim x for GPU
          int dimX;
          dimX = Integer.parseInt(args[argsIndex++]);
          if (!isCpu) {
            filterGpu = (ModeFilterGpu) filter;
            filterGpu.setBlockDimX(dimX);
          } else {
            throw new Exception("-x only available for gpu");
          }
          break;
        case "-y": // dim y for GPU
          int dimY;
          dimY = Integer.parseInt(args[argsIndex++]);
          if (!isCpu) {
            filterGpu = (ModeFilterGpu) filter;
            filterGpu.setBlockDimY(dimY);
          } else {
            throw new Exception("-y only available for gpu");
          }
          break;
        default: // unknown arg, throw exception
          throw new Exception("Unknown arg " + args[argsIndex - 1]);
      }
    }
  }

  /**
   * Mode filter an image and save it as .png
   *
   * @param args array of strings provided by user
   * @param argsIndex index pointing to args, where to start processing the args
   * @param isCpu true if using CPU, else GPU, certain options only available for each
   * @param fileNameLoad path to where the image to be filtered is located
   * @param fileNameSave path to where to save the filtered imaeg as .png
   * @throws Exception
   */
  private static void run(String[] args, int argsIndex, boolean isCpu, String fileNameLoad,
      String fileNameSave) throws Exception {

    EmpiricalNullFilter filter;
    if (isCpu) {
      filter = new ModeFilter();
    } else {
      filter = new ModeFilterGpu();
    }

    try {
      parseArg(args, argsIndex, filter, isCpu);
    } catch (Exception exception) {
      // if cannot parse args, throw exception and do not proceed further
      throw exception;
    }

    Opener opener = new Opener();
    ImagePlus image = opener.openImage(fileNameLoad);

    filter.setup("", image);
    filter.setOutputImage(0); // do not need other output images

    ImageProcessor imageProcessor = image.getProcessor();

    int nChannel;
    int imageType = image.getType();
    if (imageType == ImagePlus.COLOR_RGB || imageType == ImagePlus.COLOR_256) {
      nChannel = 3;
    } else {
      nChannel = 1;
    }

    long startTime = System.currentTimeMillis();
    for (int iChannel = 0; iChannel < nChannel; iChannel++) {
      FloatProcessor floatProcessor = null;
      floatProcessor = imageProcessor.toFloat(iChannel, floatProcessor);
      filter.run(floatProcessor);
      imageProcessor.setPixels(iChannel, floatProcessor);
    }

    long endTime = System.currentTimeMillis();
    System.out.println("Time: " + (endTime - startTime) + " ms");

    FileSaver fileSaver = new FileSaver(image);
    fileSaver.saveAsPng(fileNameSave);
  }

  /**
   * Mode filter an image using ImageJ GUI and save it as a .png using ImageJ GUI
   *
   * @param fileName path to where the image to be filtered is located
   * @param isCpu true if using CPU, else GPU, certain options only available for each
   * @throws Exception
   */
  private static void runGui(String fileName, boolean isCpu) throws Exception {
    Opener opener = new Opener();
    ImagePlus image = opener.openImage(fileName);
    image.show();

    // instantiate filter
    EmpiricalNullFilter modeFilter;
    if (isCpu) {
      modeFilter = new ModeFilter();
    } else {
      modeFilter = new ModeFilterGpu();
    }

    // do filtering
    new PlugInFilterRunner(modeFilter, "", "");

    FileSaver fileSaver = new FileSaver(image);
    fileSaver.saveAsPng();
  }

  /**
   * Print manual
   */
  public static void printManual() {
    System.out.println("Does mode filtering on an image");
    System.out.println("Usage:");
    System.out.println("  java -jar Empirical_Null_Filter-x.x.x.jar gui ['cpu' or 'gpu'] "
        + "<loc of image to filter>");
    System.out.println("OR");
    System.out.println(
        "  java -jar Empirical_Null_Filter-x.x.x.jar run ['cpu' or 'gpu'] <loc of image to filter> "
            + "<loc to save resulting .png> [options]");
    System.out.println();
    System.out.println("Options:");
    System.out.println("-r    radius of kernel");
    System.out.println("-n    number of threads");
    System.out.println("-i    number of initial points for Newton-Raphson");
    System.out.println("-s    number of steps for Newton-Raphson");
    System.out.println(
        "-t    stopping condition tolerance for Newton-Raphson (recommend negative number), "
            + "only for CPU]");
    System.out.println("-x    x block dimension, only for GPU");
    System.out.println("-y    y block dimension, only for GPU");
  }
}
