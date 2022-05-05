# Post-Training Optimization Tool {#pot_README}

## Introduction

Post-training Optimization Tool (POT) is designed to accelerate the inference of deep learning models by applying
special methods without model retraining or fine-tuning, like post-training quantization. Therefore, the tool does not
require a training dataset or a pipeline. To apply post-training algorithms from the POT, you need:
* A full precision model, FP32 or FP16, converted into the OpenVINO&trade; Intermediate Representation (IR) format
* A representative calibration dataset of data samples representing a use case scenario, for example, 300 images

The tool is aimed to fully automate the model transformation process without changing the model structure.
The POT is available only in the Intel&reg; distribution of OpenVINO&trade; toolkit and is not opensourced. For details
about the low-precision flow in OpenVINO&trade;, see the [Low Precision Optimization Guide](docs/LowPrecisionOptimizationGuide.md).

Post-training Optimization Tool includes a standalone command-line tool and a Python* API that provide the following key
features:

* Two post-training 8-bit quantization algorithms: fast [DefaultQuantization](compression/algorithms/quantization/default/README.md) and precise [AccuracyAwareQuantization](compression/algorithms/quantization/accuracy_aware/README.md).
* Global optimization of post-training quantization parameters using the [Tree-Structured Parzen Estimator](compression/optimization/tpe/README.md).
* Symmetric and asymmetric quantization schemes. For details, see the [Quantization](compression/algorithms/quantization/README.md) section.
* Compression for different hardware targets such as CPU and GPU.
* Per-channel quantization for Convolutional and Fully-Connected layers.
* Multiple domains: Computer Vision, Recommendation Systems.
* Ability to implement a custom optimization pipeline via the supported [API](compression/api/README.md).

> **TIP**: You also can work with the Post-training Optimization Tool inside the OpenVINO&trade; [Deep Learning Workbench](@ref workbench_docs_Workbench_DG_Introduction) (DL Workbench).
> [DL Workbench](@ref workbench_docs_Workbench_DG_Introduction) is a platform built upon OpenVINO&trade; and provides a web-based graphical environment that enables you to optimize, fine-tune, analyze, visualize, and compare 
> performance of deep learning models on various Intel&reg; architecture
> configurations. In the DL Workbench, you can use most of OpenVINO&trade; toolkit components.
> <br>
> Proceed to an [easy installation from Docker Hub](@ref workbench_docs_Workbench_DG_Install_from_Docker_Hub) to get started.


For benchmarking results collected for the models optimized with POT tool, see [INT8 vs FP32 Comparison on Select Networks and Platforms](@ref openvino_docs_performance_int8_vs_fp32).

Further documentation presumes that you are familiar with the basic deep learning concepts, such as model inference,
dataset preparation, model optimization, as well as with the OpenVINO&trade; toolkit and its components such 
as  [Model Optimizer](@ref openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide) 
and [Accuracy Checker Tool](@ref omz_tools_accuracy_checker_README).

To get started, follow the [Installation Guide](docs/InstallationGuide.md). The next step is either to use the POT command-line tool described below or to use the [POT API](compression/api/README.md).

## Use Post-Training Optimization Tool Command-Line Interface

Before running the POT, convert your pretrained model into the OpenVINO&trade; IR format with the 
[Model Optimizer](@ref openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide). In addition, it is highly recommended 
to use the [Accuracy Checker Tool](@ref omz_tools_accuracy_checker_README) to make sure that the model can be successfully 
inferred and achieves similar accuracy numbers as the reference model from the original framework. 

To run the command-line Post-training Optimization Tool:
1. Activate the Python environment in the command-line shell where the POT and the Accuracy Checker were installed.
2. Set up the OpenVINO&trade; environment in the command-line shell with the following script:
   ```sh
   source <INSTALL_DIR>/bin/setupvars.sh
   ```
3. Prepare a configuration file for the POT using the examples in the `configs` folder. To simplify this step, use the
Accuracy Checker configuration file for the floating-point model and refer to it when necessary. 
See [Post-Training Optimization Best Practices](docs/BestPractices.md).
4. Launch the command-line tool with the configuration file:
   ```sh
   pot -c <path_to_config_file>
   ```
    For all available usage options, use the `-h`, `--help` arguments or refer to the Command-Line Arguments below.  
5. By default, the results are dumped into the separate output subfolder inside the `results` folder that is created 
in the same directory where the tool is run from. Use the `-e` option to evaluate the accuracy directly from the tool.

See the [How to Run Examples tutorial](configs/examples/README.md) about how to run a particular example of 8-bit
quantization with the POT.

### Command-Line Arguments

The following command-line options are available to run the tool: 

| Argument                                          | Description                                             |
| ------------------------------------------------- | ------------------------------------------------------- |
| `-h`, `--help`                                    | Optional. Show help message and exit. |
| `-c CONFIG`, `--config CONFIG`                    | Path to a config file with task- or model-specific parameters.         |
| `-e`, `--evaluate`                                | Optional. Evaluate model on the whole dataset after optimization.  |
| `--output-dir OUTPUT_DIR`                         | Optional. A directory where results are saved. Default: `./results`. |
| `-sm`, `--save-model`                             | Optional. Save the original full-precision model. |
| `-d`, `--direct-dump`                             | Optional. Save results directly to output directory without additional subfolders. |
| `--log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}` | Optional. Log level to print. Default: INFO. |
| `--progress-bar`                                  | Optional. Disable CL logging and enable progress bar. |
| `--stream-output`                                 | Optional. Switch model quantization progress display to a multiline mode. Use with third-party components. |
| `--keep-uncompressed-weights`                     | Optional. Keep Convolution, Deconvolution and FullyConnected weights uncompressed. Use with third-party components.|

## See Also

* [Low Precision Optimization Guide](docs/LowPrecisionOptimizationGuide.md)
* [Post-Training Optimization Best Practices](docs/BestPractices.md)
* [POT Frequently Asked Questions](docs/FrequentlyAskedQuestions.md) 
* [INT8 Quantization by Using Web-Based Interface of the DL Workbench](https://docs.openvinotoolkit.org/latest/workbench_docs_Workbench_DG_Int_8_Quantization.html)
