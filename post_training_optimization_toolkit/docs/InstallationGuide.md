# Post-Training Optimization Tool Installation Guide {#pot_InstallationGuide}

## Prerequisites

* Python* 3.5 or higher
* [OpenVINO&trade;](@ref index)

The minimum and the recommended requirements to run the Post-training Optimization Tool are the same as in [OpenVINO&trade;](@ref index).  

## Install and Set Up Post-Training Optimization Tool

In the instructions below, `<INSTALL_DIR>` is the directory where the Intel&reg; distribution of OpenVINO&trade; toolkit
is installed. Post-training Optimization Tool (POT) is distributed as a part of the OpenVINO&trade; release package, and to use POT as a command-line tool,
you need to install OpenVINO&trade; as well as POT dependencies, namely [Model Optimizer](@ref openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide) 
and [Accuracy Checker](@ref omz_tools_accuracy_checker_README). POT source files are available from the 
`<INSTALL_DIR>/deployment_tools/tools/post_training_optimization_toolkit` directory after the OpenVINO&trade;
installation. It is recommended to create a separate [Python* environment](https://docs.python.org/3/tutorial/venv.html) before installing the OpenVINO&trade; and its components. To set up the POT in your environment, follow the steps below.

### Set up the Model Optimizer and Accuracy Checker components

- To set up the [Model Optimizer](@ref openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide):
   1. Go to `<INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites`.
   2. Run the script to configure the Model Optimizer:
   ```sh
   sudo ./install_prerequisites.sh
   ```
   3. To verify that the Model Optimizer is installed, run `<INSTALL_DIR>/deployment_tools/model_optimizer/mo.py -h`.
- To set up the [Accuracy Checker](@ref omz_tools_accuracy_checker_README):
   1. Go to `<INSTALL_DIR>/deployment_tools/open_model_zoo/tools/accuracy_checker`.
   2. Run the `setup.py` script: 
   ```sh
   python3 setup.py install
   ```
   3. Now the Accuracy Checker is available in the command line by the `accuracy_check` alias. To verify it, run `accuracy_check -h`.

### Set up the POT

1. Go to `<INSTALL_DIR>/deployment_tools/tools/post_training_optimization_toolkit`.
2. Run the `setup.py` script:  
   ```sh
   python3 setup.py install
   ```
   
   In order to enable advanced algorithms such as the Tree-Structured Parzen Estimator (TPE) based optimization, add the following flag to the installation command:
   ```sh
   python3 setup.py install --install-extras
   ```
3. Now the POT is available in the command line by the `pot` alias. To verify it, run `pot -h`. 