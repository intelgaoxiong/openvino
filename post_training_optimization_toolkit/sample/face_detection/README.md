# API usage sample for face detection task {#pot_sample_face_detection_README}

This sample demonstrates the use of the [Post-training Optimization Tool API](../../compression/api/README.md) for the task of quantizing a face detection model.
The [MTCNN](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/mtcnn/mtcnn.md) model from Caffe* is used for this purpose.
A custom `DataLoader` is created to load [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) dataset for a face detection task 
and the implementation of Recall metric is used for the model evaluation. In addition, this sample demonstrates how one can implement 
an engine to infer a cascaded (composite) model that is represented by multiple submodels in an OpenVino&trade; Intermediate Representation (IR)
and has a complex staged inference pipeline.

## How to prepare the data

To run this sample, you need to download the validation part of the Wider Face dataset http://shuoyang1213.me/WIDERFACE/.
Images with faces divided into categories are placed in the `WIDER_val/images` folder.
Annotations in .txt format containing the coordinates of the face bounding boxes of the validation part of the dataset 
can be downloaded separately and are located in the `wider_face_split/wider_face_val_bbx_gt.txt` file.

## How to Run the Sample

In the instructions below, the Post-Training Optimization Tool directory `<INSTALL_DIR>/deployment_tools/tools/post_training_optimization_toolkit` is referred to as `<POT_DIR>`. 
`<INSTALL_DIR>` is the directory where Intel&reg; Distribution of OpenVINO&trade; toolkit is installed.

1. To get started, follow the [Installation Guide](docs/InstallationGuide.md).
2. Launch the `downloader` tool to download `mtcnn` model from the Open Model Zoo repository.
   ```sh
   python3 <POT_DIR>/libs/open_model_zoo/tools/downloader/downloader.py --name mtcnn*
   ```
3. Launch `converter` tool to generate Intermediate Representation (IR) files for the model:
   ```sh
   python3 <POT_DIR>/libs/open_model_zoo/tools/downloader/converter.py --name mtcnn* --precisions FP32
   ```
4. Launch the sample script:
   ```sh
   python3 <POT_DIR>/sample/face_detection/face_detection_sample.py -pm <PATH_TO_IR_XML_OF_PNET_MODEL> 
   -rm <PATH_TO_IR_XML_OF_RNET_MODEL> -om <PATH_TO_IR_XML_OF_ONET_MODEL> -d <WIDER_val/images> -a <wider_face_split/wider_face_val_bbx_gt.txt>
   ```
   Optional: you can specify .bin files of corresponding IRs directly using the `-pw/--pnet-weights`, `-rw/--rnet-weights` and `-ow/--onet-weights` options.
