# API usage sample for object_detection {#pot_sample_object_detection_README}

This sample demonstrates the use of the [Post-training Optimization Toolkit API](../../compression/api/README.md) for the task of quantizing an object detection model.
The [MobileNetV1 FPN](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/ssd_mobilenet_v1_fpn_coco/ssd_mobilenet_v1_fpn_coco.md) model from TensorFlow* for object detection task is used for this purpose.
A custom `DataLoader` is created to load the [COCO](https://cocodataset.org/) dataset for object detection task 
and the implementation of mAP COCO is used for the model evaluation.

## How to prepare the data

To run this sample, you will need to download the validation part of the [COCO](https://cocodataset.org/). The images should be placed in a separate folder, which will be later referred as `<IMAGES_DIR>` and annotation file `instances_val2017.json` later referred as `<ANNOTATION_FILE>`.  
## How to Run the Sample

In the instructions below, the Post-Training Optimization Tool directory `<INSTALL_DIR>/deployment_tools/tools/post_training_optimization_toolkit` is referred to as `<POT_DIR>`. `<INSTALL_DIR>` is the directory where Intel&reg; Distribution of OpenVINO&trade; toolkit is installed.

1. To get started, follow the [Installation Guide](docs/InstallationGuide.md).
2. Launch the `downloader` tool to download `ssd_mobilenet_v1_fpn_coco` model from the Open Model Zoo repository.
   ```sh
   python3 <POT_DIR>/libs/open_model_zoo/tools/downloader/downloader.py --name ssd_mobilenet_v1_fpn_coco
3. Launch `converter` tool to generate Intermediate Representation (IR) files for the model:
   ```sh
   python <PATH_TO_MODEL_OPTIMIZER>/mo.py --reverse_input_channels --input_shape=[1,640,640,3] --input=image_tensor --output=detection_scores,detection_boxes,num_detections -transformations_config=<PATH_TO_MODEL_OPTIMIZER>/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config=pipeline.config --input_model=frozen_inference_graph.pb --data_type=FP16
   ```
4. Launch the sample script:
   ```sh
   python <POT_DIR>/sample/object_detection/object_detection_sample.py -m <PATH_TO_IR_XML> -d <IMAGES_DIR> --annotation-path <ANNOTATION_FILE>
   ```
   
*  Optional: you can specify .bin file of IR directly using the `-w`, `--weights` options.
