import datetime
import logging as log
import os
import sys
from argparse import ArgumentParser, SUPPRESS
from openvino import inference_engine as ie
from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np
#import ngraph as ng
np.set_printoptions(suppress=True, precision=5, formatter={'int':hex})
#np.set_printoptions(threshold=np.inf, suppress=True, precision=5, formatter={'int':hex}
def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-ref_m", "--ref_model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)

    args.add_argument("-i", "--input", help="Required. Path to an image/video file. (Specify 'cam' to work with "
                                            "camera)", required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with "
                           "the kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
                           " acceptable. The sample will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="VPUX", type=str)
    args.add_argument("-ref_d", "--ref_device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
                           " acceptable. The sample will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    return parser

THRESHOLDS = [2, 1, 2, 1]
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
NORMAL = '\033[0m'
BOLD = '\033[1m'
PURPLE = '\033[95m'

def metrics(a, b):
    ref = np.max(np.abs(b))
    total_values = int(len(b.flatten()))
    diff = np.abs(a - b)
    max_error = np.max(np.abs(a - b))
    mean_error = np.mean(np.abs(a - b))
    l2_error = np.sqrt(np.sum(np.square(a - b)) / total_values)
    if ref == 0:
        max_error = 0 if max_error == 0 else np.inf
        mean_error = 0 if mean_error == 0 else np.inf
        l2_error = 0 if l2_error == 0 else np.inf
    else:
        max_error = max_error / ref * 100
        mean_error = mean_error / ref * 100
        l2_error = l2_error / ref * 100
    percentage_wrong = len(
        np.extract(
            diff > 0.05 * ref,
            diff)) / total_values * 100
    sum_diff = np.sum(np.abs(a - b))
    return [max_error, mean_error, percentage_wrong, l2_error, sum_diff]

def metrix_comparison(results):
    status = []
    for i in range(4):
        if results[i] > THRESHOLDS[i] or np.isnan(results[0]).any():
            status.append(FAIL + "Fail" + NORMAL)
        else:
            status.append("Pass")
    print("------------------------------------------------------------")
    print(" Obtained values ")
    print("------------------------------------------------------------")
    print(
        " Obtained Min Pixel Accuracy: {}% (max allowed={}%), {}".format(
            results[0],
            THRESHOLDS[0],
            status[0]))
    print(
        " Obtained Average Pixel Accuracy: {}% (max allowed={}%), {}".format(
            results[1],
            THRESHOLDS[1],
            status[1]))
    print(
        " Obtained Percentage of wrong values: {}% (max allowed={}%), {}".format(
            results[2],
            THRESHOLDS[2],
            status[2]))
    print(
        " Obtained Pixel-wise L2 error: {}% (max allowed={}%), {}".format(
            results[3],
            THRESHOLDS[3],
            status[3]))
    print(" Obtained Global Sum Difference: {}".format(results[4]))
    print("------------------------------------------------------------")
    return status[0] == 'Pass' and status[1] == 'Pass' and status[2] == 'Pass' and status[3] == 'Pass'

def metrix_compare(act, ref):
    metrix_comparison(metrics(act, ref))


def main():
    args = build_argparser().parse_args()

    ie = IECore()

    ref_net = ie.read_network(args.ref_model, os.path.splitext(args.ref_model)[0] + ".bin")
    #assert len(ref_net.inputs) == 1, "Sample supports only one input"
    assert len(ref_net.input_info) == 1, "Sample supports only one input"

    log.info("Preparing inputs")
    input_name = next(iter(ref_net.input_info))
    #input_name = next(iter(ref_net.inputs))

    input_shape = ref_net.inputs[input_name].shape
    img_org = cv2.imread(args.input)
    img = cv2.resize(img_org,(input_shape[3], input_shape[2]))
    data_org = np.array(img)
    data = data_org
    data = data.transpose(2,0,1)
    #data = np.fromfile("./inspur/input-hwc.bin", dtype=np.uint8).reshape(3,416,416)
    input_data = {input_name: data}
    data.tofile("input.bin")
    log.info("Run CPU")
    ref_exec_net = ie.load_network(ref_net, args.ref_device)
    ref_result = ref_exec_net.infer(input_data)

    log.info("Run VPU")
    #data = np.transpose(data, (1, 2, 0))
    #input_data = {input_name: data}
    exec_net = ie.import_network(args.model, num_requests=2, device_name=args.device)
    print("VPU input layout: ", exec_net.inputs[input_name].layout, "  precision ", exec_net.inputs[input_name].precision, " shape ", exec_net.inputs[input_name].shape)
    for out_name in exec_net.outputs:
        print("VPU output layut: ", exec_net.outputs[out_name].layout, " precision ", exec_net.outputs[out_name].precision)

    result = exec_net.infer(input_data)
    for name in ref_result:
        ref = ref_result[name]
        act = result[name]
        #act = np.transpose(act, (0, 2, 1)).reshape(1,3549,85)
        print(act.shape)
        print("\n")
        print("Comparing for ", name, " : ")
        metrix_compare(act, ref)
        print(name, " Ref:")
        print(ref)
        print("\n")
        print(name, " Act:")
        print(act)
    
if __name__ == '__main__':
    sys.exit(main() or 0)

