#!/bin/bash

# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#The deploy script is for release/2021/4 and before of openvino

usage() {
    echo "Deploy inference engine release"
    echo
    echo "Options:"
    echo "  -h                       Print the help message"
    echo "  -d DEST_RELEASE_DIR      Specify the release destination directory"
    echo "  -s SOURCE_RELEASE_DIR    Specify the release source directory"
    echo "  -m MODE                  Choose release or replace libs, replace default"
    echo "  -i INSTALL_DIR           Specify the build install diretory if the mode is release"
    echo
    exit 1
}

src_dir="${PWD}"
dst_dir="${HOME}/release_prc"
install_dir="${PWD}/install"
mode="replace"

# parse command line options
while [[ $# -gt 0 ]]
do
case "$1" in
    -d | --dst_dir)
    dst_dir="$2"
    shift
    ;;
    -s | --src_dir)
    src_dir="$2"
    shift
    ;;
    -m | --mode)
    mode="$2"
    shift
    ;;
    -i | --install)
    install_dir="$2"
    shift
    ;;
    -h | --help)
    usage
    ;;
    *)
    echo "Unrecognized option specified $1"
    usage
    ;;
esac
shift
done

libs=(
     "libformat_reader.so"
     "libfrontend_mcm.so"
     "libhddl2_backend.so"
     "libHeteroPlugin.so"
     "libinference_engine_c_api.so"
     "libinference_engine_ir_reader.so"
     "libinference_engine_ir_v7_reader.so"
     "libinference_engine_legacy.so"
     "libinference_engine_lp_transformations.so"
     "libinference_engine_preproc.so"
     "libinference_engine.so"
     "libinference_engine_transformations.so"
     "libmcmCompiler.so"
     "libmcmCompiler.so.1.5"
     "libmcmCompiler.so.1.5.3"
     "libMKLDNNPlugin.so"
     "libvpux_compiler.so"
     "libVPUXPlugin.so"
     "libzero_backend.so"
     "mcm_config"
     "plugins.xml"
)

headers=(
    #update pre-proc
    "inference-engine/include/ie_blob.h"
    "inference-engine/include/ie_compound_blob.h"
    "inference-engine/include/ie_layouts.h"
    #import buffer
    "inference-engine/include/ie_core.hpp"
    #cache output blob
    "inference-engine/include/details/ie_pre_allocator.hpp"
    "inference-engine/include/ie_allocator.hpp"
    "inference-engine/include/ie_blob.h"
    "inference-engine/include/ie_compound_blob.h"
)

DEPLOY_TOOLS_PATH=${dst_dir}/deployment_tools
DEPLOY_PYTHON_PATH=${dst_dir}/python
SRC_PYTHON_PATH=${src_dir}/bin/intel64/Release/lib/python_api
SRC_LIB_PATH=${src_dir}/bin/intel64/Release/lib
DST_LIB_PATH=${DEPLOY_TOOLS_PATH}/inference_engine/lib/intel64
SRC_HEADER_PATH=${src_dir}
DST_HEADER_PATH=${DEPLOY_TOOLS_PATH}/inference_engine/include

replace() {
    # copy from src_dir to dst_dir
    if [[ (-d ${dst_dir}) && (-d ${src_dir}) ]];then
	echo "...Replace libs..."
	for lib in ${libs[*]}
	do
	    #echo "${lib}"
	    SRC_LIB_FILE=${SRC_LIB_PATH}/${lib}
	    DST_LIB_FILE=${DST_LIB_PATH}/${lib}
	    cp -a ${SRC_LIB_FILE} ${DST_LIB_FILE}
	done
	echo "...Replace headers..."
	for header in ${headers[*]}
	do
            #echo "${header}"
	    SRC_HEADER_FILE=${SRC_HEADER_PATH}/${header}
	    DST_HEADER_FILE=${DST_HEADER_PATH}/$(basename $header)
	    cp -a ${SRC_HEADER_FILE} ${DST_HEADER_FILE}
	done
	# copy POT & OMZ
	echo "...Replace POT & OMZ..."
	mkdir -p ${DEPLOY_TOOLS_PATH}/tools
	cp -r ${src_dir}/post_training_optimization_toolkit ${DEPLOY_TOOLS_PATH}/tools
        cp -r ${src_dir}/open_model_zoo ${DEPLOY_TOOLS_PATH}
	echo "...Replace python..."
	cp -a  ${SRC_PYTHON_PATH}/* ${DEPLOY_PYTHON_PATH}
    else
        echo "Please confirm source and destination directory needed existing!"
        exit 1
    fi
}

release() {
    echo "Generate a release package from sractch..."
    echo "...Release deployment_tools..."
    if [[ ! -d ${install_dir} ]];then
        echo "Please specify the build install!"
	exit 1
    fi

    mkdir -p ${dst_dir}
    mkdir -p ${DEPLOY_TOOLS_PATH}
    mkdir -p ${DEPLOY_PYTHON_PATH}

    cp -r ${install_dir}/bin ${dst_dir}
    cp -a ${src_dir}/inference-engine/temp/opencv_4.5.2_ubuntu20/opencv ${dst_dir}
    SRC_DEPLOY_TOOLS_PATH=${install_dir}/deployment_tools
    cp -a ${SRC_DEPLOY_TOOLS_PATH} ${dst_dir}
    replace

    rm -rf ${DEPLOY_TOOLS_PATH}/demo
    rm -rf ${DEPLOY_TOOLS_PATH}/inference_engine/samples

    DST_DEMO_BIN_PATH=${DEPLOY_TOOLS_PATH}/inference_engine/bin
    mkdir -p ${DST_DEMO_BIN_PATH}
    cp -a ${SRC_LIB_PATH}/../benchmark_app ${DST_DEMO_BIN_PATH}
    cp -a ${SRC_LIB_PATH}/../compile_tool ${DST_DEMO_BIN_PATH}
}

# choose and run different mode
if [[ ${mode} == "replace" ]];then
    echo "Run replace mode..."
    replace
    echo "Replace complete!"
else
    echo "Run release mode..."
    release
    echo "Release Complete!"
fi

