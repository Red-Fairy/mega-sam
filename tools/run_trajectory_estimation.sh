#!/bin/bash
# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

CKPT_PATH=checkpoints/megasam_final.pth

data_dir=$1
seq_name=$2

echo "Running trajectory estimation on $data_dir/$seq_name"

python camera_tracking_scripts/test_demo.py \
    --datapath=$data_dir/$seq_name \
    --weights=$CKPT_PATH \
    --scene_name $seq_name \
    --mono_depth_path $(pwd)/Depth-Anything/video_visualization \
    --metric_depth_path $(pwd)/UniDepth/outputs \
    --disable_vis 
