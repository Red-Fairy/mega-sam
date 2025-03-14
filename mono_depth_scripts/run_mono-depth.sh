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

data_dir=$1
seq_name=$2

echo "Running trajectory estimation on $data_dir/$seq_name"

# Run DepthAnything
python Depth-Anything/run_videos.py --encoder vitl \
  --load-from Depth-Anything/checkpoints/depth_anything_vitl14.pth \
  --img-path "$data_dir/$seq_name" \
  --outdir "Depth-Anything/video_visualization/$seq_name"

# Run UniDepth
export PYTHONPATH="${PYTHONPATH}:$(pwd)/UniDepth"

python UniDepth/scripts/demo_mega-sam.py \
  --scene-name "$seq_name" \
  --img-path "$data_dir/$seq_name" \
  --outdir UniDepth/outputs

