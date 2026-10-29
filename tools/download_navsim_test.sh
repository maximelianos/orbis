#!/bin/bash
# OpenScene test split for navtest / navhard stage 1: the log metadata and the
# front camera only. Each camera tarball is streamed and only CAM_F0 is kept.
#   tools/download_navsim_test.sh /scratch/global/velikanm/navsim/download
set -e
cd "$1"
HF=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1
if [ ! -d test_navsim_logs ]; then
  wget -q --show-progress $HF/openscene_metadata_test.tgz
  tar -xzf openscene_metadata_test.tgz && rm openscene_metadata_test.tgz
  mv openscene-v1.1/meta_datas test_navsim_logs
fi
mkdir -p test_sensor_blobs
for split in {0..31}; do
  [ -f test_sensor_blobs/.done_$split ] && continue
  echo "camera chunk $split"
  wget -q -O - $HF/openscene_sensor_test_camera/openscene_sensor_test_camera_${split}.tgz \
    | tar -xz -C test_sensor_blobs --strip-components=2 --wildcards --wildcards-match-slash '*/CAM_F0/*'
  touch test_sensor_blobs/.done_$split
done
echo ALL_DONE
