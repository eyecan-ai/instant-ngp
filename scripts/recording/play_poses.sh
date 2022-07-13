
# Object name used in folders
OBJECT_NAME=lion

# Scene base path
SCENE=/home/daniele/Desktop/experiments/2022-07-07.NerfExperiments/datasets/${OBJECT_NAME}_multilight/nerf_masked_

# Output folder for raw video frames
OUTPUT_FOLDER=/home/daniele/Desktop/experiments/2022-07-07.NerfExperiments/output/output_videos/${OBJECT_NAME}

# Trajectory underfolder
TRAJ_FOLDER=/tmp/traj0

for SIDE in "0000" "0001" "0010" "0100" "1000" "1111";
do
    python scripts/recording/play_poses.py \
        --neural-twin-file ${SCENE}${SIDE}/custom.msgpack \
        --transforms-file ${SCENE}${SIDE}/transform.json \
        --replay-dataset ${TRAJ_FOLDER} \
        --output-folder ${OUTPUT_FOLDER}/${SIDE}
done  