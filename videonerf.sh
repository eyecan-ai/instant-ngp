VIDEO_PATH=/home/daniele/Downloads/trial/v3.mp4
SCENE_PATH=${VIDEO_PATH}_frames
IMAGES_PATH=${SCENE_PATH}/images
FPS=5

mkdir -p $IMAGES_PATH
ffmpeg -i "${VIDEO_PATH}" -vf fps=5.0 "${IMAGES_PATH}/%5d_image.png"
python scripts/colmap2nerf.py --images $IMAGES_PATH --run_colmap --out ${IMAGES_PATH}/../transforms.json