ffmpeg \
    -i /home/daniele/Desktop/experiments/2022-07-07.NerfExperiments/output/combined_videos/lion.mp4 \
    -i /home/daniele/Desktop/experiments/2022-07-07.NerfExperiments/output/combined_videos/lion_carousel.mp4 \
    -i /home/daniele/Desktop/experiments/2022-07-07.NerfExperiments/output/combined_videos/lion_gyro.mp4 \
    -i /home/daniele/Desktop/experiments/2022-07-07.NerfExperiments/output/combined_videos/mars.mp4 \
    -i /home/daniele/Desktop/experiments/2022-07-07.NerfExperiments/output/combined_videos/mars_carousel.mp4 \
    -i /home/daniele/Desktop/experiments/2022-07-07.NerfExperiments/output/combined_videos/mars_gyro.mp4 \
    -i /home/daniele/Desktop/experiments/2022-07-07.NerfExperiments/output/combined_videos/twix.mp4 \
    -i /home/daniele/Desktop/experiments/2022-07-07.NerfExperiments/output/combined_videos/twix_carousel.mp4 \
    -i /home/daniele/Desktop/experiments/2022-07-07.NerfExperiments/output/combined_videos/twix_gyro.mp4 \
  -filter_complex " \
      [0:v] setpts=PTS-STARTPTS, scale=qvga [a0]; \
      [1:v] setpts=PTS-STARTPTS, scale=qvga [a1]; \
      [2:v] setpts=PTS-STARTPTS, scale=qvga [a2]; \
      [3:v] setpts=PTS-STARTPTS, scale=qvga [a3]; \
      [4:v] setpts=PTS-STARTPTS, scale=qvga [a4]; \
      [5:v] setpts=PTS-STARTPTS, scale=qvga [a5]; \
      [6:v] setpts=PTS-STARTPTS, scale=qvga [a6]; \
      [7:v] setpts=PTS-STARTPTS, scale=qvga [a7]; \
      [8:v] setpts=PTS-STARTPTS, scale=qvga [a8]; \
      [a0][a1][a2][a3][a4][a5][a6][a7][a8]xstack=inputs=9:layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0|0_h0+h1|w0_h0+h1|w0+w1_h0+h1[out] \
      " \
    -map "[out]" \
    -c:v libx264  -f matroska /tmp/matrioska.mp4
    # -c:v libx264 -f matroska -  | ffplay -autoexit  -left 30 -top 30  -