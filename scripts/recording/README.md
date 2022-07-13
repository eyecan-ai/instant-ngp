# Collect poses manually


Launch:

```bash
python scripts/recording/collect_poses.py --neural-twin-file $SCENE/custom.msgpack --transforms-file $SCENE/transform.json  --output-folder $TRAJ_FOLDER --orbit
``` 

Where `$TRAJ_FOLDER` is the folder where the poses will be saved in a Underfolder format (should ends with a data subfolder like 'one/two/three/data')


# Play poses

## Debug preview

To debug recorderd poses launch:

```bash
python scripts/recording/play_poses.py --neural-twin-file $SCENE/custom.msgpack --transforms-file $SCENE/transform.json --replay-dataset $TRAJ_FOLDER
```

To save output instead of live imshow, add option:

```bash
--output-folder $OUTPUT_FOLDER
```

## Batch record multiple scenes

Launch the script [play_poses.sh](play_poses.sh):

```bash
sh scripts/recording/play_poses.sh
```