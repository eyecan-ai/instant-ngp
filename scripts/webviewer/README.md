# Launch Orbiter server

Launch the following command to spawn a rendering server (from instanto root folder due to lazyness):

```
python scripts/webviewer/rendering_server.py --neural-twin-file ${MSGPACK_FILE} --transforms-file ${TRANSFORM_JSON_FILE}
```


# Launch web client

Open the `scripts/webviewer/index.html` file in browser. The rendered image should be displayed in the web page.
Use `WASDQE` keys to navigate the nerf.