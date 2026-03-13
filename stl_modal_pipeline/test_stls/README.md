# Test STL Drop Folder

Place test STL files in this folder.

You can then run the pipeline by file name:

```bash
python -m stl_modal_pipeline.run_modal_pipeline \
  --stl-name your_file.stl \
  --output-dir /path/to/output
```

Notes:
- `--stl-name` uses this folder by default.
- You can override with `--stl-dir /another/folder`.
- You can still pass a full path with `--stl /full/path/to/file.stl`.
