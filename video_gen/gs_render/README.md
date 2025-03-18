# 2D Condtion Render

## Installlation

```bash
cd ../diff-gaussian-rasterization
pip install ./
```

## Data Processing

```bash
# run our quick example
python render_eval_example.py

# render GT eval data
python render_eval_condition_gt.py

# render generated eval data
python render_eval_condition_gen.py
```

## Discription

This code is used to render the 2D condition images from the 3D data. The 3D data is stored in the form of a dataset, occupancy data, and layout map. The 3D data is rendered into 2D condition images using the differential gaussian rasterization method. The rendered 2D condition images are saved in the render_path.

#### --dataset_path

Path to the dataset.

#### --occ_path

Path to the occpancy data.

#### --layout_path

Path to the layout map.

#### --render_path

Path to save the rendered 2D condition images.

#### --vis

Visualize the rendered 2D condition images.

## Acknowledgement

Many thanks to these excellent projects:

- [SurroundOcc](https://github.com/weiyithu/SurroundOcc?tab=readme-ov-file)
- [NKSR](https://github.com/nv-tlabs/NKSR?tab=readme-ov-file).
