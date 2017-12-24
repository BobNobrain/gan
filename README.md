# Simple GAN

> GAN via Keras and TensorFlow

## How to run

You need to have `conda` environment named `env` with
`keras`, `tensorflow`, `matplotlib`, `scipy`, `numpy`
and other packages installed. All scripts are given
(or exist in repo) assuming this environment name.

### Training

```bash
# Starts training and logs output into log.txt
bash run
```

Modify parameters inside `__init__.py` to achieve needed results.
Datasets can be switched there too.
Net snapshots are saved to `./weights/` subdirectory, also specified in that file.

### Loading Pretrained and Generating

```bash
. activate env
python run_loaded.py
```

This will load pretrained model and generate images of each class into specified files.
Modify parameters inside to control the output.
