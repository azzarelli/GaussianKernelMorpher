# Gaussian Kernel Morpher

The idea is to make it seem like Gaussians have a structured 3-D kernel (despite obviously not having one). This is paired with the custom rasterizer [Kernel Morpher](https://github.com/azzarelli/diff-gaussian-rasterization-kernerl-morph.git)

Run command
```
python main.py --data data/model1/data.ply 
```

Rebuild command
```
CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST="8.6" pip install --no-build-isolation -e ./submodules/diff-gaussian-rasterization-kernerl-morph
```

If changes dont take effect, clean cache with
```
rm -rf submodules/diff-gaussian-rasterization-kernerl-morph/build
```

