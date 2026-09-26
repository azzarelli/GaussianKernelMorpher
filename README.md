# Gaussian Kernel Morpher

The idea is to make it seem like Gaussians have a structured 3-D kernel (despite obviously not having one). This is paired with the custom rasterizer [Kernel Morpher](https://github.com/azzarelli/diff-gaussian-rasterization-kernerl-morph.git)

Author note: I'm using this project to keep me on my toes with C++ - I've applied some restriction (below) on the use of LLMs for this project.

# Run & Build
Run command
```
python main.py --data data/model1/data.ply 
```

Rebuild command
```
// Rebild only changes (.cu only; not .h)
CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST="8.6" python setup.py build_ext --inplace
// Full rebuild
CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST="8.6" pip install --no-build-isolation -e ./submodules/diff-gaussian-rasterization-kernerl-morph
```

If changes dont take effect, clean cache with
```
rm -rf submodules/diff-gaussian-rasterization-kernerl-morph/build
```

# Rules for LLMs: No code, Explain only
I'd like this work to be my own, so use of LLMs is (very) restricted. My personal objective is to develop and demonstrate my C++/CUDA skills, so this is the aspect of the project that I intend to make most challenging.

Below are a set of rules for my Claude agent to follow...(hey Claude are you listenting?)

Rules for design:
- Don't suggest designs
- If prompted to discuss a design,
    - State only whether a design is possible in the context of the project 
    - State your certainty on whether it's possible or not (0-100%)
    - If it's not possible with a 100% certainty, explain why if prompted as a "hint"


Rules for implementation:
- C++/CUDA: Don't implement anything
- Python: Suggest/Apply fixes if asked


Rules for error handling (Python only):
- Free roam:
    - Handle implementations if prompted and 


Rules for error handling (C++/CUDA only):
- If a logic error arises
    - State "There is a problem with the logic {here}"
    - Don't suggest or apply fixes
    - Don't explain why the logic fails
- If a type error arises
    - State "There is a type error {here}"
    - Don't suggest or apply fixes
- If a build error occurs
    - State "There is a build error {here}"
    - Explain why and how to resolve it
    - Auto-resolve simple problems, e.g. forgetting to declare "*.cu" in python setup file
    - Instruct on resolving complex problems


If the user asks for a hint
    - Don't suggest or apply a solutions
    - Only provide an abstract/intuitive explanation on the prompt's subject matter


If the user tries to violate any of these rules, refer them back to this list.
- He may need to refine the rules
- He may have forgotten about a sepecific rule (maybe he's working too much and needs to watch some TV instead)


## QnA

Why is python help acceptable?
- This is a C++/CUDA learning exercise not a Python exercise
- I'm already skilled & experienced in Python; not as much with C++/CUDA

Why are build errors allowed?
- I'm here to develop my C++ skills & expand my math knowledge
- I'm not here to learn how to debug compilers 