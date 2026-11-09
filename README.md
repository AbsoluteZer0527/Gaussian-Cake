# Gaussian Cake


## Update 11/1/26
### Model Morphing: Shared Topology & Gaussian Splatting
Two new OT morphing mode were building on top of the Original Gaussian Cake.

https://github.com/user-attachments/assets/783fe288-cc6e-4bae-aed0-1fa1abfe2696


**Topology Mode**: We first check whether the two models have the sharing topology first, then we uses OT to determine the path for the two mesh's triangle topology and morph into another shape. It uses mesh with more faces as the template, and fits that topology to the other model and show the morph animation.If they don't share the same topology, such as mismatch or different number of genus, it uses the **piecewise** path (below) instead; the SDF path is still there as its own method.

**Piecewise (across a change of genus, holes or parts):** the SDF morph decides where the shape goes, and its in-between solids show *when* the topology changes (parts, genus and cavities are read off the SDF grid with the Euler characteristic, and each change is found by bisection). Between two changes the topology is fixed, so a real mesh can follow the surface: A's mesh up to the first change, B's mesh after the last, and an SDF-surface mesh for any segment in between. The morph switches meshes at each change, so a hole opens at one visible moment and both ends are the exact original meshes. Try `Guassian.obj` → `torus.obj` (genus 0 → 1).

SDF: in-between shapes are aligned by two lerp: M_t = (1−t)·I + t·T, making each model a signed distance field and match the surfaces so it creates a smooth morphing effect, but still slightly off the original shapes visually.

*Part-by-part alignment (Align Parts K, default 8):* one affine Gaussian map can only stretch the whole shape, so parts drift (a teapot's handle melts away early). With K > 1, T is the Gaussian-mixture OT map: K Gaussians per shape, matched part to part by Sinkhorn. Where neighbouring parts go different ways that map can fold space, so the morph uses `T = affine + λ(u)·(mixture − affine)` with λ as large as possible while every M_t stays one-to-one. M_t⁻¹ then has no closed form and is found by Newton's method on the GPU. On teapot → bunny, A's surface lands 0.089 from B's instead of 0.116, and the handle survives well into the morph. K = 1 is the plain affine morph.


https://github.com/user-attachments/assets/e1673f76-3a3e-4a30-afbe-c6a0df6a19c8

**Splats Mode**: Each Particles is getting replaced by 3D Gaussian blobs, it stores a center, covariance, color and opacity. Then we applies OT between the mixtures of the two models: we first estimate the moving cost for each blob(Bures cost), and matches them after (Sinkhorn), where it shifts and streches each of those blobs to the other location. At the end, we simulate morphing animation and show the cooking.

Currently supports .obj, .glb, .ply files.

**Real 3DGS captures (.ply):** trained 3D Gaussian Splatting captures load with their view-dependent colour (spherical harmonics up to degree 3), so highlights change as you orbit, also mid-morph. *Colour Detail (SH)* under Settings goes from flat colour (0) to full shading (3). Captures also work in Particles mode. No capture at hand? `python3 tools/make_capture.py meshes/bunny.obj meshes/bunny_capture.ply` makes a synthetic one from a mesh (set *Up Axis* to *As is* for it).

References: 
Chen, Georgiou, Tannenbaum 2019, Optimal Transport for Gaussian Mixture Models. arXiv:1710.07876

Dowson & Landau 1982, The Fréchet distance between multivariate normal distributions doi:10.1016/0047-259X(82)90077-X 

Peyré & Cuturi 2019, Computational Optimal Transport. arXiv:1803.00567

<img width="5600" height="1200" alt="image" src="https://github.com/user-attachments/assets/5b862766-9de2-43d1-9f02-86fc5f8303cc" />
<img width="2400" height="1800" alt="image" src="https://github.com/user-attachments/assets/b80fda21-aa75-47da-83df-9cea7d13711a" />


## Inspiration
This project is inspired by the mathematical problem of **Optimal Transport (OT)**, which seeks a mapping between two distributions that minimizes a specified transport cost. 

Think about this question: "If I have a pile of stuff in one shape, and I want to rearrange it into another shape, what is the cheapest way to move it?"

![Alt text](https://www.microsoft.com/en-us/research/wp-content/uploads/2020/09/OTDD_Shovel-Figure.png)

And if you have ever heard of this website: https://obamify.com/, it shares a similar concept, but instead, this program is 3D.

## What it does
This is an experimental graphic project that mainly serves the purpose of applying the math concept of Optimal Transport into an actual computer animation pipeline.

A low-level C++ OpenGL program that takes in **any .obj imported models** : A and B. Once you click "Bake It!", the program shows either (1) a **smooth animation of the model morphing from A to B**, or (2) transport animation of a **Gaussian distribution sampling from A to B**(any shape to any shape: the default blob, a mesh, or a .ply capture; pick **Greedy**, **Gaussian** or **Gaussian Mixture** transport under Advanced). This is a particle system animation project that showcases the concept of optimal transport in two ways: GreedyDiscrete OT for the mesh and Gaussian OT for the distribution.

The **Greedy discrete optimal transport** is a simple approximation for the surface points that allows the model to go from A to B (mesh-mesh transport).

![Gaussian Cake 2026-04-05 00-06-42](https://github.com/user-attachments/assets/14364820-fb94-488f-9677-ef261d2ac3e8)

The **Gaussian optimal transport** samples the mean and variance from the model, which is that particle-shaped blob that moves into another set of particle-shaped blobs.(sample-sample)

![Gaussian Cake 2026-04-05 00-52-48](https://github.com/user-attachments/assets/649e1400-d890-4f04-bd54-fbb08966795c)

_Since the main concept of this connects with the Gaussian distribution, the morphing process is similar to "baking", it callbacks to the idea of a Gaussian cake...Yum_

## How I built it

I only use raw code C++ and libraries(basic OpenGL, imgui UI, linear algebra libraries) for this project. I first initialized a shader, a window, and a basic mesh import. Then I worked on the particle system, which allows later stages of implementation for the Gaussian sampling and applying optimal transport. Finally, I expand it to greedy optimal transport for the actual animation of model morphing.

## More about the Math: 

The Gaussian optimal transport samples the mean and variance from the model, which yields better runtime and exact geometry morphing with cheap computation power: It's only linear algebra!

$T(x)=m2 + A(x− m1​)$

where m1 and m2 are the Gaussian means, and A encodes the covariance-based linear transformation.

However,  the Greedy OT offers a simpler approach, looks better in animation, but it loses accuracy:

$\sum_{i}\| x_i - y_{\sigma(i)}\|^2_{\text{Greedy}}$

This mapping can then be directly used to generate smooth mesh-to-mesh morphing animations.
And therefore, this transformation could be simulated with computer animation, which could be expanded to mesh-to-mesh morphing animations.

<img width="3165" height="2476" alt="graphic" src="https://github.com/user-attachments/assets/1b8b7b01-583d-4b63-8a21-b4fcb72a816c" />


More insight about OT in general: https://alexhwilliams.info/itsneuronalblog/2020/10/09/optimal-transport/

## Challenges I ran into

Debugging in C++ is painful. I had no clue why my program was not building at all-- 3hrs of debug just to realize my CMakeList.txt was poorly written.

## Accomplishments that I'm proud of
It actually worked.

## What I learned
- Different methods of Optimal Transport (and what even is optimal transport)
- A more solid understanding of Barycentric coordinates: how the particles are from a mesh through a triangle mesh.
- Constructing the 3D Gaussian formula using glm and eigen. (C++ libraries)
- The animation pipeline, using the OT method, is applied to the particle system.

## What's next for Gaussian Cake

This is an experimental project that expands on the concept of a simple particle system: It provides insight into 3D model morphing, and I can definitely incorporate the concept into larger game projects or animation programs. 

_For this project, I really want to show how theoretical concepts can actually be applied in fun settings, and how practical applications can help visualize and understand these concepts._

## Animation window (no rigging needed)

In Topology and Splats modes, the **Animation** window on the right lets you direct a morph like a shot. Hover over anything to see what it does.

- **Transport + timeline:** play/pause (Space), jump to the start or end (Home/End), and loop. The track works like a dope sheet:
  - Each shape is a ◆ key, and each step between two shapes is a clip coloured by its style.
  - Click or drag the ruler to scrub, and click a clip to edit that step.
  - Drag an inner ◆ to retime, and use **+** to add a shape (Topology).
  - Right-click a ◆ to change its shape or delete it, or right-click a clip to pick a style.
- **Step:** the step's two shapes and its length, then style tiles (Classic, Bake, Ripple, Melt, Pop, Dissolve) that show each timing as a curve.
- **Motion:**
  - *Spread from:* all at once, bottom up, top down, or a point you click. *Spread* sets how much the change rolls across the surface.
  - *Bounce* (overshoot and settle) and *Droop* (mid-change sag) in Topology, or *Clump* and *Puff* in Splats.
  - *Easing:* one curve canvas with a **Lead** curve (the parts reached first) and a **Trail** curve (the parts reached last). Link them to use one curve for everything.
- **Tools** (Alt + drag turns the camera while one is on):
  - **Navigate** (V)
  - **Start point** (O): click where the change begins.
  - **Timing brush** (B): paint parts to change *Sooner* or *Later*; Ctrl flips the direction.
  - **Pin** (P, Topology): click ① a point on the start shape, then ② where it must land on the end shape.

  While a tool is on, the shape shows the timing from orange (first) to blue (last).

Wave, brush, bounce and pins need a step whose two shapes share one mesh. Steps that change topology (Piecewise/SDF) follow the Lead curve only.

Scripted runs can set all of this too, for example:

```
menv --shot out/ripple --mode topology --keys bunny.obj,MaleHead.obj,heart_crystal.obj --style ripple --origin 0.3,0.9,0 --times 0.25,0.5,0.75
```

Other options are `--style`, `--wave`, `--bounce`, `--droop`, `--pin ax,ay,az:bx,by,bz`, `--tool start|brush|pin`, `--select k` and `--click x,y`.

## How to Run
Download the release, unzip it, and run the menv.exe.

Or build it yourself (Windows, Linux or macOS) with CMake 3.16+ and a C++17 compiler:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

Run it from the build folder (shaders and meshes are copied next to the program):

```bash
cd build && ./menv
```

* Windows uses the prebuilt GLFW in `external/lib`. Linux and macOS download and build GLFW 3.3.8 the first time you configure (pass `-DGC_SYSTEM_GLFW=ON` to use an installed one). On Linux, install the X11 headers first: `sudo apt-get install libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl-dev`.
* OpenGL is loaded with glad (`third_party/glad`), so no GLEW is needed.
* macOS only offers OpenGL 4.1, so the compute-shader playback is off there and the CPU paths run instead (same results, slower at high counts).

To import your own models (.obj, .glb, .gltf, or 3DGS .ply captures): add them to the `meshes` folder.

### Scripted runs
`menv --shot out/name --mode topology --a teapot.obj --b bunny.obj --times 0,0.5,1` bakes the morph without clicking, saves `out/name_t0.50.ppm` etc., and exits. Options: `--mode particles|topology|splats`, `--method auto|shared|sdf|piecewise`, `--transport greedy|gaussian|mixture`, `--k <components>`, `--size 1280x800`, `--nogui 1`.

### Input Bindings
* ESC — Close application
* R — Reset camera to default position
* Left Drag — Rotate view (azimuth + incline)
* Right Drag — Zoom in/out

Bear model credit: https://cse125.ucsd.edu/data/bear.html 
