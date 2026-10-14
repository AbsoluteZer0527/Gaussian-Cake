# Gaussian Cake

## Demo: https://youtu.be/RBu-PR3ZYA4
<img width="2100" height="1247" alt="Screenshot 2026-04-05 005843" src="https://github.com/user-attachments/assets/c7f926b8-0df9-4d66-8b2c-34ff7d3b350e" />
<img width="1544" height="1029" alt="image" src="https://github.com/user-attachments/assets/dcbf13d9-b03c-4d5f-9efe-094bd0207caf" />

## Update 10/13/26
### Model Morphing: Shared Topology & Gaussian Splatting
Two new OT morphing mode were building on top of the Original Gaussian Cake.


https://github.com/user-attachments/assets/98e52f4b-2f6b-402d-9be6-54df5fd2eb13


**Topology Mode**: We first check whether the two models have the sharing topology first, then we uses OT to determine the path for the two mesh's triangle topology and morph into another shape. It uses mesh with more faces as the template, and fits that topology to the other model and show the morph animation.If they don't share the same topology, such as mismatch or different number of genus, it will use the SDF path instead.

SDF: in-between shapes are aligned by two lerp: M_t = (1−t)·I + t·T, making each model a signed distance field and match the surfaces.

https://github.com/user-attachments/assets/53793805-99b7-4744-a415-52867123ba84

**Splats Mode**: Each Particles is getting replaced by 3D Gaussian blobs, it stores a center, covariance, color and opacity. Then we applies OT between the mixtures of the two models: we first estimate the moving cost for each blob(Bures cost), and matches them after (Sinkhorn), where it shifts and streches each of those blobs to the other location. At the end, we simulate morphing animation and show the cooking.

References: 
Chen, Georgiou, Tannenbaum 2019, Optimal Transport for Gaussian Mixture Models. arXiv:1710.07876

Dowson & Landau 1982, The Fréchet distance between multivariate normal distributions doi:10.1016/0047-259X(82)90077-X 

Peyré & Cuturi 2019, Computational Optimal Transport. arXiv:1803.00567

## Inspiration
This project is inspired by the mathematical problem of **Optimal Transport (OT)**, which seeks a mapping between two distributions that minimizes a specified transport cost. 

Think about this question: "If I have a pile of stuff in one shape, and I want to rearrange it into another shape, what is the cheapest way to move it?"

![Alt text](https://www.microsoft.com/en-us/research/wp-content/uploads/2020/09/OTDD_Shovel-Figure.png)

And if you have ever heard of this website: https://obamify.com/, it shares a similar concept, but instead, this program is 3D.

## What it does
This is an experimental graphic project that mainly serves the purpose of applying the math concept of Optimal Transport into an actual computer animation pipeline.

A low-level C++ OpenGL program that takes in **any .obj imported models** : A and B. Once you click "Bake It!", the program shows either (1) a **smooth animation of the model morphing from A to B**, or (2) transport animation of a **Gaussian distribution sampling from A to B**(so far the program only supports default -> model and model -> default). This is a particle system animation project that showcases the concept of optimal transport in two ways: GreedyDiscrete OT for the mesh and Gaussian OT for the distribution.

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

## How to Run
Download the release, unzip it, and run the menv.exe.

To import your own .obj models: Add to the `meshes` folder.

### Input Bindings
* ESC — Close application
* R — Reset camera to default position
* Left Drag — Rotate view (azimuth + incline)
* Right Drag — Zoom in/out

Bear model credit: https://cse125.ucsd.edu/data/bear.html 
