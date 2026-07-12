# Gaussian Cake

## Demo: https://youtu.be/RBu-PR3ZYA4
<img width="2100" height="1247" alt="Screenshot 2026-04-05 005843" src="https://github.com/user-attachments/assets/c7f926b8-0df9-4d66-8b2c-34ff7d3b350e" />
<img width="1544" height="1029" alt="image" src="https://github.com/user-attachments/assets/dcbf13d9-b03c-4d5f-9efe-094bd0207caf" />

## Update 7/11/26
### Model Morphing: Triangle mesh rather than particles
1. loads source triangles → fits Gaussians to both meshes → computes OT map → applies T(v) to every source vertex → initializes both meshRenderer and animator. This allows the model to be "linearly warped (rotation + scale + shear) into another models' Gaussian with lerp. So the source topology is preserved; only the endpoint is approximate.

2. Given a point P and triangle (A, B, C), find Q on the triangle surface (interior or boundary) minimizing ‖P − Q‖. The classic algorithm uses Voronoi regions: the plane containing the triangle is subdivided into 7 regions by the triangle edges' perpendiculars — one for each vertex (P projects to the vertex), one for each edge (P projects to somewhere on the edge), and one for the interior (P projects into the triangle). Which region P falls into is determined by sign tests on ~6 dot products.
![alt text](image.png)
Reference: Christer Ericson, Real-Time Collision Detection (2005), §5.1.5 — the standard implementation.


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
