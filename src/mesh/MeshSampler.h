#pragma once
#include "core.h"
#include "OBJLoader.h"
#include <vector>

class MeshSampler {
public:
    // Sample N points from triangle areas.
    // Large triangles has more points than small ones.
    // seed: random seed for reproducibility. 
    static std::vector<glm::vec3> Sample(const std::vector<Triangle>& triangles,
                                          int N,
                                          unsigned int seed = 42);

    // Same as Sample, but also reports which triangle each point came from,
    // and optionally where in it (barycentric weights of v0, v1, v2).
    static std::vector<glm::vec3> Sample(const std::vector<Triangle>& triangles,
                                          int N,
                                          unsigned int seed,
                                          std::vector<int>& outTriIndices,
                                          std::vector<glm::vec3>* outBary = nullptr);

    static float TotalArea(const std::vector<Triangle>& triangles);

private:
    //helper function to get triangle Area
    static float TriangleArea(const Triangle& t);
};
