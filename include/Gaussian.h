#pragma once
#include "core.h"
#include <vector>
#include <Eigen/Dense>

class Gaussian {
public:
    Eigen::Vector3d mean;
    Eigen::Matrix3d covariance;
    Gaussian();

    //find mean and covariance from the point cloud.
    void FitFromPoints(const std::vector<glm::vec3>& points, bool verbose = true);

    //sample N points from N(mean, covariance)
    //cholesky decomposition
    std::vector<glm::vec3> Sample(int N, unsigned int seed = 42) const;


    //get the eigenvalues and eigenvectors of the covariance matrix.
    //eigenvalues = variance along each principal axis.
    //eigenvectors = directions of those axes.
    //used by EllipsoidRenderer to draw the 1-sigma shell.
    void GetEigenDecomposition(Eigen::Vector3d& eigenvalues,
                                Eigen::Matrix3d& eigenvectors) const;

    //converters for glm
    glm::vec3 GetMeanGLM() const;
    glm::mat3 GetCovarianceGLM() const;
};
