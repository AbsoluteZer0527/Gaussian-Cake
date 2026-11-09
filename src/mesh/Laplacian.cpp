#include "Laplacian.h"
#include <Eigen/Dense>

// Cotan weight of every edge of forms.edges on the given positions:
// w_ij = (cot alpha + cot beta) / 2, alpha and beta the angles opposite the edge.
// This is the diagonal Hodge star on 1-forms (DEC's *1). Obtuse angles can make
// a weight negative; that's fine and kept as is: d0^T W d0 is the Dirichlet
// energy of piecewise-linear functions, so it stays positive semidefinite, and
// clamping would cost the exact "linear functions have zero Laplacian" property.
// Slivers are the real danger: a near-zero-area triangle has a cot near +-1e8,
// so those faces are skipped (their edges keep the weight from the other side).
std::vector<double> CotanWeights(const IndexedMesh& m, const DiscreteForms& forms,
                                        const std::vector<glm::vec3>& pos) {
    const int E = (int)forms.edges.size();
    std::unordered_map<uint64_t, int> edgeId;
    edgeId.reserve(E);
    for (int e = 0; e < E; ++e)
        edgeId[PairKey(forms.edges[e][0], forms.edges[e][1])] = e;

    std::vector<double> w(E, 0.0);
    for (int f = 0; f < m.FaceCount(); ++f) {
        // shape quality: 1 for equilateral, 0 for a sliver (2 area / sum of squared edges, scaled)
        glm::vec3 p0 = pos[m.indices[3 * f + 0]];
        glm::vec3 p1 = pos[m.indices[3 * f + 1]];
        glm::vec3 p2 = pos[m.indices[3 * f + 2]];
        double twiceArea = ToE(p1 - p0).cross(ToE(p2 - p0)).norm();
        double edgeSq = ToE(p1 - p0).squaredNorm() + ToE(p2 - p1).squaredNorm() + ToE(p0 - p2).squaredNorm();
        if (edgeSq <= 0.0 || 2.0 * std::sqrt(3.0) * twiceArea / edgeSq < 1e-4) continue;

        for (int k = 0; k < 3; ++k) {
            // corner c, and the edge (a, b) across from it
            int c = (int)m.indices[3 * f + k];
            int a = (int)m.indices[3 * f + (k + 1) % 3];
            int b = (int)m.indices[3 * f + (k + 2) % 3];

            Eigen::Vector3d u = ToE(pos[a] - pos[c]);
            Eigen::Vector3d v = ToE(pos[b] - pos[c]);
            double sinArea = u.cross(v).norm();   // |u||v| sin(angle), > 0 after the check above
            double cot = u.dot(v) / sinArea;      // |u||v| cos / |u||v| sin

            w[edgeId[PairKey(a, b)]] += 0.5 * cot;
        }
    }
    return w;
}


SpMat CotanMatrix(const IndexedMesh& m, const std::vector<glm::vec3>& pos) {
    DiscreteForms forms = MeshTopology::BuildForms(m);
    std::vector<double> w = CotanWeights(m, forms, pos);
    const int E = (int)forms.edges.size();
    SpMat W(E, E);
    std::vector<Eigen::Triplet<double>> diag;
    diag.reserve(E);
    for (int e = 0; e < E; ++e) diag.emplace_back(e, e, w[e]);
    W.setFromTriplets(diag.begin(), diag.end());
    return SpMat(forms.d0.transpose()) * W * forms.d0;
}

std::vector<double> VertexAreas(const IndexedMesh& m, const std::vector<glm::vec3>& pos) {
    std::vector<double> a(pos.size(), 0.0);
    for (int f = 0; f < m.FaceCount(); ++f) {
        unsigned int i = m.indices[3 * f], j = m.indices[3 * f + 1], k = m.indices[3 * f + 2];
        double area = 0.5 * glm::length(glm::cross(pos[j] - pos[i], pos[k] - pos[i]));
        a[i] += area / 3.0;
        a[j] += area / 3.0;
        a[k] += area / 3.0;
    }
    return a;
}
