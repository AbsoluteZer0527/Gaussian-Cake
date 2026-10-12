#include "MeshTopology.h"

#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>

static uint64_t EdgeKey(unsigned int a, unsigned int b) {
    if (a > b) std::swap(a, b);
    return ((uint64_t)a << 32) | (uint64_t)b;
}

// Number of faces using each undirected edge.
static std::unordered_map<uint64_t, int> EdgeFaceCounts(const IndexedMesh& m) {
    std::unordered_map<uint64_t, int> counts;
    counts.reserve(m.indices.size());
    for (size_t t = 0; t + 2 < m.indices.size(); t += 3)
        for (int k = 0; k < 3; ++k)
            counts[EdgeKey(m.indices[t + k], m.indices[t + (k + 1) % 3])]++;
    return counts;
}

std::string TopologyInfo::Describe() const {
    std::ostringstream ss;
    ss << verts << " verts, " << faces << " faces, " << parts << " part(s), chi=" << euler
       << ", " << boundaryLoops << " hole(s) (" << boundaryEdges << " open edges), "
       << nonManifoldEdges << " non-manifold edges";
    return ss.str();
}

void MeshTopology::Build(const std::vector<Triangle>& tris, IndexedMesh& out, float weldEps) {
    out.verts.clear();
    out.indices.clear();
    out.indices.reserve(tris.size() * 3);

    // Quantize to a grid of size weldEps so near-identical positions (seams
    // written twice in the OBJ) become one vertex.
    struct Key {
        int64_t x, y, z;
        bool operator==(const Key& o) const { return x == o.x && y == o.y && z == o.z; }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            return (size_t)(k.x * 73856093LL) ^ (size_t)(k.y * 19349663LL) ^ (size_t)(k.z * 83492791LL);
        }
    };
    std::unordered_map<Key, unsigned int, KeyHash> lookup;
    lookup.reserve(tris.size() * 2);

    auto addVertex = [&](const glm::vec3& v) -> unsigned int {
        Key key{ (int64_t)std::llround(v.x / weldEps),
                 (int64_t)std::llround(v.y / weldEps),
                 (int64_t)std::llround(v.z / weldEps) };
        auto it = lookup.find(key);
        if (it != lookup.end()) return it->second;
        unsigned int idx = (unsigned int)out.verts.size();
        out.verts.push_back(v);
        lookup.emplace(key, idx);
        return idx;
    };

    for (const auto& t : tris) {
        unsigned int a = addVertex(t.v0), b = addVertex(t.v1), c = addVertex(t.v2);
        if (a == b || b == c || c == a) continue;   // collapsed by the weld
        out.indices.push_back(a);
        out.indices.push_back(b);
        out.indices.push_back(c);
    }
}

std::vector<Triangle> MeshTopology::ToTriangles(const IndexedMesh& m) {
    std::vector<Triangle> tris(m.FaceCount());
    for (int f = 0; f < m.FaceCount(); ++f)
        tris[f] = { m.verts[m.indices[3 * f]], m.verts[m.indices[3 * f + 1]], m.verts[m.indices[3 * f + 2]] };
    return tris;
}

void MeshTopology::Subdivide(std::vector<Triangle>& tris, int levels) {
    for (int l = 0; l < levels; ++l) {
        std::vector<Triangle> result;
        result.reserve(tris.size() * 4);
        for (const auto& t : tris) {
            glm::vec3 m01 = (t.v0 + t.v1) * 0.5f;
            glm::vec3 m12 = (t.v1 + t.v2) * 0.5f;
            glm::vec3 m20 = (t.v2 + t.v0) * 0.5f;
            result.push_back({t.v0, m01, m20});
            result.push_back({m01, t.v1, m12});
            result.push_back({m20, m12, t.v2});
            result.push_back({m01, m12, m20});
        }
        tris = std::move(result);
    }
}

std::vector<std::vector<int>> MeshTopology::BoundaryLoops(const IndexedMesh& m) {
    auto counts = EdgeFaceCounts(m);

    // Boundary half-edges a -> b, in the direction their face uses them.
    std::vector<std::pair<int, int>> half;
    std::unordered_map<int, std::vector<int>> outgoing;   // vertex -> half-edge ids
    for (size_t t = 0; t + 2 < m.indices.size(); t += 3)
        for (int k = 0; k < 3; ++k) {
            unsigned int a = m.indices[t + k], b = m.indices[t + (k + 1) % 3];
            if (counts[EdgeKey(a, b)] != 1) continue;
            outgoing[(int)a].push_back((int)half.size());
            half.push_back({(int)a, (int)b});
        }

    std::vector<bool> used(half.size(), false);
    std::vector<std::vector<int>> loops;
    for (size_t h0 = 0; h0 < half.size(); ++h0) {
        if (used[h0]) continue;
        std::vector<int> loop;
        int h = (int)h0;
        while (h >= 0 && !used[h]) {
            used[h] = true;
            loop.push_back(half[h].first);
            int next = -1;
            for (int cand : outgoing[half[h].second])
                if (!used[cand]) { next = cand; break; }
            h = next;
        }
        if (loop.size() >= 3) loops.push_back(std::move(loop));
    }
    return loops;
}

int MeshTopology::FillSmallHoles(IndexedMesh& m, int maxLoopEdges) {
    int filled = 0;
    for (const auto& loop : BoundaryLoops(m)) {
        if ((int)loop.size() > maxLoopEdges) continue;

        glm::vec3 c(0.0f);
        for (int v : loop) c += m.verts[v];
        c /= (float)loop.size();
        unsigned int ci = (unsigned int)m.verts.size();
        m.verts.push_back(c);

        // The faces walk the loop a -> b; the patch must walk it b -> a.
        for (size_t i = 0; i < loop.size(); ++i) {
            unsigned int a = (unsigned int)loop[i];
            unsigned int b = (unsigned int)loop[(i + 1) % loop.size()];
            m.indices.push_back(b);
            m.indices.push_back(a);
            m.indices.push_back(ci);
        }
        ++filled;
    }
    return filled;
}

bool MeshTopology::OrientOutward(IndexedMesh& m) {
    double vol = 0.0;
    for (size_t t = 0; t + 2 < m.indices.size(); t += 3) {
        const glm::vec3& a = m.verts[m.indices[t]];
        const glm::vec3& b = m.verts[m.indices[t + 1]];
        const glm::vec3& c = m.verts[m.indices[t + 2]];
        vol += glm::dot(a, glm::cross(b, c));
    }
    if (vol >= 0.0) return false;
    for (size_t t = 0; t + 2 < m.indices.size(); t += 3)
        std::swap(m.indices[t + 1], m.indices[t + 2]);
    return true;
}

TopologyInfo MeshTopology::Analyze(const IndexedMesh& m) {
    TopologyInfo info;
    info.faces = m.FaceCount();

    // connected components over the vertices that faces actually use
    std::vector<int> parent(m.verts.size());
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&](int x) {
        while (parent[x] != x) x = parent[x] = parent[parent[x]];
        return x;
    };
    std::vector<bool> usedVert(m.verts.size(), false);
    for (size_t t = 0; t + 2 < m.indices.size(); t += 3) {
        int a = (int)m.indices[t], b = (int)m.indices[t + 1], c = (int)m.indices[t + 2];
        usedVert[a] = usedVert[b] = usedVert[c] = true;
        parent[find(a)] = find(b);
        parent[find(b)] = find(c);
    }
    for (size_t v = 0; v < m.verts.size(); ++v) {
        if (!usedVert[v]) continue;
        info.verts++;
        if (find((int)v) == (int)v) info.parts++;
    }

    auto counts = EdgeFaceCounts(m);
    for (const auto& kv : counts) {
        if (kv.second == 1) info.boundaryEdges++;
        else if (kv.second > 2) info.nonManifoldEdges++;
    }
    info.euler = info.verts - (int)counts.size() + info.faces;
    info.boundaryLoops = (int)BoundaryLoops(m).size();
    return info;
}

std::vector<std::vector<int>> MeshTopology::OneRings(const IndexedMesh& m) {
    std::vector<std::vector<int>> rings(m.verts.size());
    auto link = [&](int a, int b) {
        auto& r = rings[a];
        if (std::find(r.begin(), r.end(), b) == r.end()) r.push_back(b);
    };
    for (size_t t = 0; t + 2 < m.indices.size(); t += 3)
        for (int k = 0; k < 3; ++k) {
            int a = (int)m.indices[t + k], b = (int)m.indices[t + (k + 1) % 3];
            link(a, b);
            link(b, a);
        }
    return rings;
}

std::vector<glm::vec3> MeshTopology::VertexNormals(const IndexedMesh& m,
                                                   const std::vector<glm::vec3>& positions) {
    std::vector<glm::vec3> n(positions.size(), glm::vec3(0.0f));
    for (size_t t = 0; t + 2 < m.indices.size(); t += 3) {
        unsigned int a = m.indices[t], b = m.indices[t + 1], c = m.indices[t + 2];
        glm::vec3 fn = glm::cross(positions[b] - positions[a], positions[c] - positions[a]);
        n[a] += fn; n[b] += fn; n[c] += fn;
    }
    for (auto& v : n) {
        float len = glm::length(v);
        v = (len > 1e-20f) ? v / len : glm::vec3(0.0f, 1.0f, 0.0f);
    }
    return n;
}
