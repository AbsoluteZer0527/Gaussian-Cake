#include "PiecewiseMorph.h"
#include "Parallel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

// The SDF surface at t as an indexed mesh, without specks (connected pieces
// with fewer than 1% of the faces, or 20). stride > 1: a coarser surface.
static IndexedMesh CleanSurface(const SDFMorph& sdf, float t, int stride = 1) {
    IndexedMesh m;
    sdf.Extract(t, m.verts, m.indices, nullptr, stride);
    const int faces = m.FaceCount();
    if (faces == 0) return m;

    std::vector<int> parent(m.verts.size());
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&](int x) {
        while (parent[x] != x) x = parent[x] = parent[parent[x]];
        return x;
    };
    for (int f = 0; f < faces; ++f) {
        int a = find((int)m.indices[3 * f]);
        for (int c = 1; c < 3; ++c) {
            int b = find((int)m.indices[3 * f + c]);
            if (a != b) parent[b] = a;
        }
    }
    std::vector<int> size(m.verts.size(), 0);
    for (int f = 0; f < faces; ++f) size[find((int)m.indices[3 * f])]++;
    const int minFaces = std::max(20, faces / 100);

    IndexedMesh out;
    std::vector<int> remap(m.verts.size(), -1);
    for (int f = 0; f < faces; ++f) {
        if (size[find((int)m.indices[3 * f])] < minFaces) continue;
        for (int c = 0; c < 3; ++c) {
            unsigned v = m.indices[3 * f + c];
            if (remap[v] < 0) {
                remap[v] = (int)out.verts.size();
                out.verts.push_back(m.verts[v]);
            }
            out.indices.push_back((unsigned)remap[v]);
        }
    }
    return out;
}

// Labels the 6-connected pieces of the grid points where mask == want.
// Returns the piece sizes; label[i] = piece index, or -1.
static std::vector<int> Pieces(const std::vector<char>& mask, char want, int n, std::vector<int>& label,
                               std::vector<char>* touchesBorder = nullptr) {
    const size_t count = mask.size();
    label.assign(count, -1);
    std::vector<int> sizes;
    std::vector<int> stack;
    if (touchesBorder) touchesBorder->clear();
    for (size_t s = 0; s < count; ++s) {
        if (mask[s] != want || label[s] >= 0) continue;
        int id = (int)sizes.size();
        int size = 0;
        bool border = false;
        stack.push_back((int)s);
        label[s] = id;
        while (!stack.empty()) {
            int p = stack.back();
            stack.pop_back();
            ++size;
            int i = p % n, j = (p / n) % n, k = p / (n * n);
            if (i == 0 || j == 0 || k == 0 || i == n - 1 || j == n - 1 || k == n - 1) border = true;
            const int nb[6][3] = { {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1} };
            for (const auto& d : nb) {
                int a = i + d[0], b = j + d[1], c = k + d[2];
                if (a < 0 || b < 0 || c < 0 || a >= n || b >= n || c >= n) continue;
                int q = (c * n + b) * n + a;
                if (mask[q] == want && label[q] < 0) {
                    label[q] = id;
                    stack.push_back(q);
                }
            }
        }
        sizes.push_back(size);
        if (touchesBorder) touchesBorder->push_back(border ? 1 : 0);
    }
    return sizes;
}

// Read off the grid, not the extracted mesh: surface nets can leave
// non-manifold spots that throw the mesh's Euler count off, while the solid
// (the grid points with phi_t < 0, joined along grid edges into a cubical
// complex) has an exact Euler characteristic chi = V - E + F - C, and for a
// solid chi = parts - genus + cavities.
PiecewiseMorph::Topology PiecewiseMorph::SolidTopology(const SDFMorph& sdf, float t) {
    int n;
    glm::vec3 lo;
    float h;
    std::vector<float> f;
    sdf.Field(t, 1, n, lo, h, f);
    const size_t count = f.size();
    std::vector<char> in(count);
    size_t inside = 0;
    for (size_t i = 0; i < count; ++i) {
        in[i] = f[i] < 0.0f;
        inside += in[i];
    }
    Topology topo;
    if (inside == 0) return topo;
    const int minSize = std::max<int>(8, (int)(inside / 200));

    // drop specks of solid, fill tiny bubbles
    std::vector<int> label;
    std::vector<int> sizes = Pieces(in, 1, n, label);
    for (size_t i = 0; i < count; ++i)
        if (in[i] && sizes[label[i]] < minSize) in[i] = 0;
    std::vector<char> border;
    std::vector<int> holes = Pieces(in, 0, n, label, &border);
    for (size_t i = 0; i < count; ++i)
        if (!in[i] && !border[label[i]] && holes[label[i]] < minSize) in[i] = 1;

    sizes = Pieces(in, 1, n, label);
    topo.parts = (int)sizes.size();
    holes = Pieces(in, 0, n, label, &border);
    for (char b : border) topo.cavities += b ? 0 : 1;

    // chi of the cubical complex
    auto I = [&](int i, int j, int k) -> int {
        return (i < n && j < n && k < n) ? in[((size_t)k * n + j) * n + i] : 0;
    };
    long long V = 0, E = 0, F = 0, C = 0;
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                if (!I(i, j, k)) continue;
                ++V;
                int x = I(i + 1, j, k), y = I(i, j + 1, k), z = I(i, j, k + 1);
                E += x + y + z;
                int xy = x && y && I(i + 1, j + 1, k);
                int yz = y && z && I(i, j + 1, k + 1);
                int xz = x && z && I(i + 1, j, k + 1);
                F += xy + yz + xz;
                C += xy && z && I(i + 1, j, k + 1) && I(i, j + 1, k + 1) && I(i + 1, j + 1, k + 1);
            }
    long long chi = V - E + F - C;
    topo.genus = (int)(topo.parts + topo.cavities - chi);
    return topo;
}

static std::string DescribeChange(const PiecewiseMorph::Topology& a, const PiecewiseMorph::Topology& b) {
    std::string out;
    char buf[64];
    if (a.parts != b.parts) {
        snprintf(buf, sizeof(buf), "%d -> %d parts", a.parts, b.parts);
        out += buf;
    }
    if (a.genus != b.genus) {
        snprintf(buf, sizeof(buf), "%sgenus %d -> %d", out.empty() ? "" : ", ", a.genus, b.genus);
        out += buf;
    }
    if (a.cavities != b.cavities) {
        snprintf(buf, sizeof(buf), "%s%d -> %d cavities", out.empty() ? "" : ", ", a.cavities, b.cavities);
        out += buf;
    }
    return out;
}

bool PiecewiseMorph::Build(const SDFMorph& sdf, const IndexedMesh& a, const IndexedMesh& b,
                           const std::vector<glm::vec3>& colorsA, const std::vector<glm::vec3>& colorsB,
                           const SurfaceColor* colorA, const SurfaceColor* colorB, const Params& params) {
    auto t0 = std::chrono::steady_clock::now();
    segments.clear();
    events.clear();
    fits = 0;
    if (!sdf.IsBuilt()) return false;

    // 1. Topology along the timeline, then each change narrowed down by
    //    bisection. left[i] / right[i]: the last time before and the first
    //    time after event i (both on the SDF morph's own surfaces).
    const int samples = std::max(4, params.keyframes);
    // Topology at a finer row of times than the keyframes, so short-lived
    // changes are seen (and can be dropped below)
    const int probes = 4 * samples;
    std::vector<Topology> topo(probes + 1);
    for (int k = 0; k <= probes; ++k) {
        topo[k] = SolidTopology(sdf, (float)k / probes);
        if (params.progress) params.progress(0, k + 1, probes + 1);
    }

    // A change that undoes itself within a short time (a hole that flickers
    // open and shut) is noise, not an event: smooth those runs away.
    const int minRun = std::max(2, probes / 16);
    for (int pass = 0; pass < 3; ++pass) {
        int k = 1;
        while (k < probes) {
            if (topo[k] == topo[k - 1]) { ++k; continue; }
            int run = k;
            while (run + 1 <= probes && topo[run + 1] == topo[k]) ++run;
            int len = run - k + 1;
            if (run < probes && len < minRun)            // short island: back to what came before
                for (int q = k; q <= run; ++q) topo[q] = topo[k - 1];
            k = run + 1;
        }
    }

    std::vector<float> left, right;
    for (int k = 0; k < probes; ++k) {
        if (topo[k] == topo[k + 1]) continue;
        float lo = (float)k / probes, hi = (float)(k + 1) / probes;
        Topology before = topo[k], after = topo[k + 1];
        for (int it = 0; it < 5; ++it) {
            float mid = 0.5f * (lo + hi);
            if (SolidTopology(sdf, mid) == before) lo = mid; else hi = mid;
        }
        left.push_back(lo);
        right.push_back(hi);
        events.push_back({ 0.5f * (lo + hi), DescribeChange(before, after) });
    }
    // No change on the way (A and B differ only in ways the SDF surfaces
    // don't show, e.g. a seam): A's mesh carries the first half, B's the second.
    if (events.empty()) {
        left.push_back(0.5f);
        right.push_back(0.5f);
        events.push_back({ 0.5f, "" });
    }

    // 2. Segments: [0, e1], [e1, e2], ..., [em, 1]
    const int segCount = (int)events.size() + 1;
    segments.resize(segCount);
    for (int s = 0; s < segCount; ++s) {
        segments[s].tStart = (s == 0) ? 0.0f : events[s - 1].t;
        segments[s].tEnd   = (s == segCount - 1) ? 1.0f : events[s].t;
    }

    // keyframe times strictly between lo and hi, on the regular grid
    auto inner = [&](float lo, float hi) {
        std::vector<float> ts;
        for (int k = 1; k < samples; ++k) {
            float t = (float)k / samples;
            if (t > lo + 1e-4f && t < hi - 1e-4f) ts.push_back(t);
        }
        return ts;
    };

    // Fits the template along `times` (in that order), starting from `start`
    // at the first time; returns one position set per time.
    int plannedFits = 1, fitsDone = 0;   // for the progress (counted below)
    auto follow = [&](const IndexedMesh& tmpl, const std::vector<glm::vec3>& start,
                      const std::vector<float>& times, bool firstIsExact) {
        std::vector<std::vector<glm::vec3>> out;
        std::vector<glm::vec3> p = start;
        for (size_t i = 0; i < times.size(); ++i) {
            if (i > 0 || !firstIsExact) {
                IndexedMesh target = CleanSurface(sdf, times[i]);
                if (target.FaceCount() > 0) {
                    p = MeshFitter::Fit(tmpl, p, MeshTopology::ToTriangles(target), params.fit);
                    ++fits;
                    if (params.progress) params.progress(1, std::min(++fitsDone, plannedFits), plannedFits);
                }
            }
            out.push_back(p);
        }
        return out;
    };

    // Colours: a segment's own end mesh keeps its exact colours and blends
    // toward the other shape's colour where the morph says it came from.
    auto colorsFor = [&](PiecewiseSegment& seg, int s) {
        if (!colorA || !colorB) return;
        for (size_t k = 0; k < seg.times.size(); ++k) {
            const float t = seg.times[k];
            const std::vector<glm::vec3>& p = seg.positions[k];
            std::vector<glm::vec3> us(p.size()), tus(p.size());
            ParallelFor((int)p.size(), [&](int v) { us[v] = sdf.Preimage(p[v], t, &tus[v]); });
            std::vector<glm::vec3> ca = (s == 0 && !colorsA.empty()) ? colorsA : colorA->At(us);
            std::vector<glm::vec3> cb = (s == segCount - 1 && !colorsB.empty()) ? colorsB : colorB->At(tus);
            std::vector<glm::vec3> c(p.size());
            for (size_t v = 0; v < c.size(); ++v) c[v] = glm::mix(ca[v], cb[v], t);
            seg.colors.push_back(std::move(c));
        }
    };

    // How many fits are coming (for the progress): each keyframe but the
    // exact first one of every chain
    plannedFits = 0;
    for (int s = 0; s < segCount; ++s) {
        const float lo = (s == 0) ? 0.0f : right[s - 1];
        const float hi = (s == segCount - 1) ? 1.0f : left[s];
        if (s == 0) plannedFits += (int)inner(0.0f, hi).size() + (hi > 0.0f ? 1 : 0);
        else if (s == segCount - 1) plannedFits += (int)inner(lo, 1.0f).size() + (lo < 1.0f ? 1 : 0);
        else {
            const float mid = 0.5f * (lo + hi);
            plannedFits += (int)inner(mid, hi).size() + 1 + (int)inner(lo, mid).size() + 1;
        }
    }
    plannedFits = std::max(plannedFits, 1);
    if (params.progress) params.progress(1, 0, plannedFits);

    for (int s = 0; s < segCount; ++s) {
        PiecewiseSegment& seg = segments[s];
        const float lo = (s == 0) ? 0.0f : right[s - 1];
        const float hi = (s == segCount - 1) ? 1.0f : left[s];
        if (s == 0) {
            // A, forwards from t = 0
            std::vector<float> ts = { 0.0f };
            for (float t : inner(0.0f, hi)) ts.push_back(t);
            if (hi > 0.0f) ts.push_back(hi);
            seg.indices = a.indices;
            seg.times = ts;
            seg.positions = follow(a, a.verts, ts, true);
        } else if (s == segCount - 1) {
            // B, backwards from t = 1
            std::vector<float> ts = { 1.0f };
            std::vector<float> in = inner(lo, 1.0f);
            for (auto it = in.rbegin(); it != in.rend(); ++it) ts.push_back(*it);
            if (lo < 1.0f) ts.push_back(lo);
            std::vector<std::vector<glm::vec3>> pos = follow(b, b.verts, ts, true);
            std::reverse(ts.begin(), ts.end());
            std::reverse(pos.begin(), pos.end());
            seg.indices = b.indices;
            seg.times = ts;
            seg.positions = std::move(pos);
        } else {
            // the SDF surface in the middle of the segment, both ways from there
            const float mid = 0.5f * (lo + hi);
            // a coarser extraction keeps the template light enough to fit quickly
            IndexedMesh tmpl = CleanSurface(sdf, mid, 2);
            if (tmpl.FaceCount() == 0) return false;
            std::vector<float> fwd = { mid }, back = { mid };
            for (float t : inner(mid, hi)) fwd.push_back(t);
            fwd.push_back(hi);
            std::vector<float> in = inner(lo, mid);
            for (auto it = in.rbegin(); it != in.rend(); ++it) back.push_back(*it);
            back.push_back(lo);
            auto pf = follow(tmpl, tmpl.verts, fwd, true);
            auto pb = follow(tmpl, tmpl.verts, back, true);
            seg.indices = tmpl.indices;
            for (int i = (int)back.size() - 1; i >= 1; --i) {
                seg.times.push_back(back[i]);
                seg.positions.push_back(pb[i]);
            }
            for (size_t i = 0; i < fwd.size(); ++i) {
                seg.times.push_back(fwd[i]);
                seg.positions.push_back(pf[i]);
            }
        }
        colorsFor(seg, s);
    }

    ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

int PiecewiseMorph::SegmentAt(float t) const {
    for (int s = 0; s + 1 < (int)segments.size(); ++s)
        if (t < segments[s].tEnd) return s;
    return (int)segments.size() - 1;
}

void PiecewiseMorph::Evaluate(int segment, float t, std::vector<glm::vec3>& positions,
                              std::vector<glm::vec3>& colors) const {
    const PiecewiseSegment& seg = segments[segment];
    const std::vector<float>& ts = seg.times;
    t = glm::clamp(t, ts.front(), ts.back());
    size_t k = 0;
    while (k + 2 < ts.size() && t > ts[k + 1]) ++k;
    size_t k1 = std::min(k + 1, ts.size() - 1);
    float span = ts[k1] - ts[k];
    float w = span > 0.0f ? (t - ts[k]) / span : 0.0f;

    const auto& p0 = seg.positions[k];
    const auto& p1 = seg.positions[k1];
    positions.resize(p0.size());
    for (size_t v = 0; v < p0.size(); ++v) positions[v] = glm::mix(p0[v], p1[v], w);
    colors.clear();
    if (!seg.colors.empty()) {
        const auto& c0 = seg.colors[k];
        const auto& c1 = seg.colors[k1];
        colors.resize(c0.size());
        for (size_t v = 0; v < c0.size(); ++v) colors[v] = glm::mix(c0[v], c1[v], w);
    }
}
