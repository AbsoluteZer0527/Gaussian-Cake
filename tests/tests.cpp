// Small checks that run without a window or GPU: ctest (or ./gc_tests).
#include "PlyLoader.h"
#include "MeshSampler.h"
#include "MeshTopology.h"
#include "OptimalTransport.h"
#include "SDFMorph.h"
#include "PiecewiseMorph.h"
#include "SphericalHarmonics.h"
#include "SplatHeat.h"
#include "MeshFitter.h"
#include "MorphCurve.h"
#include "MorphStyle.h"
#include "MorphSpring.h"
#include "MorphTimeline.h"
#include "Picking.h"

#include <algorithm>
#include <numeric>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

static int g_failed = 0;

#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            ++g_failed;                                               \
            std::printf("  FAIL %s:%d: ", __FILE__, __LINE__);        \
            std::printf(__VA_ARGS__);                                 \
            std::printf("\n");                                        \
        }                                                             \
    } while (0)

static glm::vec3 RandomDir(std::mt19937& rng) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    return glm::normalize(glm::vec3(g(rng), g(rng), g(rng)));
}

static SH::Coeffs RandomCoeffs(std::mt19937& rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    SH::Coeffs c;
    for (glm::vec3& v : c) v = glm::vec3(u(rng), u(rng), u(rng));
    return c;
}

// Turning coefficients by R must equal reading the old function at R^T d.
static void TestShRotation() {
    std::mt19937 rng(7);
    float worst = 0.0f;
    for (int trial = 0; trial < 20; ++trial) {
        glm::vec3 axis = RandomDir(rng);
        float angle = std::uniform_real_distribution<float>(0.0f, 6.28f)(rng);
        glm::mat3 R = glm::mat3_cast(glm::angleAxis(angle, axis));
        SH::Coeffs c = RandomCoeffs(rng), turned = c;
        SH::Rotate(turned, SH::RotationMatrix(R));
        for (int k = 0; k < 20; ++k) {
            glm::vec3 d = RandomDir(rng);
            glm::vec3 a = SH::Eval(turned, 3, d);
            glm::vec3 b = SH::Eval(c, 3, glm::transpose(R) * d);
            worst = std::max(worst, glm::length(a - b));
        }
    }
    std::printf("SH rotation: worst error %.2e\n", worst);
    CHECK(worst < 1e-4f, "rotated SH differs by %g", worst);
}

// Writes a 3DGS-layout .ply with `count` splats and degree-3 colour.
static void WritePly(const std::string& path, const std::vector<std::vector<float>>& rows) {
    std::ofstream out(path, std::ios::binary);
    out << "ply\nformat binary_little_endian 1.0\nelement vertex " << rows.size() << "\n";
    const char* names[] = { "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2" };
    for (const char* n : names) out << "property float " << n << "\n";
    for (int r = 0; r < 45; ++r) out << "property float f_rest_" << r << "\n";
    for (const char* n : { "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3" })
        out << "property float " << n << "\n";
    out << "end_header\n";
    for (const auto& row : rows) out.write((const char*)row.data(), row.size() * sizeof(float));
}

// f_rest is channel-major in the file; Splat::sh is coefficient-major.
static void TestPlyShLayout() {
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    const int n = 200;
    std::vector<std::vector<float>> rows(n);
    for (int i = 0; i < n; ++i) {
        std::vector<float>& r = rows[i];
        r = { u(rng), u(rng), u(rng), 0, 0, 0, u(rng), u(rng), u(rng) };
        for (int k = 0; k < 45; ++k) r.push_back(1000.0f * i + k);   // value says where it came from
        // high opacity, small isotropic scale, identity rotation (w first)
        for (float v : { 5.0f, -4.0f, -4.0f, -4.5f, 1.0f, 0.0f, 0.0f, 0.0f }) r.push_back(v);
    }
    const std::string path = "gc_test_capture.ply";
    WritePly(path, rows);

    PlyLoadOptions opt;
    opt.up = PlyLoadOptions::Up::AsIs;
    opt.cropRadius = 100.0f;
    std::vector<Splat> splats;
    CHECK(PlyLoader::Load(path, splats, opt), "load failed");
    CHECK((int)splats.size() == n, "%d splats kept", (int)splats.size());
    int bad = 0;
    for (const Splat& s : splats) {
        int i = (int)std::lround(s.sh[0].r / 1000.0f);   // which row it was
        for (int k = 0; k < 15; ++k)
            for (int ch = 0; ch < 3; ++ch)
                if (s.sh[k][ch] != 1000.0f * i + ch * 15 + k) ++bad;
    }
    std::printf("PLY SH layout: %d mismatched coefficients\n", bad);
    CHECK(bad == 0, "%d coefficients in the wrong place", bad);

    // FlipY turns the capture 180 degrees about X: colour seen from d before
    // must be seen from the turned direction after.
    std::vector<Splat> flipped;
    opt.up = PlyLoadOptions::Up::FlipY;
    for (auto& r : rows) for (int k = 9; k < 54; ++k) r[k] = u(rng);
    WritePly(path, rows);
    PlyLoader::Load(path, splats, PlyLoadOptions{ 100.0f, PlyLoadOptions::Up::AsIs });
    PlyLoader::Load(path, flipped, PlyLoadOptions{ 100.0f, PlyLoadOptions::Up::FlipY });
    glm::mat3 R = glm::mat3_cast(glm::angleAxis(glm::pi<float>(), glm::vec3(1, 0, 0)));
    float worst = 0.0f;
    for (size_t i = 0; i < std::min(splats.size(), flipped.size()); ++i) {
        glm::vec3 d = RandomDir(rng);
        worst = std::max(worst, glm::length(SH::Eval(flipped[i].sh, 3, R * d) - SH::Eval(splats[i].sh, 3, d)));
    }
    std::printf("PLY FlipY SH: worst error %.2e\n", worst);
    CHECK(worst < 1e-4f, "flipped SH differs by %g", worst);
    std::remove(path.c_str());
}

// Sphere of radius r at c: a subdivided octahedron pushed onto the sphere.
static std::vector<Triangle> Sphere(glm::vec3 c, float r, int levels = 3) {
    glm::vec3 p[6] = { {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1} };
    int f[8][3] = { {0,2,4}, {2,1,4}, {1,3,4}, {3,0,4}, {2,0,5}, {1,2,5}, {3,1,5}, {0,3,5} };
    std::vector<Triangle> tris;
    for (auto& t : f) tris.push_back({ p[t[0]], p[t[1]], p[t[2]] });
    MeshTopology::Subdivide(tris, levels);
    for (Triangle& t : tris) {
        t.v0 = c + r * glm::normalize(t.v0);
        t.v1 = c + r * glm::normalize(t.v1);
        t.v2 = c + r * glm::normalize(t.v2);
    }
    return tris;
}

// The mixture-aligned SDF morph: fold-free, Newton finds the preimage, and
// A lines up with B better than with one affine map.
static void TestMixtureWarp() {
    std::vector<Triangle> a = Sphere(glm::vec3(0.0f), 0.8f);
    std::vector<Triangle> b = Sphere(glm::vec3(-0.6f, 0.0f, 0.0f), 0.35f);
    std::vector<Triangle> b2 = Sphere(glm::vec3(0.6f, 0.2f, 0.0f), 0.3f);
    b.insert(b.end(), b2.begin(), b2.end());   // two blobs

    std::vector<glm::vec3> sa = MeshSampler::Sample(a, 4000), sb = MeshSampler::Sample(b, 4000);
    Gaussian ga, gb;
    ga.FitFromPoints(sa, false);
    gb.FitFromPoints(sb, false);
    OTMap map = OptimalTransport::Compute(ga, gb);
    GaussianMixture ma, mb;
    ma.Fit(sa, 6);
    mb.Fit(sb, 6);
    MixtureOTMap mix = OptimalTransport::ComputeMixture(ma, mb, 0.05, 1.0f);

    SDFMorph morph;
    morph.Build(a, b, map, 48, nullptr, nullptr, &mix);
    std::printf("Mixture warp: strength %.2f\n", morph.warpStrength);
    CHECK(morph.HasWarp() && morph.warpStrength > 0.2f, "strength %g", morph.warpStrength);

    // Newton: M_t(Preimage(y)) = y, where the shapes are. (Far out in the
    // grid corners the preimage can lie past the grid, where the warp is only
    // extrapolated; both distances are large there, so no surface depends on it.)
    float worst = 0.0f;
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> u(-0.9f, 0.9f);
    for (float t : { 0.25f, 0.5f, 0.75f }) {
        for (int k = 0; k < 500; ++k) {
            glm::vec3 y(u(rng), u(rng), u(rng)), Tu;
            glm::vec3 x = morph.Preimage(y, t, &Tu);
            float r = glm::length((1.0f - t) * x + t * Tu - y);
            worst = std::max(worst, r);
        }
    }
    // Newton converges to well inside a grid cell (the trilinear warp's
    // derivative jumps at cell faces, so not to round-off everywhere)
    const float cell = morph.GridA().h;
    std::printf("Mixture warp: worst preimage residual %.2e (%.3f grid cells)\n", worst, worst / cell);
    CHECK(worst < 0.2f * cell, "preimage residual %g (%g cells)", worst, worst / cell);

    float affine = morph.AlignmentError(sa, true), used = morph.AlignmentError(sa, false);
    std::printf("Mixture warp: A lands %.3f from B (affine %.3f)\n", used, affine);
    CHECK(used < affine, "mixture alignment %g not better than affine %g", used, affine);

    // the surface exists all the way through
    for (float t : { 0.0f, 0.5f, 1.0f }) {
        std::vector<glm::vec3> v;
        std::vector<unsigned int> idx;
        morph.Extract(t, v, idx);
        CHECK(!idx.empty(), "empty surface at t = %g", t);
    }
}

static std::vector<Triangle> Torus(float R, float r, int N = 48, int M = 24) {
    auto P = [&](int i, int j) {
        float u = 6.2831853f * i / N, v = 6.2831853f * j / M;
        return glm::vec3((R + r * std::cos(v)) * std::cos(u), r * std::sin(v), (R + r * std::cos(v)) * std::sin(u));
    };
    std::vector<Triangle> tris;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j) {
            glm::vec3 a = P(i, j), b = P(i + 1, j), c = P(i + 1, j + 1), d = P(i, j + 1);
            tris.push_back({ a, d, c });
            tris.push_back({ a, c, b });
        }
    return tris;
}

// Sphere -> torus: the solid's topology read off the grid, one genus event,
// and the real meshes at both ends.
static void TestPiecewise() {
    std::vector<Triangle> a = Sphere(glm::vec3(0.0f), 0.8f);
    std::vector<Triangle> b = Torus(0.65f, 0.3f);
    std::vector<glm::vec3> sa = MeshSampler::Sample(a, 4000), sb = MeshSampler::Sample(b, 4000);
    Gaussian ga, gb;
    ga.FitFromPoints(sa, false);
    gb.FitFromPoints(sb, false);
    SDFMorph sdf;
    sdf.Build(a, b, OptimalTransport::Compute(ga, gb), 48);

    PiecewiseMorph::Topology t0 = PiecewiseMorph::SolidTopology(sdf, 0.0f);
    PiecewiseMorph::Topology t1 = PiecewiseMorph::SolidTopology(sdf, 1.0f);
    std::printf("Piecewise: solid at t=0 parts %d genus %d, at t=1 parts %d genus %d\n",
                t0.parts, t0.genus, t1.parts, t1.genus);
    CHECK(t0.parts == 1 && t0.genus == 0 && t0.cavities == 0, "sphere read as %d/%d/%d", t0.parts, t0.genus, t0.cavities);
    CHECK(t1.parts == 1 && t1.genus == 1 && t1.cavities == 0, "torus read as %d/%d/%d", t1.parts, t1.genus, t1.cavities);

    IndexedMesh ma, mb;
    MeshTopology::Build(a, ma);
    MeshTopology::Build(b, mb);
    PiecewiseMorph pw;
    PiecewiseMorph::Params params;
    params.keyframes = 8;
    params.fit.stages = 3;
    CHECK(pw.Build(sdf, ma, mb, {}, {}, nullptr, nullptr, params), "build failed");
    std::printf("Piecewise: %d events", (int)pw.events.size());
    for (const auto& e : pw.events) std::printf(" [%s at %.2f]", e.what.c_str(), e.t);
    std::printf("\n");
    CHECK(pw.events.size() == 1 && pw.events[0].what == "genus 0 -> 1", "expected one genus event");

    std::vector<glm::vec3> p, c;
    pw.Evaluate(pw.SegmentAt(0.0f), 0.0f, p, c);
    bool exactA = p == ma.verts;
    pw.Evaluate(pw.SegmentAt(1.0f), 1.0f, p, c);
    bool exactB = p == mb.verts;
    CHECK(exactA && exactB, "ends are not the real meshes (A %d, B %d)", exactA, exactB);
    CHECK(pw.segments.front().indices == ma.indices && pw.segments.back().indices == mb.indices,
          "end segments don't use A's and B's triangles");
}


// ---- Morph Director ----

// Curves: exact ends; Pop pulls back below 0 and overshoots above 1.
static void TestCurves() {
    for (const MorphStyle& st : BuiltinStyles()) {
        for (const MorphCurve* c : { &st.early, &st.late }) {
            CHECK(c->Eval(0.0f) == 0.0f && c->Eval(1.0f) == 1.0f, "%s: curve ends %g %g",
                  st.name.c_str(), c->Eval(0.0f), c->Eval(1.0f));
        }
        // every point is at A at t = 0 and at B at t = 1, whatever its delay
        for (float d : { 0.0f, 0.3f, 1.0f }) {
            float a = VertexTime(0.0f, st.wave, d, st.early, st.late);
            float b = VertexTime(1.0f, st.wave, d, st.early, st.late);
            CHECK(a == 0.0f && b == 1.0f, "%s: delay %g runs %g..%g", st.name.c_str(), d, a, b);
        }
    }
    const MorphStyle& pop = BuiltinStyles()[FindStyle("pop")];
    float lo = 1.0f, hi = 0.0f;
    for (int i = 0; i <= 200; ++i) {
        float y = pop.early.Eval(i / 200.0f);
        lo = std::min(lo, y);
        hi = std::max(hi, y);
    }
    std::printf("Curves: Pop runs %.3f .. %.3f\n", lo, hi);
    CHECK(lo < -0.02f && hi > 1.05f, "Pop should anticipate and overshoot (%g, %g)", lo, hi);
    // linear curve is the identity
    float worst = 0.0f;
    for (int i = 0; i <= 50; ++i) worst = std::max(worst, std::abs(MorphCurve::Linear().Eval(i / 50.0f) - i / 50.0f));
    CHECK(worst < 1e-4f, "linear curve off by %g", worst);
}

// Spearman rank correlation of two equally long lists.
static double RankCorrelation(const std::vector<float>& a, const std::vector<float>& b) {
    auto ranks = [](const std::vector<float>& v) {
        std::vector<int> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int i, int j) { return v[i] < v[j]; });
        std::vector<double> r(v.size());
        for (size_t k = 0; k < idx.size(); ++k) r[idx[k]] = (double)k;
        return r;
    };
    std::vector<double> ra = ranks(a), rb = ranks(b);
    double n = (double)a.size(), d2 = 0.0;
    for (size_t i = 0; i < a.size(); ++i) d2 += (ra[i] - rb[i]) * (ra[i] - rb[i]);
    return 1.0 - 6.0 * d2 / (n * (n * n - 1.0));
}

// Heat wave from a clicked point: the delay grows with the distance along
// the surface (on a sphere, the angle from the seed). The bottom seed of
// the new overload is exactly the old wave.
static void TestWaveFromPoint() {
    std::vector<glm::vec3> pts = MeshSampler::Sample(Sphere(glm::vec3(0.0f), 0.8f), 3000);
    glm::vec3 seed(0.8f, 0.0f, 0.0f);
    float cell = 0.12f;
    std::vector<float> d = HeatWaveDelay(pts, cell, 0.6, 0.02, WaveSeed::Point, seed);
    std::vector<float> angle(pts.size());
    for (size_t i = 0; i < pts.size(); ++i)
        angle[i] = std::acos(std::clamp(glm::dot(glm::normalize(pts[i]), glm::vec3(1, 0, 0)), -1.0f, 1.0f));
    double rho = RankCorrelation(d, angle);
    std::printf("Wave from a point: rank correlation with distance %.3f\n", rho);
    CHECK(rho > 0.8, "delay doesn't follow the distance from the seed (%g)", rho);

    std::vector<float> oldWave = HeatWaveDelay(pts, cell, 0.6, 0.02);
    std::vector<float> bottom  = HeatWaveDelay(pts, cell, 0.6, 0.02, WaveSeed::Bottom, glm::vec3(0.0f));
    CHECK(oldWave == bottom, "bottom seed differs from the original wave");
    std::vector<float> top = HeatWaveDelay(pts, cell, 0.6, 0.02, WaveSeed::Top, glm::vec3(0.0f));
    std::vector<float> negY(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) negY[i] = -pts[i].y;
    CHECK(RankCorrelation(top, negY) > 0.8, "top seed doesn't start at the top");
}

// Springs: overshoot and settle with jiggle; exact without.
static void TestSprings() {
    ScalarSprings s;
    s.Reset({ 0.0f });
    float peak = 0.0f;
    for (int f = 0; f < 240; ++f) peak = std::max(peak, s.Step({ 1.0f }, 1.0f / 60.0f, 0.8f)[0]);
    std::printf("Springs: jiggle 0.8 peaks at %.3f, ends at %.5f\n", peak, s.Values()[0]);
    CHECK(peak > 1.05f, "no overshoot with jiggle (%g)", peak);
    CHECK(std::abs(s.Values()[0] - 1.0f) < 1e-3f && s.Settled(), "did not settle (%g)", s.Values()[0]);
    s.Reset({ 0.0f });
    CHECK(s.Step({ 0.7f }, 1.0f / 60.0f, 0.0f)[0] == 0.7f, "jiggle 0 should follow exactly");
}

// Pins: the pinned vertices land exactly on their targets, and the fit
// itself gets them much closer than an unpinned fit.
static void TestPins() {
    IndexedMesh tmpl;
    MeshTopology::Build(Sphere(glm::vec3(0.0f), 0.8f), tmpl);
    // target: the sphere stretched into an ellipsoid
    std::vector<Triangle> target = Sphere(glm::vec3(0.0f), 0.8f);
    glm::vec3 scale(1.3f, 0.8f, 1.0f);
    for (Triangle& t : target) { t.v0 *= scale; t.v1 *= scale; t.v2 *= scale; }

    // three vertices each pinned to the stretched position of a vertex
    // rotated 40 degrees around y: the fit has to twist to honour them
    glm::mat3 R = glm::mat3_cast(glm::angleAxis(0.7f, glm::vec3(0, 1, 0)));
    FitParams params;
    const int n = (int)tmpl.verts.size();
    for (int v : { 0, n / 3, 2 * n / 3 })
        params.pins.push_back({ v, scale * (R * tmpl.verts[v]) });

    FitParams none;
    FitStats sNone, sPins;
    std::vector<glm::vec3> free = MeshFitter::Fit(tmpl, tmpl.verts, target, none, &sNone);
    std::vector<glm::vec3> pinned = MeshFitter::Fit(tmpl, tmpl.verts, target, params, &sPins);
    double freeErr = 0.0;
    bool exact = true;
    for (const auto& p : params.pins) {
        freeErr += glm::length(free[p.first] - p.second) / params.pins.size();
        exact &= pinned[p.first] == p.second;
    }
    std::printf("Pins: pinned fit off by %.4f before the lock, unpinned by %.4f\n", sPins.pinError, freeErr);
    CHECK(exact, "pinned vertices are not on their targets");
    CHECK(sPins.pinError < 0.3 * freeErr, "pins barely steer the fit (%g vs %g)", sPins.pinError, freeErr);
    CHECK(sPins.meanDist < 0.05f, "pinned fit left the surface (%g)", sPins.meanDist);
}

// Timeline: segment boundaries and durations.
static void TestTimeline() {
    MorphTimeline tl;
    tl.keys = { 1, 2, 3 };
    tl.SyncSegments();
    tl.segs[0].duration = 1.0f;
    tl.segs[1].duration = 3.0f;
    CHECK(tl.Total() == 4.0f, "total %g", tl.Total());
    auto at = [&](float p) { return tl.Locate(p); };
    CHECK(at(0.0f).seg == 0 && at(0.0f).local == 0.0f, "start");
    CHECK(at(0.125f).seg == 0 && std::abs(at(0.125f).local - 0.5f) < 1e-5f, "inside the first step");
    CHECK(at(0.25f).seg == 1 && at(0.25f).local == 0.0f, "the middle key starts the second step");
    CHECK(at(1.0f).seg == 1 && at(1.0f).local == 1.0f, "end");
    CHECK(std::abs(tl.SegmentStart(1) - 0.25f) < 1e-6f && tl.SegmentEnd(1) == 1.0f, "segment bounds");

    // retiming a key moves time between its two steps, never below the minimum
    tl.Retime(1, 2.5f);
    CHECK(std::abs(tl.segs[0].duration - 2.5f) < 1e-5f && std::abs(tl.segs[1].duration - 1.5f) < 1e-5f &&
          std::abs(tl.Total() - 4.0f) < 1e-5f, "retime to 2.5 s: %g + %g", tl.segs[0].duration, tl.segs[1].duration);
    tl.Retime(1, 100.0f);
    CHECK(std::abs(tl.segs[1].duration - MorphTimeline::kMinStep) < 1e-5f, "retime keeps the minimum step");
    tl.Retime(0, 1.0f);   // the ends don't move
    CHECK(std::abs(tl.Total() - 4.0f) < 1e-5f && tl.KeyTime(0) == 0.0f, "the start key moved");

    // adding and removing keys keeps one step per pair; removing an inner key merges its steps
    tl.segs[0].pins.push_back({});
    tl.AddKey(4);
    CHECK(tl.keys.size() == 4 && tl.segs.size() == 3 && tl.segs[2].pins.empty(), "add key");
    float before = tl.Total();
    tl.RemoveKey(1);
    CHECK(tl.keys.size() == 3 && tl.segs.size() == 2 && tl.keys[1] == 3, "remove key");
    CHECK(std::abs(tl.Total() - before) < 1e-5f, "removing an inner key changed the length (%g vs %g)", tl.Total(), before);
    tl.SetKey(0, 7);
    CHECK(tl.keys[0] == 7 && tl.segs[0].pins.empty(), "set key clears the step's pins");
}

// Picking: the frontmost point near the ray.
static void TestPicking() {
    std::vector<glm::vec3> pts = { { 0, 0, -1 }, { 0, 0, 1 }, { 0.5f, 0, 0 } };
    Ray ray{ glm::vec3(0, 0, 5), glm::vec3(0, 0, -1) };
    CHECK(PickPoint(pts, ray, 0.1f) == 1, "picked %d", PickPoint(pts, ray, 0.1f));
    Ray miss{ glm::vec3(3, 0, 5), glm::vec3(0, 0, -1) };
    CHECK(PickPoint(pts, miss, 0.1f) == -1, "should miss");
    // the screen centre looks straight at the target
    glm::mat4 view = glm::lookAt(glm::vec3(0, 0, 5), glm::vec3(0), glm::vec3(0, 1, 0));
    glm::mat4 proj = glm::perspective(0.8f, 1.5f, 0.1f, 100.0f);
    Ray c = ScreenRay(300.0f, 200.0f, 600.0f, 400.0f, view, proj);
    CHECK(glm::length(c.dir - glm::vec3(0, 0, -1)) < 1e-4f, "centre ray off");
}

int main() {
    TestShRotation();
    TestPlyShLayout();
    TestMixtureWarp();
    TestPiecewise();
    TestCurves();
    TestWaveFromPoint();
    TestSprings();
    TestPins();
    TestTimeline();
    TestPicking();
    if (g_failed) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
