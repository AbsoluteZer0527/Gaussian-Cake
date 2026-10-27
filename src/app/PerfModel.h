#pragma once

// Learns how fast this machine runs the splat morph and predicts the largest
// splat count that fits a time budget. Two simple cost models:
//   bake  ms ~ bakeCost  * n * log2(n)   (matching and kNN are n log n)
//   frame ms ~ frameCost * n             (GPU prep + sort + draw)
// Each observation nudges its cost toward the measured value.
class PerfModel {
public:
    // A finished bake of n splats took ms (without loading the files).
    void ObserveBake(int n, double ms);
    // One GPU frame of n splats took ms.
    void ObserveFrame(int n, double ms);

    double PredictBakeMs(int n) const;
    double PredictFrameMs(int n) const;

    // Largest n in [lo, hi] whose predicted bake and frame both fit their
    // budgets; lo if even lo doesn't fit.
    int MaxCount(int lo, int hi, double bakeBudgetMs, double frameBudgetMs) const;

private:
    // Seeds, until the first observations come in: a 20k bake measured 546 ms;
    // the GPU path drew 1.5M splats well under 16 ms.
    double bakeCost  = 546.0 / (20000.0 * 14.3);   // log2(20000) is about 14.3
    double frameCost = 1e-5;
};
