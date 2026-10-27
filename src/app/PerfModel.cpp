#include "PerfModel.h"

#include <cmath>

// Each measurement implies a cost; blend it in so one noisy sample can't
// swing the model (an exponential average, like SmoothMs in Window.cpp).
void PerfModel::ObserveBake(int n, double ms) {
    if (n < 2 || ms <= 0.0) return;
    double measured = ms / ((double)n * std::log2((double)n));
    bakeCost = 0.7 * bakeCost + 0.3 * measured;
}

void PerfModel::ObserveFrame(int n, double ms) {
    if (n < 2 || ms <= 0.0) return;   // the GPU timers report -1 before their first result
    double measured = ms / (double)n;
    frameCost = 0.7 * frameCost + 0.3 * measured;
}

double PerfModel::PredictBakeMs(int n) const {
    return bakeCost * (double)n * std::log2((double)n);
}

double PerfModel::PredictFrameMs(int n) const {
    return frameCost * (double)n;
}

// Both predictions grow with n, so "fits" holds up to some n and fails after:
// binary search for that n.
int PerfModel::MaxCount(int lo, int hi, double bakeBudgetMs, double frameBudgetMs) const {
    auto fits = [&](int n) {
        return PredictBakeMs(n) <= bakeBudgetMs && PredictFrameMs(n) <= frameBudgetMs;
    };
    if (!fits(lo)) return lo;
    while (lo < hi) {
        int mid = lo + (hi - lo + 1) / 2;   // round up, so lo = mid always makes progress
        if (fits(mid)) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}
