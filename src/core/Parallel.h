#pragma once
#include <algorithm>
#include <thread>
#include <vector>

// Portable stand-ins for std::execution::par, which libc++ (macOS) lacks and
// libstdc++ (Linux) only runs in parallel when linked against TBB.

// Run f(i) for i in [0, n) on all hardware threads, in contiguous chunks.
// minPerThread: don't start a thread for fewer items than this (1 for a few
// big jobs, like one linear solve per column).
template <typename F>
void ParallelFor(int n, F f, int minPerThread = 256) {
    int threads = (int)std::max(1u, std::thread::hardware_concurrency());
    threads = std::min(threads, std::max(1, n / std::max(1, minPerThread)));
    if (threads <= 1) {
        for (int i = 0; i < n; ++i) f(i);
        return;
    }
    std::vector<std::thread> pool;
    int chunk = (n + threads - 1) / threads;
    for (int t = 0; t < threads; ++t) {
        int begin = t * chunk, end = std::min(n, begin + chunk);
        if (begin >= end) break;
        pool.emplace_back([=, &f]() {
            for (int i = begin; i < end; ++i) f(i);
        });
    }
    for (auto& th : pool) th.join();
}

// Sum of f(i) for i in [0, n), in parallel. Each chunk sums in order, so the
// result only depends on n and the thread count.
template <typename F>
double ParallelSum(int n, F f) {
    int threads = (int)std::max(1u, std::thread::hardware_concurrency());
    threads = std::min(threads, std::max(1, n / 256));
    std::vector<double> partial(threads, 0.0);
    int chunk = (n + threads - 1) / threads;
    ParallelFor(threads, [&](int t) {
        int begin = t * chunk, end = std::min(n, begin + chunk);
        double s = 0.0;
        for (int i = begin; i < end; ++i) s += f(i);
        partial[t] = s;
    }, 1);
    double total = 0.0;
    for (double s : partial) total += s;
    return total;
}
