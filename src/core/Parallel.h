#pragma once
#include <algorithm>
#include <thread>
#include <vector>

// Run f(i) for i in [0, n) on all hardware threads, in contiguous chunks.
template <typename F>
void ParallelFor(int n, F f) {
    int threads = (int)std::max(1u, std::thread::hardware_concurrency());
    threads = std::min(threads, std::max(1, n / 256));
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
