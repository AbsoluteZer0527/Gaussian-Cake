#pragma once
#include "core.h"

// Measures how long a block of GPU work takes, without stalling.
//
// OpenGL calls only queue work, so CPU timers can't see GPU time. A
// GL_TIME_ELAPSED query asks the GPU to record it, but reading the answer
// right away would make the CPU wait for the GPU to catch up. So the timer
// keeps a small ring of queries and reads one from a few frames ago, which is
// long finished.
class GpuTimer {
public:
    ~GpuTimer() { if (created) glDeleteQueries(kRing, queries); }

    void Begin() {
        if (!created) {
            glGenQueries(kRing, queries);
            created = true;
        }
        Collect();
        glBeginQuery(GL_TIME_ELAPSED, queries[next]);
    }

    void End() {
        glEndQuery(GL_TIME_ELAPSED);
        pending[next] = true;
        next = (next + 1) % kRing;
    }

    // Smoothed GPU time of the block, in ms (-1 until the first result).
    double Ms() const { return ms; }

    // The work changed (e.g. a new splat count): forget the average, and drop
    // the queries still in flight, which timed the old work.
    void Reset() {
        for (int k = 0; k < kRing; ++k)
            if (pending[k]) stale[k] = true;
        ms = -1.0;
    }

private:
    static const int kRing = 4;

    // Read every finished query (oldest first) into the running average.
    void Collect() {
        for (int k = 0; k < kRing; ++k) {
            int q = (next + k) % kRing;
            if (!pending[q]) continue;
            GLint ready = 0;
            glGetQueryObjectiv(queries[q], GL_QUERY_RESULT_AVAILABLE, &ready);
            if (!ready) continue;
            GLuint64 ns = 0;
            glGetQueryObjectui64v(queries[q], GL_QUERY_RESULT, &ns);
            pending[q] = false;
            if (stale[q]) { stale[q] = false; continue; }
            double sample = (double)ns * 1e-6;
            ms = (ms < 0.0) ? sample : 0.9 * ms + 0.1 * sample;
        }
        // the slot about to be reused must be free
        if (pending[next]) {
            GLuint64 ns = 0;
            glGetQueryObjectui64v(queries[next], GL_QUERY_RESULT, &ns);
            pending[next] = false;
            stale[next] = false;
        }
    }

    GLuint queries[kRing] = {};
    bool   pending[kRing] = {};
    bool   stale[kRing] = {};     // in flight across a Reset: the result is thrown away
    bool   created = false;
    int    next = 0;
    double ms = -1.0;
};
