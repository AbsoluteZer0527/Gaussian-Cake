#pragma once
#include "core.h"
#include "Particle.h"

class SpringDamper {
public:
    Particle *P1, *P2;

    // The three tunable constants
    float restLength;
    float springConstant;
    float dampingConstant;

    //one or more initialization functions
    SpringDamper();
    SpringDamper(Particle* p1, Particle* p2, float ks, float kd);

    void ComputeForce();
};
