#include "SpringDamper.h"

SpringDamper::SpringDamper() {
    P1 = P2 = nullptr;
    restLength = 1.0f;
    springConstant = 100.0f;
    dampingConstant = 1.0f;
}

SpringDamper::SpringDamper(Particle* p1, Particle* p2, float ks, float kd) {
    P1 = p1;
    P2 = p2;
    
    springConstant = 100.0f;
    dampingConstant = 1.0f;

    //distance between two particles
    restLength = glm::distance(P1->position, P2->position);
}

void SpringDamper::ComputeForce(){
    glm::vec3 eStar = P2->position - P1->position;
    float l = glm::length(eStar);

    //fix zero length issue.
    if(l < 0.0001f) return;

    glm::vec3 e = eStar / l;

    float vClose = glm::dot(P1->velocity - P2->velocity, e);
    float f = -springConstant * (restLength - l) - dampingConstant * vClose;

    P1 -> ApplyForce(f * e);
    P2 -> ApplyForce(-f *e);
}

