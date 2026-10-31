enum {VelScale, NoiseScale, MaxParams};

/**
 * The velocity step for one atom.
 */
inline DEVICE mixed4 langevinMiddleVelocityStep(mixed4 velocity, int index, int paddedNumAtoms, GLOBAL const mm_long* RESTRICT force, mixed fscale) {
    if (velocity.w != 0.0) {
        velocity.x += fscale*velocity.w*force[index];
        velocity.y += fscale*velocity.w*force[index+paddedNumAtoms];
        velocity.z += fscale*velocity.w*force[index+paddedNumAtoms*2];
    }
    return velocity;
}

/**
 * The position half step, the interaction with the heat bath, and the second position half step for one atom.  This
 * updates the velocity and returns the change in position.
 */
inline DEVICE mixed4 langevinMiddleHeatBathStep(PRIVATE mixed4* velocityPtr, float4 random, mixed vscale, mixed noisescale, mixed halfdt) {
    mixed4 velocity = *velocityPtr;
    mixed4 delta = make_mixed4(halfdt*velocity.x, halfdt*velocity.y, halfdt*velocity.z, 0);
    mixed sqrtInvMass = SQRT(velocity.w);
    velocity.x = vscale*velocity.x + noisescale*sqrtInvMass*random.x;
    velocity.y = vscale*velocity.y + noisescale*sqrtInvMass*random.y;
    velocity.z = vscale*velocity.z + noisescale*sqrtInvMass*random.z;
    delta += make_mixed4(halfdt*velocity.x, halfdt*velocity.y, halfdt*velocity.z, 0);
    *velocityPtr = velocity;
    return delta;
}

/**
 * Apply the constraint forces to the velocity of one atom, and move it to its constrained position.
 */
inline DEVICE void langevinMiddleFinalStep(PRIVATE mixed4* velocity, PRIVATE mixed4* pos, mixed4 delta, mixed4 oldDelta, mixed invDt) {
    (*velocity).x += (delta.x-oldDelta.x)*invDt;
    (*velocity).y += (delta.y-oldDelta.y)*invDt;
    (*velocity).z += (delta.z-oldDelta.z)*invDt;
    (*pos).x += delta.x;
    (*pos).y += delta.y;
    (*pos).z += delta.z;
}

/**
 * Perform the first part of integration: velocity step.
 */

KERNEL void integrateLangevinMiddlePart1(int numAtoms, int paddedNumAtoms, GLOBAL mixed4* RESTRICT velm, GLOBAL const mm_long* RESTRICT force,
        GLOBAL const mixed2* RESTRICT dt) {
    mixed fscale = dt[0].y/(mixed) 0x100000000;
    for (int index = GLOBAL_ID; index < numAtoms; index += GLOBAL_SIZE) {
        mixed4 velocity = velm[index];
        if (velocity.w != 0.0)
            velm[index] = langevinMiddleVelocityStep(velocity, index, paddedNumAtoms, force, fscale);
    }
}

/**
 * Perform the second part of integration: position half step, then interact with heat bath,
 * then another position half step.
 */

KERNEL void integrateLangevinMiddlePart2(int numAtoms, GLOBAL mixed4* RESTRICT velm, GLOBAL mixed4* RESTRICT posDelta,
        GLOBAL mixed4* RESTRICT oldDelta, GLOBAL const mixed* RESTRICT paramBuffer, GLOBAL const mixed2* RESTRICT dt, GLOBAL const float4* RESTRICT random, unsigned int startRandomIndex
        ) {
    mixed vscale = paramBuffer[VelScale];
    mixed noisescale = paramBuffer[NoiseScale];
    mixed halfdt = 0.5f*dt[0].y;
    int index = GLOBAL_ID;
    unsigned int randomIndex = startRandomIndex+index;
    while (index < numAtoms) {
        mixed4 velocity = velm[index];
        if (velocity.w != 0.0) {
            mixed4 delta = langevinMiddleHeatBathStep(&velocity, random[randomIndex], vscale, noisescale, halfdt);
            velm[index] = velocity;
            posDelta[index] = delta;
            oldDelta[index] = delta;
        }
        randomIndex += GLOBAL_SIZE;
        index += GLOBAL_SIZE;
    }
}

/**
 * Perform the third part of integration: apply constraint forces to velocities, then record
 * the constrained positions.
 */

KERNEL void integrateLangevinMiddlePart3(int numAtoms, GLOBAL real4* RESTRICT posq, GLOBAL mixed4* RESTRICT velm,
         GLOBAL mixed4* RESTRICT posDelta, GLOBAL mixed4* RESTRICT oldDelta, GLOBAL const mixed2* RESTRICT dt
#ifdef USE_MIXED_PRECISION
        , GLOBAL real4* RESTRICT posqCorrection
#endif
        ) {
#ifndef USE_MIXED_PRECISION
    GLOBAL real4* posqCorrection = 0;
#endif
    mixed invDt = 1/dt[0].y;
    for (int index = GLOBAL_ID; index < numAtoms; index += GLOBAL_SIZE) {
        mixed4 velocity = velm[index];
        if (velocity.w != 0.0) {
            mixed4 pos = loadPos(posq, posqCorrection, index);
            langevinMiddleFinalStep(&velocity, &pos, posDelta[index], oldDelta[index], invDt);
            velm[index] = velocity;
            storePos(posq, posqCorrection, index, pos);
        }
    }
}

/**
 * Perform a whole step when every constraint belongs to a SETTLE or SHAKE cluster.  Neither the constraints nor any
 * part of the step couple atoms in different clusters, so each thread can take one cluster, or one unconstrained
 * atom, through all three parts and both constraint steps without the separate kernels and their global memory
 * traffic.  The arithmetic is the same as in those kernels.
 */
KERNEL void integrateLangevinMiddleClusters(int numSettle, int numShake, int numFree, int paddedNumAtoms, mixed tol,
        GLOBAL real4* RESTRICT posq, GLOBAL mixed4* RESTRICT velm, GLOBAL const mm_long* RESTRICT force,
        GLOBAL const mixed2* RESTRICT dt, GLOBAL const mixed* RESTRICT paramBuffer, GLOBAL const float4* RESTRICT random,
        unsigned int startRandomIndex, GLOBAL const int4* RESTRICT settleAtoms, GLOBAL const float2* RESTRICT settleParams,
        GLOBAL const int4* RESTRICT shakeAtoms, GLOBAL const float4* RESTRICT shakeParams, GLOBAL const int* RESTRICT freeAtoms
#ifdef USE_MIXED_PRECISION
        , GLOBAL real4* RESTRICT posqCorrection
#endif
        ) {
#ifndef USE_MIXED_PRECISION
    GLOBAL real4* posqCorrection = 0;
#endif
    mixed fscale = dt[0].y/(mixed) 0x100000000;
    mixed vscale = paramBuffer[VelScale];
    mixed noisescale = paramBuffer[NoiseScale];
    mixed halfdt = 0.5f*dt[0].y;
    mixed invDt = 1/dt[0].y;
    for (int index = GLOBAL_ID; index < numSettle+numShake+numFree; index += GLOBAL_SIZE) {
        if (index < numSettle) {
            // A rigid water.

            int4 atoms = settleAtoms[index];
            mixed4 pos0 = loadPos(posq, posqCorrection, atoms.x);
            mixed4 pos1 = loadPos(posq, posqCorrection, atoms.y);
            mixed4 pos2 = loadPos(posq, posqCorrection, atoms.z);
            mixed4 v0 = langevinMiddleVelocityStep(velm[atoms.x], atoms.x, paddedNumAtoms, force, fscale);
            mixed4 v1 = langevinMiddleVelocityStep(velm[atoms.y], atoms.y, paddedNumAtoms, force, fscale);
            mixed4 v2 = langevinMiddleVelocityStep(velm[atoms.z], atoms.z, paddedNumAtoms, force, fscale);
            settleClusterVelocities(pos0, pos1, pos2, &v0, &v1, &v2);
            mixed4 delta0 = langevinMiddleHeatBathStep(&v0, random[startRandomIndex+atoms.x], vscale, noisescale, halfdt);
            mixed4 delta1 = langevinMiddleHeatBathStep(&v1, random[startRandomIndex+atoms.y], vscale, noisescale, halfdt);
            mixed4 delta2 = langevinMiddleHeatBathStep(&v2, random[startRandomIndex+atoms.z], vscale, noisescale, halfdt);
            mixed4 oldDelta0 = delta0, oldDelta1 = delta1, oldDelta2 = delta2;
            settleClusterPositions(settleParams[index], pos0, pos1, pos2, 1/v0.w, 1/v1.w, 1/v2.w, &delta0, &delta1, &delta2);
            langevinMiddleFinalStep(&v0, &pos0, delta0, oldDelta0, invDt);
            langevinMiddleFinalStep(&v1, &pos1, delta1, oldDelta1, invDt);
            langevinMiddleFinalStep(&v2, &pos2, delta2, oldDelta2, invDt);
            velm[atoms.x] = v0;
            velm[atoms.y] = v1;
            velm[atoms.z] = v2;
            storePos(posq, posqCorrection, atoms.x, pos0);
            storePos(posq, posqCorrection, atoms.y, pos1);
            storePos(posq, posqCorrection, atoms.z, pos2);
        }
        else if (index < numSettle+numShake) {
            // A central atom with up to three atoms constrained to it.

            int4 atoms = shakeAtoms[index-numSettle];
            float4 params = shakeParams[index-numSettle];
            mixed4 pos0 = loadPos(posq, posqCorrection, atoms.x);
            mixed4 pos1 = loadPos(posq, posqCorrection, atoms.y);
            mixed4 pos2 = make_mixed4(0), pos3 = make_mixed4(0);
            mixed4 v0 = langevinMiddleVelocityStep(velm[atoms.x], atoms.x, paddedNumAtoms, force, fscale);
            mixed4 v1 = langevinMiddleVelocityStep(velm[atoms.y], atoms.y, paddedNumAtoms, force, fscale);
            mixed4 v2 = make_mixed4(0), v3 = make_mixed4(0);
            if (atoms.z != -1) {
                pos2 = loadPos(posq, posqCorrection, atoms.z);
                v2 = langevinMiddleVelocityStep(velm[atoms.z], atoms.z, paddedNumAtoms, force, fscale);
            }
            if (atoms.w != -1) {
                pos3 = loadPos(posq, posqCorrection, atoms.w);
                v3 = langevinMiddleVelocityStep(velm[atoms.w], atoms.w, paddedNumAtoms, force, fscale);
            }
            shakeClusterVelocities(atoms, params, tol, pos0, pos1, pos2, pos3, &v0, &v1, &v2, &v3);
            mixed4 delta0 = langevinMiddleHeatBathStep(&v0, random[startRandomIndex+atoms.x], vscale, noisescale, halfdt);
            mixed4 delta1 = langevinMiddleHeatBathStep(&v1, random[startRandomIndex+atoms.y], vscale, noisescale, halfdt);
            mixed4 delta2 = make_mixed4(0), delta3 = make_mixed4(0);
            if (atoms.z != -1)
                delta2 = langevinMiddleHeatBathStep(&v2, random[startRandomIndex+atoms.z], vscale, noisescale, halfdt);
            if (atoms.w != -1)
                delta3 = langevinMiddleHeatBathStep(&v3, random[startRandomIndex+atoms.w], vscale, noisescale, halfdt);
            mixed4 oldDelta0 = delta0, oldDelta1 = delta1, oldDelta2 = delta2, oldDelta3 = delta3;
            shakeClusterPositions(atoms, params, tol, pos0, pos1, pos2, pos3, &delta0, &delta1, &delta2, &delta3);
            langevinMiddleFinalStep(&v0, &pos0, delta0, oldDelta0, invDt);
            langevinMiddleFinalStep(&v1, &pos1, delta1, oldDelta1, invDt);
            velm[atoms.x] = v0;
            velm[atoms.y] = v1;
            storePos(posq, posqCorrection, atoms.x, pos0);
            storePos(posq, posqCorrection, atoms.y, pos1);
            if (atoms.z != -1) {
                langevinMiddleFinalStep(&v2, &pos2, delta2, oldDelta2, invDt);
                velm[atoms.z] = v2;
                storePos(posq, posqCorrection, atoms.z, pos2);
            }
            if (atoms.w != -1) {
                langevinMiddleFinalStep(&v3, &pos3, delta3, oldDelta3, invDt);
                velm[atoms.w] = v3;
                storePos(posq, posqCorrection, atoms.w, pos3);
            }
        }
        else {
            // An atom without constraints.

            int atom = freeAtoms[index-numSettle-numShake];
            mixed4 velocity = velm[atom];
            if (velocity.w != 0.0) {
                velocity = langevinMiddleVelocityStep(velocity, atom, paddedNumAtoms, force, fscale);
                mixed4 delta = langevinMiddleHeatBathStep(&velocity, random[startRandomIndex+atom], vscale, noisescale, halfdt);
                mixed4 pos = loadPos(posq, posqCorrection, atom);
                langevinMiddleFinalStep(&velocity, &pos, delta, delta, invDt);
                velm[atom] = velocity;
                storePos(posq, posqCorrection, atom, pos);
            }
        }
    }
}

/**
 * Select the step size to use for the next step.
 */

KERNEL void selectLangevinStepSize(int numAtoms, int paddedNumAtoms, mixed maxStepSize, mixed errorTol, mixed friction, mixed kT, GLOBAL mixed2* RESTRICT dt,
        GLOBAL const mixed4* RESTRICT velm, GLOBAL const mm_long* RESTRICT force, GLOBAL mixed* RESTRICT paramBuffer) {
    // Calculate the error.

    LOCAL mixed error[256];
    LOCAL mixed params[MaxParams];
    mixed err = 0;
    const mixed scale = RECIP((mixed) 0x100000000);
    for (int index = LOCAL_ID; index < numAtoms; index += LOCAL_SIZE) {
        mixed3 f = make_mixed3(scale*force[index], scale*force[index+paddedNumAtoms], scale*force[index+paddedNumAtoms*2]);
        mixed invMass = velm[index].w;
        err += (f.x*f.x + f.y*f.y + f.z*f.z)*invMass*invMass;
    }
    error[LOCAL_ID] = err;
    SYNC_THREADS;

    // Sum the errors from all threads.

    for (unsigned int offset = 1; offset < LOCAL_SIZE; offset *= 2) {
        if (LOCAL_ID+offset < LOCAL_SIZE && (LOCAL_ID&(2*offset-1)) == 0)
            error[LOCAL_ID] += error[LOCAL_ID+offset];
        SYNC_THREADS;
    }
    if (GLOBAL_ID == 0) {
        // Select the new step size.

        mixed totalError = SQRT(error[0]/(numAtoms*3));
        mixed newStepSize = SQRT(errorTol/totalError);
        mixed oldStepSize = dt[0].y;
        if (oldStepSize > 0.0f)
            newStepSize = min(newStepSize, oldStepSize*2.0f); // For safety, limit how quickly dt can increase.
        if (newStepSize > oldStepSize && newStepSize < 1.1f*oldStepSize)
            newStepSize = oldStepSize; // Keeping dt constant between steps improves the behavior of the integrator.
        if (newStepSize > maxStepSize)
            newStepSize = maxStepSize;
        dt[0].y = newStepSize;

        // Recalculate the integration parameters.

        mixed vscale = exp(-newStepSize*friction);
        mixed noisescale = sqrt(kT*(1-vscale*vscale));
        params[VelScale] = vscale;
        params[NoiseScale] = noisescale;
    }
    SYNC_THREADS;
    if (LOCAL_ID < MaxParams)
        paramBuffer[LOCAL_ID] = params[LOCAL_ID];
}
