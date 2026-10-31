/**
 * Compute the reciprocal space correction for excluded pairs, with the forces multiplied by forceScale.  NonbondedForce
 * uses this in place of its bonded interaction once multiple time stepping has required the correction to be scaled
 * together with reciprocal space.
 */
KERNEL void computeExclusionCorrection(GLOBAL mm_ulong* RESTRICT forceBuffer, GLOBAL mixed* RESTRICT energyBuffer, GLOBAL const real4* RESTRICT posq,
        GLOBAL const int2* RESTRICT exclusionAtoms, GLOBAL const float4* RESTRICT exclusionParamArray, int numExclusions,
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real forceScale, int includeEnergy) {
    mixed energy = 0;
    for (int index = GLOBAL_ID; index < numExclusions; index += GLOBAL_SIZE) {
        int2 atoms = exclusionAtoms[index];
        real4 pos1 = posq[atoms.x];
        real4 pos2 = posq[atoms.y];
        COMPUTE_EXCLUSION
        ATOMIC_ADD(&forceBuffer[atoms.x], (mm_ulong) realToFixedPoint(forceScale*force1.x));
        ATOMIC_ADD(&forceBuffer[atoms.x+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(forceScale*force1.y));
        ATOMIC_ADD(&forceBuffer[atoms.x+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(forceScale*force1.z));
        ATOMIC_ADD(&forceBuffer[atoms.y], (mm_ulong) realToFixedPoint(forceScale*force2.x));
        ATOMIC_ADD(&forceBuffer[atoms.y+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(forceScale*force2.y));
        ATOMIC_ADD(&forceBuffer[atoms.y+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(forceScale*force2.z));
    }
    if (includeEnergy)
        energyBuffer[GLOBAL_ID] += energy;
}
