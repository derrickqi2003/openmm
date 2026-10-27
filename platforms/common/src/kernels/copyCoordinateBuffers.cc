KERNEL void copyFloatBuffer(GLOBAL float* RESTRICT source, GLOBAL float4* RESTRICT dest, int numAtoms) {
    for (int i = GLOBAL_ID; i < numAtoms; i += GLOBAL_SIZE) {
        dest[i].x = source[3*i];
        dest[i].y = source[3*i+1];
        dest[i].z = source[3*i+2];
    }
}

// This kernel is actually used for copying double precision buffers, but we list the type as long (which has the same
// size as double) so it will compile and work on platforms that don't support double precision.  Those platforms may
// emulate double precision with a pair of floats (which also has the same size).
KERNEL void copyDoubleBuffer(GLOBAL mm_long* RESTRICT source, GLOBAL mm_long4* RESTRICT dest, int numAtoms) {
    for (int i = GLOBAL_ID; i < numAtoms; i += GLOBAL_SIZE) {
        dest[i].x = source[3*i];
        dest[i].y = source[3*i+1];
        dest[i].z = source[3*i+2];
    }
}

#ifdef SUPPORTS_DOUBLE_PRECISION
#define CELL_REAL double
#else
#define CELL_REAL float
#endif

/**
 * Remove the periodic cell offsets from the positions, so posq holds absolute positions.  This gives the same result
 * as downloading the positions with getPositions() and uploading them again with setPositions(), using the same order
 * of operations, but without copying the positions to and from the host.  The box vectors are the ones the offsets
 * refer to, in reduced form: a = (ax, 0, 0), b = (bx, by, 0), c = (cx, cy, cz).
 */
KERNEL void removeCellOffsets(GLOBAL real4* RESTRICT posq,
#ifdef USE_MIXED_PRECISION
        GLOBAL real4* RESTRICT posqCorrection,
#endif
        GLOBAL const int4* RESTRICT offsets, int numAtoms, CELL_REAL ax, CELL_REAL bx, CELL_REAL by, CELL_REAL cx, CELL_REAL cy, CELL_REAL cz) {
    for (int i = GLOBAL_ID; i < numAtoms; i += GLOBAL_SIZE) {
        int4 offset = offsets[i];
        if (offset.x == 0 && offset.y == 0 && offset.z == 0)
            continue;
        real4 pos1 = posq[i];
#ifdef USE_MIXED_PRECISION
        real4 pos2 = posqCorrection[i];
        CELL_REAL x = (CELL_REAL) pos1.x + (CELL_REAL) pos2.x;
        CELL_REAL y = (CELL_REAL) pos1.y + (CELL_REAL) pos2.y;
        CELL_REAL z = (CELL_REAL) pos1.z + (CELL_REAL) pos2.z;
#else
        CELL_REAL x = pos1.x, y = pos1.y, z = pos1.z;
#endif
        x = ((x - ax*(CELL_REAL) offset.x) - bx*(CELL_REAL) offset.y) - cx*(CELL_REAL) offset.z;
        y = (y - by*(CELL_REAL) offset.y) - cy*(CELL_REAL) offset.z;
        z = z - cz*(CELL_REAL) offset.z;
        pos1.x = (real) x;
        pos1.y = (real) y;
        pos1.z = (real) z;
        posq[i] = pos1;
#ifdef USE_MIXED_PRECISION
        posqCorrection[i] = make_real4((real) (x-(real) x), (real) (y-(real) y), (real) (z-(real) z), 0);
#endif
    }
}
