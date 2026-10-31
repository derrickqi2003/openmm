#define GROUP_SIZE 256
#define BUFFER_SIZE 256

// The largest distance between an atom and a neighbor the outer list could include.  With anchors, the cutoff is
// extended by the margins of the two atoms, and blocks' bounding boxes already include their own atoms' margins.
#ifdef USE_ANCHORS
#define BUILD_REACH (PADDED_CUTOFF+MAX_ANCHOR_MARGIN)
#else
#define BUILD_REACH PADDED_CUTOFF
#endif

/**
 * To use half precision, we're supposed to include cuda_fp16.h.  Unfortunately,
 * it isn't included in the search path automatically, and there's no reliable
 * way to find where it's located on disk.  Instead we provide our own definitions
 * for the few symbols we need.
 */
struct __align__(2) __half {
    unsigned short x;
};
__device__ __half __float2half_ru(const float f) {
    __half h;
    asm("{cvt.rp.f16.f32 %0, %1;}" : "=h"(*reinterpret_cast<unsigned short *>(&h)) : "f"(f));
    return h;
}
__device__ float __half2float(const __half h) {
    float f;
    asm("{cvt.f32.f16 %0, %1;}" : "=f"(f) : "h"(*reinterpret_cast<const unsigned short *>(&h)));
    return f;
}
struct half3 {
    __device__ half3(real3 f) {
        // Round up so we'll err on the side of making the box a little too large.
        // This ensures interactions will never be missed.
        v[0] = __float2half_ru((float) f.x);
        v[1] = __float2half_ru((float) f.y);
        v[2] = __float2half_ru((float) f.z);
    }
    __device__ real3 toReal3() const {
        return make_real3(__half2float(v[0]), __half2float(v[1]), __half2float(v[2]));
    }
private:
    __half v[3];
};

/**
 * Find a bounding box for the atoms in each block.
 */
extern "C" __global__ void findBlockBounds(int numAtoms, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        const real4* __restrict__ posq, real4* __restrict__ blockCenter, real4* __restrict__ blockBoundingBox, int* __restrict__ rebuildNeighborList,
        real2* __restrict__ blockSizeRange) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int base = index*TILE_SIZE;
    real minSize = 1e38, maxSize = 0;
    while (base < numAtoms) {
        real4 pos = posq[base];
#ifdef USE_PERIODIC
        APPLY_PERIODIC_TO_POS(pos)
#endif
        real4 minPos = pos;
        real4 maxPos = pos;
        int last = min(base+TILE_SIZE, numAtoms);
        for (int i = base+1; i < last; i++) {
            pos = posq[i];
#ifdef USE_PERIODIC
            real4 center = 0.5f*(maxPos+minPos);
            APPLY_PERIODIC_TO_POS_WITH_CENTER(pos, center)
#endif
            minPos = make_real4(min(minPos.x,pos.x), min(minPos.y,pos.y), min(minPos.z,pos.z), 0);
            maxPos = make_real4(max(maxPos.x,pos.x), max(maxPos.y,pos.y), max(maxPos.z,pos.z), 0);
        }
        real4 blockSize = 0.5f*(maxPos-minPos);
        real4 center = 0.5f*(maxPos+minPos);
        center.w = 0;
        for (int i = base; i < last; i++) {
            pos = posq[i];
            real4 delta = posq[i]-center;
#ifdef USE_PERIODIC
            APPLY_PERIODIC_TO_DELTA(delta)
#endif
            center.w = max(center.w, delta.x*delta.x+delta.y*delta.y+delta.z*delta.z);
        }
        center.w = sqrt(center.w);
        blockBoundingBox[index] = blockSize;
        blockCenter[index] = center;
        real totalSize = blockSize.x+blockSize.y+blockSize.z;
        minSize = min(minSize, totalSize);
        maxSize = max(maxSize, totalSize);
        index += blockDim.x*gridDim.x;
        base = index*TILE_SIZE;
    }
    
    // Record the range of sizes seen by threads in this block.

    __shared__ real minBuffer[64], maxBuffer[64];
    minBuffer[threadIdx.x] = minSize;
    maxBuffer[threadIdx.x] = maxSize;
    __syncthreads();
    for (int step = 1; step < 64; step *= 2) {
        if (threadIdx.x+step < 64 && threadIdx.x%(2*step) == 0) {
            minBuffer[threadIdx.x] = min(minBuffer[threadIdx.x], minBuffer[threadIdx.x+step]);
            maxBuffer[threadIdx.x] = max(maxBuffer[threadIdx.x], maxBuffer[threadIdx.x+step]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        blockSizeRange[blockIdx.x] = make_real2(minBuffer[0], maxBuffer[0]);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        rebuildNeighborList[0] = 0;
#ifdef USE_PRUNING
        rebuildNeighborList[1] = 0;
#endif
    }
}

/**
 * Find a bounding box for the atoms in each block, like findBlockBounds, with one warp per block instead of one thread.
 * Each lane loads one atom, so the loads happen in parallel, and the lanes then process the atoms in the same order as
 * findBlockBounds, so the boxes are identical.  The size of the box is returned to every lane.
 */
__device__ real findWarpBlockBounds(int block, int numAtoms, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        const real4* __restrict__ posq, real4* __restrict__ blockCenter, real4* __restrict__ blockBoundingBox) {
    const int indexInWarp = threadIdx.x%32;
    const int base = block*TILE_SIZE;
    const int count = min(TILE_SIZE, numAtoms-base);
    const real4 atomPos = posq[base+(indexInWarp < count ? indexInWarp : 0)];
    real4 pos = make_real4(__shfl_sync(0xffffffff, atomPos.x, 0), __shfl_sync(0xffffffff, atomPos.y, 0), __shfl_sync(0xffffffff, atomPos.z, 0), 0);
#ifdef USE_PERIODIC
    APPLY_PERIODIC_TO_POS(pos)
#endif
    real4 minPos = pos;
    real4 maxPos = pos;
    for (int i = 1; i < count; i++) {
        pos = make_real4(__shfl_sync(0xffffffff, atomPos.x, i), __shfl_sync(0xffffffff, atomPos.y, i), __shfl_sync(0xffffffff, atomPos.z, i), 0);
#ifdef USE_PERIODIC
        real4 center = 0.5f*(maxPos+minPos);
        APPLY_PERIODIC_TO_POS_WITH_CENTER(pos, center)
#endif
        minPos = make_real4(min(minPos.x,pos.x), min(minPos.y,pos.y), min(minPos.z,pos.z), 0);
        maxPos = make_real4(max(maxPos.x,pos.x), max(maxPos.y,pos.y), max(maxPos.z,pos.z), 0);
    }
    real4 blockSize = 0.5f*(maxPos-minPos);
    real4 center = 0.5f*(maxPos+minPos);
    center.w = 0;
    real4 delta = atomPos-center;
#ifdef USE_PERIODIC
    APPLY_PERIODIC_TO_DELTA(delta)
#endif
    real dist2 = (indexInWarp < count ? delta.x*delta.x+delta.y*delta.y+delta.z*delta.z : 0);
    for (int i = 16; i > 0; i /= 2)
        dist2 = max(dist2, __shfl_xor_sync(0xffffffff, dist2, i));
    center.w = SQRT(dist2);
    if (indexInWarp == 0) {
        blockBoundingBox[block] = blockSize;
        blockCenter[block] = center;
    }
    return blockSize.x+blockSize.y+blockSize.z;
}

/**
 * This does the same as findBlockBounds, using findWarpBlockBounds.  It must be launched with exactly as many work
 * groups as blockSizeRange has elements, so each one records the range of sizes for its blocks.
 *
 * With CHECK_REBUILD_IN_BOUNDS it also does the check of checkRebuildNeighborList, since it loads every atom's position
 * anyway: each work group records in rebuildFlags whether any of its atoms has moved more than half the padding since
 * the list was built, and computeSortKeys combines them.  That saves launching a kernel on every step.
 */
extern "C" __global__ void findBlockBoundsByWarp(int numAtoms, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        const real4* __restrict__ posq, real4* __restrict__ blockCenter, real4* __restrict__ blockBoundingBox, int* __restrict__ rebuildNeighborList,
        real2* __restrict__ blockSizeRange
#ifdef CHECK_REBUILD_IN_BOUNDS
        , const real4* __restrict__ oldPositions, int* __restrict__ rebuildFlags
#endif
        ) {
    const int totalWarps = blockDim.x*gridDim.x/32;
    real minSize = 1e38, maxSize = 0;
#ifdef CHECK_REBUILD_IN_BOUNDS
    bool rebuild = false;
#endif
    for (int block = (blockIdx.x*blockDim.x+threadIdx.x)/32; block < NUM_BLOCKS; block += totalWarps) {
        real totalSize = findWarpBlockBounds(block, numAtoms, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ, posq, blockCenter, blockBoundingBox);
        minSize = min(minSize, totalSize);
        maxSize = max(maxSize, totalSize);
#ifdef CHECK_REBUILD_IN_BOUNDS
        const int atom = block*TILE_SIZE+threadIdx.x%32;
        if (atom < numAtoms) {
            real4 delta = oldPositions[atom]-posq[atom];
            if (delta.x*delta.x + delta.y*delta.y + delta.z*delta.z > 0.25f*PADDING*PADDING)
                rebuild = true;
        }
#endif
    }

    // Record the range of sizes seen by warps in this work group.

    __shared__ real minBuffer[32], maxBuffer[32];
    if (threadIdx.x%32 == 0) {
        minBuffer[threadIdx.x/32] = minSize;
        maxBuffer[threadIdx.x/32] = maxSize;
    }
#ifdef CHECK_REBUILD_IN_BOUNDS
    rebuild = __syncthreads_or(rebuild);
#else
    __syncthreads();
#endif
    if (threadIdx.x == 0) {
        for (int i = 1; i < blockDim.x/32; i++) {
            minSize = min(minSize, minBuffer[i]);
            maxSize = max(maxSize, maxBuffer[i]);
        }
        blockSizeRange[blockIdx.x] = make_real2(minSize, maxSize);
#ifdef CHECK_REBUILD_IN_BOUNDS
        rebuildFlags[blockIdx.x] = rebuild;
#endif
    }
#ifndef CHECK_REBUILD_IN_BOUNDS
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        rebuildNeighborList[0] = 0;
#ifdef USE_PRUNING
        rebuildNeighborList[1] = 0;
#endif
    }
#endif
}

/**
 * Check whether any atom has moved enough so that we really need to rebuild the neighbor list.  The kernels that
 * build the neighbor list return immediately if it does not need to be rebuilt.
 *
 * With USE_PRUNING there are two lists.  The outer list (built by findBlocksWithInteractions, counted in outerCount)
 * uses the padding PADDING, and must be rebuilt once any atom has moved half of it.  The list used to compute forces
 * (built from the outer list by pruneFilter and prunePack, counted in interactionCount) uses the much smaller INNER_PADDING,
 * and is pruned again once any atom has moved half of that.  rebuildNeighborList[0] and [1] record the two decisions.
 *
 * With USE_ANCHORS the outer list is built from anchored positions (see findOuterBlockBounds), which oldPositions
 * holds for the last build.  It must be rebuilt once any anchor has moved half of PADDING, or an atom is farther from
 * its anchor than the margin the list assumed.
 */
extern "C" __global__ void checkRebuildNeighborList(const real4* __restrict__ posq, const real4* __restrict__ oldPositions,
        unsigned int* __restrict__ interactionCount, int* __restrict__ rebuildNeighborList, bool forceRebuild
#ifdef USE_PRUNING
        , const real4* __restrict__ prunePositions, unsigned int* __restrict__ outerCount
#endif
#ifdef USE_ANCHORS
        , real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        const int* __restrict__ anchorOffsets
#endif
        ) {
    bool rebuild = forceRebuild;
#ifdef USE_PRUNING
    bool prune = forceRebuild;
#endif
    for (int i = threadIdx.x+blockIdx.x*blockDim.x; i < NUM_ATOMS; i += blockDim.x*gridDim.x) {
        real4 pos = posq[i];
#ifdef USE_ANCHORS
        real4 old = oldPositions[i];
        real4 anchorPos = (old.w > 0 ? posq[i+anchorOffsets[i]] : pos);
        real4 delta = old-anchorPos;
        if (old.w > 0) {
            real4 offset = pos-anchorPos;
#ifdef USE_PERIODIC
            APPLY_PERIODIC_TO_DELTA(offset)
#endif
            if (offset.x*offset.x + offset.y*offset.y + offset.z*offset.z > old.w*old.w)
                rebuild = true;
        }
#else
        real4 delta = oldPositions[i]-pos;
#endif
        if (delta.x*delta.x + delta.y*delta.y + delta.z*delta.z > 0.25f*PADDING*PADDING)
            rebuild = true;
#ifdef USE_PRUNING
        delta = prunePositions[i]-pos;
        if (delta.x*delta.x + delta.y*delta.y + delta.z*delta.z > 0.25f*INNER_PADDING*INNER_PADDING)
            prune = true;
#endif
    }
#ifdef USE_PRUNING
    if (rebuild) {
        rebuildNeighborList[0] = 1;
        outerCount[0] = 0;
        outerCount[1] = 0;
    }
    if (rebuild || prune) {
        rebuildNeighborList[1] = 1;
        interactionCount[0] = 0;
        interactionCount[1] = 0;
    }
#else
    if (rebuild) {
        rebuildNeighborList[0] = 1;
        interactionCount[0] = 0;
        interactionCount[1] = 0;
    }
#endif
}

extern "C" __global__ void computeSortKeys(const real4* __restrict__ blockBoundingBox, unsigned int* __restrict__ sortedBlocks, real2* __restrict__ blockSizeRange, int numSizes,
        int* __restrict__ rebuildNeighborList
#ifdef CHECK_REBUILD_IN_BOUNDS
        , const int* __restrict__ rebuildFlags, unsigned int* __restrict__ interactionCount, bool forceRebuild
#endif
        ) {
#ifdef CHECK_REBUILD_IN_BOUNDS
    // Combine the decisions of the work groups of findBlockBoundsByWarp.  Every work group does this, and the first one
    // records the decision for the kernels that follow, and resets the counts as checkRebuildNeighborList would.

    bool rebuild = forceRebuild;
    for (int i = threadIdx.x; i < numSizes; i += blockDim.x)
        if (rebuildFlags[i])
            rebuild = true;
    rebuild = __syncthreads_or(rebuild);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        rebuildNeighborList[0] = rebuild;
        if (rebuild) {
            interactionCount[0] = 0;
            interactionCount[1] = 0;
        }
    }
    if (!rebuild)
        return; // The neighbor list doesn't need to be rebuilt.
#else
    if (rebuildNeighborList[0] == 0)
        return; // The neighbor list doesn't need to be rebuilt.
#endif

    // Find the total range of sizes recorded by all blocks.  The threads of each work group share the work, since
    // findBlockBoundsByWarp records many ranges.

    __shared__ real2 sizeRange;
    __shared__ real minBuffer[32], maxBuffer[32];
    real minSize = 1e38, maxSize = 0;
    for (int i = threadIdx.x; i < numSizes; i += blockDim.x) {
        real2 size = blockSizeRange[i];
        if (i == 0 || size.x > 0)
            minSize = min(minSize, size.x);
        maxSize = max(maxSize, size.y);
    }
    for (int i = 16; i > 0; i /= 2) {
        minSize = min(minSize, __shfl_xor_sync(0xffffffff, minSize, i));
        maxSize = max(maxSize, __shfl_xor_sync(0xffffffff, maxSize, i));
    }
    if (threadIdx.x%32 == 0) {
        minBuffer[threadIdx.x/32] = minSize;
        maxBuffer[threadIdx.x/32] = maxSize;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < (blockDim.x+31)/32; i++) {
            minSize = min(minSize, minBuffer[i]);
            maxSize = max(maxSize, maxBuffer[i]);
        }
        sizeRange.x = LOG(minSize);
        sizeRange.y = LOG(maxSize);
    }
    __syncthreads();

    // Sort keys store the bin in the high order part and the block in the low
    // order part.

    int numSizeBins = 20;
    real scale = numSizeBins/(sizeRange.y-sizeRange.x);
    for (unsigned int i = threadIdx.x+blockIdx.x*blockDim.x; i < NUM_BLOCKS; i += blockDim.x*gridDim.x) {
        real4 box = blockBoundingBox[i];
        real size = LOG(box.x+box.y+box.z);
        int bin = (size-sizeRange.x)*scale;
        bin = max(0, min(bin, numSizeBins-1));
        sortedBlocks[i] = (((unsigned int) bin)<<BIN_SHIFT) + i;
    }
}

/**
 * Sort the keys computed by computeSortKeys.  Every key is unique, since its low order bits hold the block index,
 * so each one goes to the position given by the number of smaller keys.  One warp counts them for each key, so the
 * work is spread over the whole GPU instead of a few work groups each comparing a key to all the others.
 */
extern "C" __global__ void sortBlockKeys(const unsigned int* __restrict__ keys, unsigned int* __restrict__ sortedKeys, const int* __restrict__ rebuildNeighborList) {
    if (rebuildNeighborList[0] == 0)
        return; // The neighbor list doesn't need to be rebuilt.
    const int indexInWarp = threadIdx.x%32;
    const int totalWarps = blockDim.x*gridDim.x/32;
    for (int index = (blockIdx.x*blockDim.x+threadIdx.x)/32; index < NUM_BLOCKS; index += totalWarps) {
        const unsigned int key = keys[index];
        int count = 0;
        for (int i = indexInWarp; i < NUM_BLOCKS; i += 32)
            count += (keys[i] < key ? 1 : 0);
        for (int offset = 16; offset > 0; offset /= 2)
            count += __shfl_xor_sync(0xffffffff, count, offset);
        if (indexInWarp == 0)
            sortedKeys[count] = key;
    }
}

/**
 * Sort the data about bounding boxes so it can be accessed more efficiently in the next kernel.
 */
extern "C" __global__ void sortBoxData(const unsigned int* __restrict__ sortedBlocks, const real4* __restrict__ blockCenter,
        const real4* __restrict__ blockBoundingBox, real4* __restrict__ sortedBlockCenter, half3* __restrict__ sortedBlockBoundingBox,
#ifdef USE_LARGE_BLOCKS
        real4* __restrict__ largeBlockCenter, half3* __restrict__ largeBlockBoundingBox, real4 periodicBoxSize,
        real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
#endif
        const int* __restrict__ rebuildNeighborList
#ifdef USE_PRUNING
        , int* __restrict__ outerBlockCounts
#endif
        ) {
    if (rebuildNeighborList[0] == 0)
        return; // The neighbor list doesn't need to be rebuilt.
    for (int i = threadIdx.x+blockIdx.x*blockDim.x; i < NUM_BLOCKS; i += blockDim.x*gridDim.x) {
#ifdef USE_PRUNING
        outerBlockCounts[i] = 0;
#endif
        unsigned int index = sortedBlocks[i] & BLOCK_INDEX_MASK;
        sortedBlockCenter[i] = blockCenter[index];
        sortedBlockBoundingBox[i] = half3(trimTo3(blockBoundingBox[index]));

#ifdef USE_LARGE_BLOCKS
        // Compute the sizes of large blocks (composed of 32 regular blocks) starting from each block.
    
        real4 minPos = blockCenter[index]-blockBoundingBox[index];
        real4 maxPos = blockCenter[index]+blockBoundingBox[index];
        int last = min(i+32, NUM_BLOCKS);
        for (int j = i+1; j < last; j++) {
            unsigned int index2 = sortedBlocks[j] & BLOCK_INDEX_MASK;
            real4 blockPos = blockCenter[index2];
            real4 width = blockBoundingBox[index2];
#ifdef USE_PERIODIC
            real4 center = 0.5f*(maxPos+minPos);
            APPLY_PERIODIC_TO_POS_WITH_CENTER(blockPos, center)
#endif
            minPos = make_real4(min(minPos.x, blockPos.x-width.x), min(minPos.y, blockPos.y-width.y), min(minPos.z, blockPos.z-width.z), 0);
            maxPos = make_real4(max(maxPos.x, blockPos.x+width.x), max(maxPos.y, blockPos.y+width.y), max(maxPos.z, blockPos.z+width.z), 0);
        }
        largeBlockCenter[i] = 0.5f*(maxPos+minPos);
        largeBlockBoundingBox[i] = half3(trimTo3(0.5f*(maxPos-minPos)));
#endif
    }
}

#ifdef USE_ANCHORS
/**
 * Compute the anchored positions the outer list is built from, and bounding boxes for them, for one block.  All 32
 * threads of a warp call this together.
 *
 * An atom whose only purpose here is to follow another one (a hydrogen constrained to a heavy atom) is represented by
 * its anchor's position, with a margin equal to the largest distance it can be from it.  w holds the margin, or 0 for
 * an atom that is its own anchor.  If atoms i and j are within a distance r, their anchors are within r plus the sum of
 * the margins, so a list of anchors within PADDED_CUTOFF plus the margins stays valid until some anchor has moved half
 * the padding.  The anchors move much less per step than the hydrogens, so the list lasts longer.  An atom that is
 * farther from its anchor than its margin, such as before constraints have been applied, is its own anchor.
 *
 * The bounding box of a block contains a sphere of the margin's radius around each anchored position, and center.w is
 * the radius of a sphere around the center containing them all, so the tests on boxes in findBlocksWithInteractions are
 * unchanged.
 */
__device__ void findOuterBlockBounds(int block, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        const real4* __restrict__ posq, const int* __restrict__ anchorOffsets, const float* __restrict__ anchorMargins,
        real4* __restrict__ outerPositions, real4* __restrict__ outerBlockCenter, real4* __restrict__ outerBlockBoundingBox) {
    const int indexInWarp = threadIdx.x%32;
    const int atom = block*TILE_SIZE+indexInWarp;
    const int count = min(TILE_SIZE, NUM_ATOMS-block*TILE_SIZE);
    real4 pos = posq[atom];
    real4 anchored = make_real4(pos.x, pos.y, pos.z, 0);
    if (indexInWarp < count && anchorOffsets[atom] != 0) {
        real4 anchorPos = posq[atom+anchorOffsets[atom]];
        real4 offset = pos-anchorPos;
#ifdef USE_PERIODIC
        APPLY_PERIODIC_TO_DELTA(offset)
#endif
        real margin = anchorMargins[atom];
        if (offset.x*offset.x + offset.y*offset.y + offset.z*offset.z <= margin*margin)
            anchored = make_real4(anchorPos.x, anchorPos.y, anchorPos.z, margin);
    }
    outerPositions[atom] = anchored;
    pos = make_real4(__shfl_sync(0xffffffff, anchored.x, 0), __shfl_sync(0xffffffff, anchored.y, 0), __shfl_sync(0xffffffff, anchored.z, 0), __shfl_sync(0xffffffff, anchored.w, 0));
#ifdef USE_PERIODIC
    APPLY_PERIODIC_TO_POS(pos)
#endif
    real4 minPos = make_real4(pos.x-pos.w, pos.y-pos.w, pos.z-pos.w, 0);
    real4 maxPos = make_real4(pos.x+pos.w, pos.y+pos.w, pos.z+pos.w, 0);
    for (int i = 1; i < count; i++) {
        pos = make_real4(__shfl_sync(0xffffffff, anchored.x, i), __shfl_sync(0xffffffff, anchored.y, i), __shfl_sync(0xffffffff, anchored.z, i), __shfl_sync(0xffffffff, anchored.w, i));
#ifdef USE_PERIODIC
        real4 center = 0.5f*(maxPos+minPos);
        APPLY_PERIODIC_TO_POS_WITH_CENTER(pos, center)
#endif
        minPos = make_real4(min(minPos.x,pos.x-pos.w), min(minPos.y,pos.y-pos.w), min(minPos.z,pos.z-pos.w), 0);
        maxPos = make_real4(max(maxPos.x,pos.x+pos.w), max(maxPos.y,pos.y+pos.w), max(maxPos.z,pos.z+pos.w), 0);
    }
    real4 center = 0.5f*(maxPos+minPos);
    real4 delta = anchored-center;
#ifdef USE_PERIODIC
    APPLY_PERIODIC_TO_DELTA(delta)
#endif
    real radius = (indexInWarp < count ? SQRT(delta.x*delta.x+delta.y*delta.y+delta.z*delta.z)+anchored.w : 0);
    for (int i = 16; i > 0; i /= 2)
        radius = max(radius, __shfl_xor_sync(0xffffffff, radius, i));
    center.w = radius;
    if (indexInWarp == 0) {
        outerBlockBoundingBox[block] = 0.5f*(maxPos-minPos);
        outerBlockCenter[block] = center;
    }
}

/**
 * Compute the anchored positions and their bounding boxes for every block (see findOuterBlockBounds), when the outer
 * list is about to be rebuilt.  This uses one warp per block.
 */
extern "C" __global__ void findOuterBlockBoundsKernel(real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        const real4* __restrict__ posq, const int* __restrict__ anchorOffsets, const float* __restrict__ anchorMargins,
        real4* __restrict__ outerPositions, real4* __restrict__ outerBlockCenter, real4* __restrict__ outerBlockBoundingBox,
        const int* __restrict__ rebuildNeighborList) {
    if (rebuildNeighborList[0] == 0)
        return; // The neighbor list doesn't need to be rebuilt.
    const int totalWarps = blockDim.x*gridDim.x/32;
    for (int block = (blockIdx.x*blockDim.x+threadIdx.x)/32; block < NUM_BLOCKS; block += totalWarps)
        findOuterBlockBounds(block, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ,
                posq, anchorOffsets, anchorMargins, outerPositions, outerBlockCenter, outerBlockBoundingBox);
}
#endif

__device__ int saveSinglePairs(int x, int* atoms, int* flags, int length, unsigned int maxSinglePairs, unsigned int* singlePairCount, int2* singlePairs, int* sumBuffer, volatile unsigned int& pairStartIndex) {
    // Record interactions that should be computed as single pairs rather than in blocks.
    
    const int indexInWarp = threadIdx.x%32;
    int sum = 0;
    #pragma unroll 8 // (GROUP_SIZE / TILE_SIZE)
    for (int i = indexInWarp; i < length; i += 32) {
        int count = __popc(flags[i]);
        sum += (count <= MAX_BITS_FOR_PAIRS ? count : 0);
    }
    for (int i = 1; i < 32; i *= 2) {
        int n = __shfl_up_sync(0xffffffff, sum, i);
        if (indexInWarp >= i)
            sum += n;
    }
    if (indexInWarp == 31)
        pairStartIndex = atomicAdd(singlePairCount,(unsigned int) sum);
    __syncwarp();
    int prevSum = __shfl_up_sync(0xffffffff, sum, 1);
    unsigned int pairIndex = pairStartIndex + (indexInWarp > 0 ? prevSum : 0);
    for (int i = indexInWarp; i < length; i += 32) {
        int count = __popc(flags[i]);
        if (count <= MAX_BITS_FOR_PAIRS && pairIndex+count <= maxSinglePairs) {
            int f = flags[i];
            while (f != 0) {
                singlePairs[pairIndex] = make_int2(atoms[i], x*TILE_SIZE+__ffs(f)-1);
                f &= f-1;
                pairIndex++;
            }
        }
    }
    
    // Compact the remaining interactions.
    
    const int warpMask = (1<<indexInWarp)-1;
    int numCompacted = 0;
    for (int start = 0; start < length; start += 32) {
        int i = start+indexInWarp;
        int atom = atoms[i];
        int flag = flags[i];
        bool include = (i < length && __popc(flags[i]) > MAX_BITS_FOR_PAIRS);
        int includeFlags = BALLOT(include);
        if (include) {
            int index = numCompacted+__popc(includeFlags&warpMask);
            atoms[index] = atom;
            flags[index] = flag;
        }
        numCompacted += __popc(includeFlags);
    }
    return numCompacted;
}

#ifdef USE_PRUNING
/**
 * Append the atoms in a warp's buffer to block x's part of the outer list.  Several warps may append to the same block,
 * so space is reserved with an atomic add.  outerCount[0] records the largest count for any block, so the host can tell
 * whether the space for each block (capacity) was enough.
 */
__device__ void appendToOuterList(int x, const int* buffer, int count, unsigned int* outerCount, int* outerBlockCounts,
        unsigned int* outerAtoms, unsigned int capacity) {
    const int indexInWarp = threadIdx.x%32;
    __syncwarp();
    unsigned int start = 0;
    if (indexInWarp == 0) {
        start = atomicAdd((unsigned int*) &outerBlockCounts[x], (unsigned int) count);
        atomicMax(&outerCount[0], start+count);
    }
    start = __shfl_sync(0xffffffff, start, 0);
    for (int j = indexInWarp; j < count; j += 32)
        if (start+j < capacity)
            outerAtoms[(size_t) x*capacity+start+j] = buffer[j];
    __syncwarp();
}
#endif

/**
 * Compare count atoms from a warp's queue (at most 32, one per thread) to the 32 atoms of block x, add those that
 * interact with any of them to the buffer, and store full tiles (or append to the outer list) when it gets full.
 * The atoms of block x are in xPos, translated to the periodic copy nearest its center, with half their squared
 * distance from the origin in w.
 */
__device__ void compareQueuedAtoms(int count, int x, const int* queue, const real4* __restrict__ posq, real4 blockCenterX,
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        const real4* xPos,
#ifdef USE_ANCHORS
        const real* xMargin,
#endif
        int* buffer, int* flagsBuffer, int& neighborsInBuffer, unsigned int* __restrict__ interactionCount,
        int* __restrict__ interactingTiles, unsigned int* __restrict__ interactingAtoms, int2* __restrict__ singlePairs,
        unsigned int maxTiles, unsigned int maxSinglePairs, int* sumBuffer, volatile unsigned int& pairStartIndex,
        volatile unsigned int& tileStartIndex) {
    const int indexInWarp = threadIdx.x%32;
    const int warpMask = (1<<indexInWarp)-1;
    int atom2 = 0;
    int interacts = 0;
    if (indexInWarp < count) {
        atom2 = queue[indexInWarp];
        real4 pos2 = posq[atom2];
#ifdef USE_PERIODIC
        APPLY_PERIODIC_TO_POS_WITH_CENTER(pos2, blockCenterX)
#endif
#ifdef USE_ANCHORS
        const real margin2 = pos2.w;
#endif
        pos2.w = 0.5f * (pos2.x * pos2.x + pos2.y * pos2.y + pos2.z * pos2.z);
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            real4 posj = xPos[j];
            real halfDist2 = posj.w + pos2.w - posj.x*pos2.x - posj.y*pos2.y - posj.z*pos2.z;
#ifdef USE_ANCHORS
            const real reach = PADDED_CUTOFF+margin2+xMargin[j];
            interacts |= (halfDist2 < 0.5f*reach*reach ? 1<<j : 0);
#else
            interacts |= (halfDist2 < 0.5f * PADDED_CUTOFF_SQUARED ? 1<<j : 0);
#endif
        }
    }
    int includeAtomFlags = BALLOT(interacts);
    if (interacts) {
        int index = neighborsInBuffer+__popc(includeAtomFlags&warpMask);
        buffer[index] = atom2;
        flagsBuffer[index] = interacts;
    }
    neighborsInBuffer += __popc(includeAtomFlags);
    __syncwarp();
#ifdef USE_PRUNING
    if (neighborsInBuffer > BUFFER_SIZE-TILE_SIZE) {
        appendToOuterList(x, buffer, neighborsInBuffer, interactionCount, interactingTiles, interactingAtoms, maxTiles);
        neighborsInBuffer = 0;
    }
#else
    if (neighborsInBuffer > BUFFER_SIZE-TILE_SIZE) {
        // Store the new tiles to memory.

#if MAX_BITS_FOR_PAIRS > 0
        neighborsInBuffer = saveSinglePairs(x, buffer, flagsBuffer, neighborsInBuffer, maxSinglePairs, &interactionCount[1], singlePairs, sumBuffer, pairStartIndex);
#endif
        unsigned int tilesToStore = neighborsInBuffer/TILE_SIZE;
        if (tilesToStore > 0) {
            if (indexInWarp == 0)
                tileStartIndex = atomicAdd(&interactionCount[0], tilesToStore);
            __syncwarp();
            unsigned int newTileStartIndex = tileStartIndex;
            if (newTileStartIndex+tilesToStore <= maxTiles) {
                if (indexInWarp < tilesToStore)
                    interactingTiles[newTileStartIndex+indexInWarp] = x;
                #pragma unroll 8 // (GROUP_SIZE / TILE_SIZE)
                for (int j = 0; j < tilesToStore; j++)
                    interactingAtoms[(newTileStartIndex+j)*TILE_SIZE+indexInWarp] = buffer[indexInWarp+j*TILE_SIZE];
            }
            __syncwarp();
            if (indexInWarp+TILE_SIZE*tilesToStore < BUFFER_SIZE)
                buffer[indexInWarp] = buffer[indexInWarp+TILE_SIZE*tilesToStore];
            neighborsInBuffer -= TILE_SIZE*tilesToStore;
        }
        __syncwarp();
    }
#endif
}

/**
 * Compare the bounding boxes for each pair of atom blocks (comprised of 32 atoms each), forming a tile. If the two
 * atom blocks are sufficiently far apart, mark them as non-interacting. There are two stages in the algorithm.
 *
 * STAGE 1:
 *
 * A coarse grained atom block against interacting atom block neighbour list is constructed. 
 *
 * Each warp first loads in some block X of interest. Each thread within the warp then loads 
 * in a different atom block Y. If Y has exclusions with X, then Y is not processed.  If the bounding boxes 
 * of the two atom blocks are within the cutoff distance, then the two atom blocks are considered to be
 * interacting and Y is added to the buffer for X.
 *
 * STAGE 2:
 *
 * A fine grained atom block against interacting atoms neighbour list is constructed.
 *
 * The warp loops over atom blocks Y that were found to (possibly) interact with atom block X.  Each thread
 * in the warp loops over the 32 atoms in X and compares their positions to one particular atom from block Y.
 * If it finds one closer than the cutoff distance, the atom is added to the list of atoms interacting with block X.
 * This continues until the buffer fills up, at which point the results are written to global memory.
 *
 * [in] periodicBoxSize        - size of the rectangular periodic box
 * [in] invPeriodicBoxSize     - inverse of the periodic box
 * [in] blockCenter            - the center of each bounding box
 * [in] blockBoundingBox       - bounding box of each atom block
 * [out] interactionCount      - total number of tiles that have interactions
 * [out] interactingTiles      - set of blocks that have interactions
 * [out] interactingAtoms      - a list of atoms that interact with each atom block
 * [in] posq                   - x,y,z coordinates of each atom and charge q
 * [in] maxTiles               - maximum number of tiles to process, used for multi-GPUs
 * [in] startBlockIndex        - first block to process, used for multi-GPUs,
 * [in] numBlocks              - total number of atom blocks
 * [in] sortedBlocks           - a sorted list of atom blocks based on volume
 * [in] sortedBlockCenter      - sorted centers, duplicated for fast access to avoid indexing
 * [in] sortedBlockBoundingBox - sorted bounding boxes, duplicated for fast access
 * [in] exclusionIndices       - maps into exclusionRowIndices with the starting position for a given atom
 * [in] exclusionRowIndices    - stores the a continuous list of exclusions
 *           eg: block 0 is excluded from atom 3,5,6
 *               block 1 is excluded from atom 3,4
 *               block 2 is excluded from atom 1,3,5,6
 *              exclusionIndices[0][3][5][8]
 *           exclusionRowIndices[3][5][6][3][4][1][3][5][6]
 *                         index 0  1  2  3  4  5  6  7  8 
 * [out] oldPos                - stores the positions of the atoms in which this neighbourlist was built on
 *                             - this is used to decide when to rebuild a neighbourlist
 * [in] rebuildNeighbourList   - whether or not to execute this kernel
 *
 * With USE_PRUNING this builds the outer list instead, which pruneFilter and prunePack turn into tiles.  It has no
 * tiles or single pairs: block x's neighbors are stored in interactingAtoms[x*maxTiles] onwards, interactingTiles[x]
 * holds how many there are (set to 0 by sortBoxData), and interactionCount[0] the largest number for any block.
 * maxTiles is the space for each block.  The neighbors of a block are stored in no particular order.
 *
 * With USE_ANCHORS posq holds the anchored positions computed by findOuterBlockBounds, with each atom's margin in w,
 * and the block data describes their bounding boxes.  Atoms are neighbors if their anchored positions are within
 * PADDED_CUTOFF plus the two margins.  oldPositions then records the anchored positions, for checkRebuildNeighborList.
 */
extern "C" __global__ __launch_bounds__(GROUP_SIZE,3) void findBlocksWithInteractions(real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        unsigned int* __restrict__ interactionCount, int* __restrict__ interactingTiles, unsigned int* __restrict__ interactingAtoms,
        int2* __restrict__ singlePairs, const real4* __restrict__ posq, unsigned int maxTiles, unsigned int maxSinglePairs, unsigned int startBlockIndex,
        unsigned int numBlocks, unsigned int* __restrict__ sortedBlocks, const real4* __restrict__ sortedBlockCenter, const half3* __restrict__ sortedBlockBoundingBox,
#ifdef USE_LARGE_BLOCKS
        real4* __restrict__ largeBlockCenter, half3* __restrict__ largeBlockBoundingBox,
#endif
        const unsigned int* __restrict__ exclusionIndices, const unsigned int* __restrict__ exclusionRowIndices,
        real4* __restrict__ oldPositions, const int* __restrict__ rebuildNeighborList
#ifdef SPLIT_BUILD
        , unsigned int* __restrict__ leftoverAtoms, int* __restrict__ leftoverCount, unsigned int* __restrict__ segmentsDone
#endif
#ifdef BUILD_TIMING
        , long long* __restrict__ buildTiming
#endif
        ) {

    if (rebuildNeighborList[0] == 0)
        return; // The neighbor list doesn't need to be rebuilt.

    const int indexInWarp = threadIdx.x%32;
    const int warpStart = threadIdx.x-indexInWarp;
    const int totalWarps = blockDim.x*gridDim.x/32;
    const int warpIndex = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int warpMask = (1<<indexInWarp)-1;
    __shared__ int workgroupBuffer[BUFFER_SIZE*(GROUP_SIZE/32)];
    __shared__ int workgroupFlagsBuffer[BUFFER_SIZE*(GROUP_SIZE/32)];
    __shared__ int warpExclusions[MAX_EXCLUSIONS*(GROUP_SIZE/32)];
    __shared__ real4 posBuffer[GROUP_SIZE];
#ifdef USE_ANCHORS
    __shared__ real marginBuffer[GROUP_SIZE];
#endif
    __shared__ volatile unsigned int workgroupTileIndex[GROUP_SIZE/32];
    __shared__ unsigned int workgroupPairStartIndex[GROUP_SIZE/32];
    __shared__ int workgroupQueue[2*TILE_SIZE*(GROUP_SIZE/32)];
    int* sumBuffer = (int*) posBuffer; // Reuse the same buffer to save memory
    int* queue = workgroupQueue+2*TILE_SIZE*(warpStart/32);
    int* buffer = workgroupBuffer+BUFFER_SIZE*(warpStart/32);
    int* flagsBuffer = workgroupFlagsBuffer+BUFFER_SIZE*(warpStart/32);
    int* exclusionsForX = warpExclusions+MAX_EXCLUSIONS*(warpStart/32);
    volatile unsigned int& tileStartIndex = workgroupTileIndex[warpStart/32];
    volatile unsigned int& pairStartIndex = workgroupPairStartIndex[warpStart/32];

    // Loop over blocks.  Each block only looks for neighbors in the blocks after it, so the first blocks have far more
    // work than the last, and the kernel would last as long as the warp with the most.  With pruning, the blocks to
    // search are therefore split into segments of SEGMENT_BLOCKS, each searched by its own warp, which append to the
    // block's part of the outer list.  Without pruning, one warp searches all of them, since it packs them into tiles.

#if defined(USE_PRUNING) || defined(SPLIT_BUILD)
    // With SPLIT_BUILD the ordinary list is split the same way, and each segment's warp stores its own tiles, so a
    // block may end up with a partially filled tile for each segment.
    const int numSegments = (NUM_BLOCKS+SEGMENT_BLOCKS-1)/SEGMENT_BLOCKS;
    for (int item = warpIndex; item < numBlocks*numSegments; item += totalWarps) {
        const int block1 = startBlockIndex+item%numBlocks;
        const int segmentStart = block1+1+(item/numBlocks)*SEGMENT_BLOCKS;
        if (segmentStart >= NUM_BLOCKS)
            continue;
        const int segmentEnd = min(segmentStart+SEGMENT_BLOCKS, NUM_BLOCKS);
#else
    for (int block1 = startBlockIndex+warpIndex; block1 < startBlockIndex+numBlocks; block1 += totalWarps) {
        const int segmentStart = block1+1, segmentEnd = NUM_BLOCKS;
#endif
        int queued = 0;
#ifdef BUILD_TIMING
        const long long itemStart = clock64();
        long long compareTime = 0;
        int numCandidates = 0;
#endif
        // Load data for this block.  Note that all threads in a warp are processing the same block.
        
        int x = sortedBlocks[block1] & BLOCK_INDEX_MASK;
        real4 blockCenterX = sortedBlockCenter[block1];
        real3 blockSizeX = sortedBlockBoundingBox[block1].toReal3();
        int neighborsInBuffer = 0;
        real4 pos1 = posq[x*TILE_SIZE+indexInWarp];
#ifdef USE_PERIODIC
        const bool singlePeriodicCopy = (0.5f*periodicBoxSize.x-blockSizeX.x >= BUILD_REACH &&
                                         0.5f*periodicBoxSize.y-blockSizeX.y >= BUILD_REACH &&
                                         0.5f*periodicBoxSize.z-blockSizeX.z >= BUILD_REACH);
        if (singlePeriodicCopy) {
            // The box is small enough that we can just translate all the atoms into a single periodic
            // box, then skip having to apply periodic boundary conditions later.
            
            APPLY_PERIODIC_TO_POS_WITH_CENTER(pos1, blockCenterX)
        }
#endif
#ifdef USE_ANCHORS
        const real margin1 = pos1.w;
        marginBuffer[threadIdx.x] = margin1;
#endif
        pos1.w = 0.5f * (pos1.x * pos1.x + pos1.y * pos1.y + pos1.z * pos1.z);
        posBuffer[threadIdx.x] = pos1;

        // Atoms of other blocks are first tested against this block's bounding box, and only those within the cutoff
        // of it are compared to all its atoms, 32 at a time from a queue.  On this box two thirds of the atoms in
        // candidate blocks fail that test.  It can only reject atoms that are farther than the cutoff from every atom
        // of the block, so the list is unchanged.  It is skipped when the block is too large for a single periodic
        // copy, since the box then does not bound the periodic copies the comparison uses.
#if defined(DISABLE_BUILD_PREFILTER)
        const bool useQueue = false;
#elif defined(USE_PERIODIC)
        const bool useQueue = singlePeriodicCopy;
#else
        const bool useQueue = true;
#endif

        // Load exclusion data for block x.
        
        const int exclusionStart = exclusionRowIndices[x];
        const int exclusionEnd = exclusionRowIndices[x+1];
        const int numExclusions = exclusionEnd-exclusionStart;
        #pragma unroll 4 // (MAX_EXCLUSIONS)
        for (int j = indexInWarp; j < numExclusions; j += 32)
            exclusionsForX[j] = exclusionIndices[exclusionStart+j];
        if (MAX_EXCLUSIONS > 32)
            __syncthreads();
        
        // Loop over atom blocks to search for neighbors.  The threads in a warp compare block1 against 32
        // other blocks in parallel.

#ifdef USE_LARGE_BLOCKS
        int largeBlockFlags = 0;
        int loadedLargeBlocks = 0;
#endif
        for (int block2Base = segmentStart; block2Base < segmentEnd; block2Base += 32) {
#ifdef USE_LARGE_BLOCKS
            if (loadedLargeBlocks == 0) {
                // Check the next set of large blocks.

                int largeBlockIndex = block2Base + 32*indexInWarp;
                bool includeLargeBlock = false;
                if (largeBlockIndex < NUM_BLOCKS) {
                    real4 largeCenter = largeBlockCenter[largeBlockIndex];
                    real3 largeSize = largeBlockBoundingBox[largeBlockIndex].toReal3();
                    real4 blockDelta = blockCenterX-largeCenter;
#ifdef USE_PERIODIC
                    APPLY_PERIODIC_TO_DELTA(blockDelta)
#endif
                    blockDelta.x = max(0.0f, fabs(blockDelta.x)-blockSizeX.x-largeSize.x);
                    blockDelta.y = max(0.0f, fabs(blockDelta.y)-blockSizeX.y-largeSize.y);
                    blockDelta.z = max(0.0f, fabs(blockDelta.z)-blockSizeX.z-largeSize.z);
                    includeLargeBlock = (blockDelta.x*blockDelta.x+blockDelta.y*blockDelta.y+blockDelta.z*blockDelta.z < PADDED_CUTOFF_SQUARED);
#ifdef TRICLINIC
                    // The calculation to find the nearest periodic copy is only guaranteed to work if the nearest copy is less than half a box width away.
                    // If there's any possibility we might have missed it, do a detailed check.

                    if (periodicBoxSize.z/2-blockSizeX.z-largeSize.z < BUILD_REACH || periodicBoxSize.y/2-blockSizeX.y-largeSize.y < BUILD_REACH)
                        includeLargeBlock = true;
#endif
                }
                largeBlockFlags = BALLOT(includeLargeBlock);
                loadedLargeBlocks = 32;
            }
            loadedLargeBlocks--;
            if ((largeBlockFlags&1) == 0) {
                // None of the next 32 blocks interact with block 1.

                largeBlockFlags >>= 1;
                continue;
            }
            largeBlockFlags >>= 1;
#endif
            int block2 = block2Base+indexInWarp;
            bool includeBlock2 = (block2 < segmentEnd);
            bool forceInclude = false;
            if (includeBlock2) {
                real4 blockCenterY = sortedBlockCenter[block2];
                real3 blockSizeY = sortedBlockBoundingBox[block2].toReal3();
                real4 blockDelta = blockCenterX-blockCenterY;
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(blockDelta)
#endif
                includeBlock2 &= (blockDelta.x*blockDelta.x+blockDelta.y*blockDelta.y+blockDelta.z*blockDelta.z < (PADDED_CUTOFF+blockCenterX.w+blockCenterY.w)*(PADDED_CUTOFF+blockCenterX.w+blockCenterY.w));
                blockDelta.x = max(0.0f, fabs(blockDelta.x)-blockSizeX.x-blockSizeY.x);
                blockDelta.y = max(0.0f, fabs(blockDelta.y)-blockSizeX.y-blockSizeY.y);
                blockDelta.z = max(0.0f, fabs(blockDelta.z)-blockSizeX.z-blockSizeY.z);
                includeBlock2 &= (blockDelta.x*blockDelta.x+blockDelta.y*blockDelta.y+blockDelta.z*blockDelta.z < PADDED_CUTOFF_SQUARED);
#ifdef TRICLINIC
                // The calculation to find the nearest periodic copy is only guaranteed to work if the nearest copy is less than half a box width away.
                // If there's any possibility we might have missed it, do a detailed check.

                if (periodicBoxSize.z/2-blockSizeX.z-blockSizeY.z < BUILD_REACH || periodicBoxSize.y/2-blockSizeX.y-blockSizeY.y < BUILD_REACH)
                    includeBlock2 = forceInclude = true;
#endif
                if (includeBlock2) {
                    int y = sortedBlocks[block2] & BLOCK_INDEX_MASK;
                    #pragma unroll 4 // (MAX_EXCLUSIONS)
                    for (int k = 0; k < numExclusions; k++)
                        includeBlock2 &= (exclusionsForX[k] != y);
                }
            }
            
            // Loop over any blocks we identified as potentially containing neighbors.
            
            int includeBlockFlags = BALLOT(includeBlock2);
            int forceIncludeFlags = BALLOT(forceInclude);
            while (includeBlockFlags != 0) {
                int i = __ffs(includeBlockFlags)-1;
                includeBlockFlags &= includeBlockFlags-1;
                forceInclude = (forceIncludeFlags>>i) & 1;
                int y = sortedBlocks[block2Base+i] & BLOCK_INDEX_MASK;
#ifdef BUILD_TIMING
                numCandidates++;
#endif

                // Check each atom in block Y for interactions.

                int atom2 = y*TILE_SIZE+indexInWarp;
                real4 pos2 = posq[atom2];
#ifdef USE_PERIODIC
                if (singlePeriodicCopy) {
                    APPLY_PERIODIC_TO_POS_WITH_CENTER(pos2, blockCenterX)
                }
#endif
#ifdef USE_ANCHORS
                const real margin2 = pos2.w;
#endif
                pos2.w = 0.5f * (pos2.x * pos2.x + pos2.y * pos2.y + pos2.z * pos2.z);

                real4 blockCenterY = sortedBlockCenter[block2Base+i];
                real3 atomDelta = trimTo3(posBuffer[warpStart+indexInWarp])-trimTo3(blockCenterY);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(atomDelta)
#endif
#ifdef USE_ANCHORS
                const real atomReach = PADDED_CUTOFF+blockCenterY.w+margin1;
#else
                const real atomReach = PADDED_CUTOFF+blockCenterY.w;
#endif
                int atomFlags = BALLOT(forceInclude || atomDelta.x*atomDelta.x+atomDelta.y*atomDelta.y+atomDelta.z*atomDelta.z < atomReach*atomReach);
                if (useQueue) {
                    bool enqueue = (atom2 < NUM_ATOMS && atomFlags != 0);
                    if (enqueue && !forceInclude) {
                        real3 boxDelta = trimTo3(pos2)-trimTo3(blockCenterX);
                        boxDelta.x = max((real) 0, fabs(boxDelta.x)-blockSizeX.x);
                        boxDelta.y = max((real) 0, fabs(boxDelta.y)-blockSizeX.y);
                        boxDelta.z = max((real) 0, fabs(boxDelta.z)-blockSizeX.z);
#ifdef USE_ANCHORS
                        const real boxReach = PADDED_CUTOFF+margin2;
#else
                        const real boxReach = PADDED_CUTOFF;
#endif
                        enqueue = (boxDelta.x*boxDelta.x+boxDelta.y*boxDelta.y+boxDelta.z*boxDelta.z < boxReach*boxReach);
                    }
                    int enqueueFlags = BALLOT(enqueue);
                    if (enqueue)
                        queue[queued+__popc(enqueueFlags&warpMask)] = atom2;
                    queued += __popc(enqueueFlags);
                    if (queued >= 32) {
                        __syncwarp();
#ifdef BUILD_TIMING
                        const long long compareStart = clock64();
#endif
                        compareQueuedAtoms(32, x, queue, posq, blockCenterX, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ,
                                posBuffer+warpStart,
#ifdef USE_ANCHORS
                                marginBuffer+warpStart,
#endif
                                buffer, flagsBuffer, neighborsInBuffer, interactionCount, interactingTiles, interactingAtoms, singlePairs,
                                maxTiles, maxSinglePairs, sumBuffer+warpStart, pairStartIndex, tileStartIndex);
#ifdef BUILD_TIMING
                        compareTime += clock64()-compareStart;
#endif
                        if (indexInWarp+32 < queued)
                            queue[indexInWarp] = queue[indexInWarp+32];
                        queued -= 32;
                        __syncwarp();
                    }
                    continue;
                }
                int interacts = 0;
                if (atom2 < NUM_ATOMS && atomFlags != 0) {
#ifdef USE_PERIODIC
                    if (!singlePeriodicCopy) {
                        int first = __ffs(atomFlags)-1;
                        int last = 32-__clz(atomFlags);
                        for (int j = first; j < last; j++) {
                            real3 delta = trimTo3(pos2)-trimTo3(posBuffer[warpStart+j]);
                            APPLY_PERIODIC_TO_DELTA(delta)
#ifdef USE_ANCHORS
                            const real reach = PADDED_CUTOFF+margin2+marginBuffer[warpStart+j];
                            interacts |= (delta.x*delta.x+delta.y*delta.y+delta.z*delta.z < reach*reach ? 1<<j : 0);
#else
                            interacts |= (delta.x*delta.x+delta.y*delta.y+delta.z*delta.z < PADDED_CUTOFF_SQUARED ? 1<<j : 0);
#endif
                        }
                    }
                    else {
#endif
                        #pragma unroll
                        for (int j = 0; j < 32; j++) {
                            real4 posj = posBuffer[warpStart+j];
                            real halfDist2 = posj.w + pos2.w - posj.x*pos2.x - posj.y*pos2.y - posj.z*pos2.z;
#ifdef USE_ANCHORS
                            const real reach = PADDED_CUTOFF+margin2+marginBuffer[warpStart+j];
                            interacts |= (halfDist2 < 0.5f*reach*reach ? 1<<j : 0);
#else
                            interacts |= (halfDist2 < 0.5f * PADDED_CUTOFF_SQUARED ? 1<<j : 0);
#endif
                        }
#ifdef USE_PERIODIC
                    }
#endif
                }
                
                // Add any interacting atoms to the buffer.
                
                int includeAtomFlags = BALLOT(interacts);
                if (interacts) {
                    int index = neighborsInBuffer+__popc(includeAtomFlags&warpMask);
                    buffer[index] = atom2;
                    flagsBuffer[index] = interacts;
                }
                neighborsInBuffer += __popc(includeAtomFlags);
#ifdef USE_PRUNING
                if (neighborsInBuffer > BUFFER_SIZE-TILE_SIZE) {
                    appendToOuterList(x, buffer, neighborsInBuffer, interactionCount, interactingTiles, interactingAtoms, maxTiles);
                    neighborsInBuffer = 0;
                }
#else
                if (neighborsInBuffer > BUFFER_SIZE-TILE_SIZE) {
                    // Store the new tiles to memory.
                    
#if MAX_BITS_FOR_PAIRS > 0
                    neighborsInBuffer = saveSinglePairs(x, buffer, flagsBuffer, neighborsInBuffer, maxSinglePairs, &interactionCount[1], singlePairs, sumBuffer+warpStart, pairStartIndex);
#endif
                    unsigned int tilesToStore = neighborsInBuffer/TILE_SIZE;
                    if (tilesToStore > 0) {
                        if (indexInWarp == 0)
                            tileStartIndex = atomicAdd(&interactionCount[0], tilesToStore);
                        unsigned int newTileStartIndex = tileStartIndex;
                        if (newTileStartIndex+tilesToStore <= maxTiles) {
                            if (indexInWarp < tilesToStore)
                                interactingTiles[newTileStartIndex+indexInWarp] = x;
                            #pragma unroll 8 // (GROUP_SIZE / TILE_SIZE)
                            for (int j = 0; j < tilesToStore; j++)
                                interactingAtoms[(newTileStartIndex+j)*TILE_SIZE+indexInWarp] = buffer[indexInWarp+j*TILE_SIZE];
                        }
                        if (indexInWarp+TILE_SIZE*tilesToStore < BUFFER_SIZE)
                            buffer[indexInWarp] = buffer[indexInWarp+TILE_SIZE*tilesToStore];
                        neighborsInBuffer -= TILE_SIZE*tilesToStore;
                    }
                }
#endif
            }
        }
        
        // Compare any atoms left in the queue.

        if (queued > 0) {
            __syncwarp();
#ifdef BUILD_TIMING
            const long long compareStart = clock64();
#endif
            compareQueuedAtoms(queued, x, queue, posq, blockCenterX, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ,
                    posBuffer+warpStart,
#ifdef USE_ANCHORS
                    marginBuffer+warpStart,
#endif
                    buffer, flagsBuffer, neighborsInBuffer, interactionCount, interactingTiles, interactingAtoms, singlePairs,
                    maxTiles, maxSinglePairs, sumBuffer+warpStart, pairStartIndex, tileStartIndex);
#ifdef BUILD_TIMING
            compareTime += clock64()-compareStart;
#endif
        }

        // If we have a partially filled buffer,  store it to memory.
        
#ifdef USE_PRUNING
        if (neighborsInBuffer > 0)
            appendToOuterList(x, buffer, neighborsInBuffer, interactionCount, interactingTiles, interactingAtoms, maxTiles);
#else
#if MAX_BITS_FOR_PAIRS > 0
        if (neighborsInBuffer > 32)
            neighborsInBuffer = saveSinglePairs(x, buffer, flagsBuffer, neighborsInBuffer, maxSinglePairs, &interactionCount[1], singlePairs, sumBuffer+warpStart, pairStartIndex);
#endif
#ifdef SPLIT_BUILD
        // Store the full tiles, and leave the rest in the block's leftovers, so the segments of a block share their
        // partly filled tiles instead of each storing one.
        if (neighborsInBuffer > 0) {
            unsigned int tilesToStore = neighborsInBuffer/TILE_SIZE;
            if (tilesToStore > 0) {
                if (indexInWarp == 0)
                    tileStartIndex = atomicAdd(&interactionCount[0], tilesToStore);
                __syncwarp();
                unsigned int newTileStartIndex = tileStartIndex;
                if (newTileStartIndex+tilesToStore <= maxTiles) {
                    if (indexInWarp < tilesToStore)
                        interactingTiles[newTileStartIndex+indexInWarp] = x;
                    for (int j = 0; j < tilesToStore; j++)
                        interactingAtoms[(newTileStartIndex+j)*TILE_SIZE+indexInWarp] = buffer[indexInWarp+j*TILE_SIZE];
                }
            }
            const int remainder = neighborsInBuffer-TILE_SIZE*tilesToStore;
            if (remainder > 0) {
                unsigned int start = 0;
                if (indexInWarp == 0)
                    start = atomicAdd((unsigned int*) &leftoverCount[x], (unsigned int) remainder);
                start = __shfl_sync(0xffffffff, start, 0);
                if (indexInWarp < remainder && start+indexInWarp < LEFTOVER_CAPACITY)
                    leftoverAtoms[x*LEFTOVER_CAPACITY+start+indexInWarp] = buffer[TILE_SIZE*tilesToStore+indexInWarp];
            }
            __syncwarp();
        }

        // The warp that finishes the block's last segment packs the leftovers of all its segments into tiles, then
        // resets the counts for the next build.  Every segment of the block counts itself in segmentsDone, after a
        // fence that makes its leftovers visible to the warp that turns out to be last, which reads them with volatile
        // loads so they cannot come from a stale L1 line.

        __threadfence();
        __syncwarp();
        unsigned int done = 0;
        if (indexInWarp == 0)
            done = atomicAdd(&segmentsDone[x], 1u)+1;
        done = __shfl_sync(0xffffffff, done, 0);
        if (done == (NUM_BLOCKS-2-block1)/SEGMENT_BLOCKS+1) {
            __threadfence();
            const int count = min(((volatile int*) leftoverCount)[x], LEFTOVER_CAPACITY);
            for (int base = 0; base < count; base += TILE_SIZE) {
                unsigned int tile = 0;
                if (indexInWarp == 0)
                    tile = atomicAdd(&interactionCount[0], 1u);
                tile = __shfl_sync(0xffffffff, tile, 0);
                if (tile < maxTiles) {
                    if (indexInWarp == 0)
                        interactingTiles[tile] = x;
                    interactingAtoms[tile*TILE_SIZE+indexInWarp] = (base+indexInWarp < count ? ((volatile unsigned int*) leftoverAtoms)[x*LEFTOVER_CAPACITY+base+indexInWarp] : NUM_ATOMS);
                }
            }
            __syncwarp();
            if (indexInWarp == 0) {
                leftoverCount[x] = 0;
                segmentsDone[x] = 0;
            }
        }
#else
        if (neighborsInBuffer > 0) {
            unsigned int tilesToStore = (neighborsInBuffer+TILE_SIZE-1)/TILE_SIZE;
            if (indexInWarp == 0)
                tileStartIndex = atomicAdd(&interactionCount[0], tilesToStore);
            unsigned int newTileStartIndex = tileStartIndex;
            if (newTileStartIndex+tilesToStore <= maxTiles) {
                if (indexInWarp < tilesToStore)
                    interactingTiles[newTileStartIndex+indexInWarp] = x;
                #pragma unroll 8 // (GROUP_SIZE / TILE_SIZE)
                for (int j = 0; j < tilesToStore; j++)
                    interactingAtoms[(newTileStartIndex+j)*TILE_SIZE+indexInWarp] = (indexInWarp+j*TILE_SIZE < neighborsInBuffer ? buffer[indexInWarp+j*TILE_SIZE] : NUM_ATOMS);
            }
        }
#endif
#endif
#ifdef BUILD_TIMING
        // Diagnostic: cycles spent on this item, how many of them comparing queued atoms, the number of candidate
        // blocks, and the block's position in sorted order (negative if it is too large for one periodic copy).
        if (indexInWarp == 0) {
#if defined(USE_PRUNING) || defined(SPLIT_BUILD)
            const int timingIndex = item;
#else
            const int timingIndex = block1;
#endif
            buildTiming[4*timingIndex] = clock64()-itemStart;
            buildTiming[4*timingIndex+1] = compareTime;
            buildTiming[4*timingIndex+2] = numCandidates;
#ifdef USE_PERIODIC
            buildTiming[4*timingIndex+3] = (singlePeriodicCopy ? block1 : -1-block1);
#else
            buildTiming[4*timingIndex+3] = block1;
#endif
        }
#endif
    }
    
    // Record the positions the neighbor list is based on.
    
    for (int i = threadIdx.x+blockIdx.x*blockDim.x; i < NUM_ATOMS; i += blockDim.x*gridDim.x)
        oldPositions[i] = posq[i];
}

#ifdef USE_PRUNING
/**
 * The first half of building the neighbor list used to compute forces from the outer list built by
 * findBlocksWithInteractions.  The outer list holds every atom within the cutoff plus PADDING of some atom in each
 * block, and is only rebuilt once an atom has moved half of PADDING.  This finds the ones within the cutoff plus
 * INNER_PADDING at the current positions, with flags for the atoms of the block they interact with, and appends them to
 * the block's survivors.  Each block's outer atoms are split into segments of PRUNE_SEGMENT, each checked by its own
 * warp, so blocks with many neighbors don't make the kernel wait for one warp.  prunePack then packs them into tiles.
 */
extern "C" __global__ __launch_bounds__(GROUP_SIZE,3) void pruneFilter(real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        unsigned int* __restrict__ interactionCount, const real4* __restrict__ posq, const real4* __restrict__ blockCenter,
        const real4* __restrict__ blockBoundingBox, const unsigned int* __restrict__ outerCount, const int* __restrict__ outerBlockCounts,
        const unsigned int* __restrict__ outerAtoms, unsigned int outerCapacity, unsigned int* __restrict__ survivorAtoms,
        int* __restrict__ survivorFlags, int* __restrict__ survivorCounts, const int* __restrict__ rebuildNeighborList) {
    if (rebuildNeighborList[1] == 0)
        return; // The neighbor list doesn't need to be pruned.
    if (outerCount[0] > outerCapacity) {
        // Some block had more neighbors than the outer list has room for, so it is missing interactions.  Report an
        // overflow, so forces are computed without the list until the host has enlarged it.

        if (blockIdx.x == 0 && threadIdx.x == 0)
            interactionCount[0] = 0xFFFFFFFF;
        return;
    }
    const int indexInWarp = threadIdx.x%32;
    const int warpStart = threadIdx.x-indexInWarp;
    const int totalWarps = blockDim.x*gridDim.x/32;
    const int warpIndex = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int warpMask = (1<<indexInWarp)-1;
    __shared__ real4 posBuffer[GROUP_SIZE];
    const int numSegments = (outerCapacity+PRUNE_SEGMENT-1)/PRUNE_SEGMENT;
    for (int item = warpIndex; item < NUM_BLOCKS*numSegments; item += totalWarps) {
        const int x = item%NUM_BLOCKS;
        const int start = (item/NUM_BLOCKS)*PRUNE_SEGMENT;
        const int end = min(start+PRUNE_SEGMENT, outerBlockCounts[x]);
        if (start >= end)
            continue;
        // Load the atoms of this block.

        real4 blockCenterX = blockCenter[x];
        real4 pos1 = posq[x*TILE_SIZE+indexInWarp];
#ifdef USE_PERIODIC
        real4 blockSizeX = blockBoundingBox[x];
        const bool singlePeriodicCopy = (0.5f*periodicBoxSize.x-blockSizeX.x >= INNER_PADDED_CUTOFF &&
                                         0.5f*periodicBoxSize.y-blockSizeX.y >= INNER_PADDED_CUTOFF &&
                                         0.5f*periodicBoxSize.z-blockSizeX.z >= INNER_PADDED_CUTOFF);
        if (singlePeriodicCopy) {
            // The box is small enough that we can just translate all the atoms into a single periodic
            // box, then skip having to apply periodic boundary conditions later.

            APPLY_PERIODIC_TO_POS_WITH_CENTER(pos1, blockCenterX)
        }
#endif
        posBuffer[threadIdx.x] = pos1;
        __syncwarp();

        // Check which of this segment's outer neighbors are close enough to keep, and append them to the survivors.

        for (int base = start; base < end; base += 32) {
            int atom2 = 0;
            int interacts = 0;
            if (base+indexInWarp < end) {
                atom2 = outerAtoms[(size_t) x*outerCapacity+base+indexInWarp];
                real4 pos2 = posq[atom2];
#ifdef USE_PERIODIC
                if (!singlePeriodicCopy) {
                    for (int j = 0; j < 32; j++) {
                        real3 delta = trimTo3(pos2)-trimTo3(posBuffer[warpStart+j]);
                        APPLY_PERIODIC_TO_DELTA(delta)
                        interacts |= (delta.x*delta.x+delta.y*delta.y+delta.z*delta.z < INNER_PADDED_CUTOFF_SQUARED ? 1<<j : 0);
                    }
                }
                else {
                    APPLY_PERIODIC_TO_POS_WITH_CENTER(pos2, blockCenterX)
#endif
                    #pragma unroll
                    for (int j = 0; j < 32; j++) {
                        real3 delta = trimTo3(pos2)-trimTo3(posBuffer[warpStart+j]);
                        interacts |= (delta.x*delta.x+delta.y*delta.y+delta.z*delta.z < INNER_PADDED_CUTOFF_SQUARED ? 1<<j : 0);
                    }
#ifdef USE_PERIODIC
                }
#endif
            }

            int includeAtomFlags = BALLOT(interacts);
            if (includeAtomFlags != 0) {
                unsigned int first = 0;
                if (indexInWarp == 0)
                    first = atomicAdd((unsigned int*) &survivorCounts[x], (unsigned int) __popc(includeAtomFlags));
                first = __shfl_sync(0xffffffff, first, 0);
                if (interacts) {
                    size_t index = (size_t) x*outerCapacity+first+__popc(includeAtomFlags&warpMask);
                    survivorAtoms[index] = atom2;
                    survivorFlags[index] = interacts;
                }
            }
        }
        __syncwarp();
    }
}

/**
 * The second half of pruning: pack each block's survivors from pruneFilter into tiles and single pairs exactly as
 * findBlocksWithInteractions would, then reset the block's count of survivors.  Each warp handles one block at a
 * time, so every block's atoms are packed together.  This only moves data, so it is cheap even for blocks with many.
 */
extern "C" __global__ __launch_bounds__(GROUP_SIZE,3) void prunePack(unsigned int* __restrict__ interactionCount, int* __restrict__ interactingTiles,
        unsigned int* __restrict__ interactingAtoms, int2* __restrict__ singlePairs, const real4* __restrict__ posq, unsigned int maxTiles,
        unsigned int maxSinglePairs, const unsigned int* __restrict__ outerCount, unsigned int outerCapacity,
        const unsigned int* __restrict__ survivorAtoms, const int* __restrict__ survivorFlags, int* __restrict__ survivorCounts,
        real4* __restrict__ prunePositions, const int* __restrict__ rebuildNeighborList) {
    if (rebuildNeighborList[1] == 0)
        return; // The neighbor list doesn't need to be pruned.
    if (outerCount[0] > outerCapacity)
        return; // pruneFilter reported an overflow.
    const int indexInWarp = threadIdx.x%32;
    const int warpStart = threadIdx.x-indexInWarp;
    const int totalWarps = blockDim.x*gridDim.x/32;
    const int warpIndex = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    __shared__ int workgroupBuffer[BUFFER_SIZE*(GROUP_SIZE/32)];
    __shared__ int workgroupFlagsBuffer[BUFFER_SIZE*(GROUP_SIZE/32)];
    __shared__ volatile unsigned int workgroupTileIndex[GROUP_SIZE/32];
    __shared__ unsigned int workgroupPairStartIndex[GROUP_SIZE/32];
    int* buffer = workgroupBuffer+BUFFER_SIZE*(warpStart/32);
    int* flagsBuffer = workgroupFlagsBuffer+BUFFER_SIZE*(warpStart/32);
    volatile unsigned int& tileStartIndex = workgroupTileIndex[warpStart/32];
    volatile unsigned int& pairStartIndex = workgroupPairStartIndex[warpStart/32];
    for (int x = warpIndex; x < NUM_BLOCKS; x += totalWarps) {
        const int numSurvivors = survivorCounts[x];
        int neighborsInBuffer = 0;
        for (int base = 0; base < numSurvivors; base += 32) {
            if (base+indexInWarp < numSurvivors) {
                size_t index = (size_t) x*outerCapacity+base+indexInWarp;
                buffer[neighborsInBuffer+indexInWarp] = survivorAtoms[index];
                flagsBuffer[neighborsInBuffer+indexInWarp] = survivorFlags[index];
            }
            neighborsInBuffer += min(32, numSurvivors-base);
            __syncwarp();
            if (neighborsInBuffer > BUFFER_SIZE-TILE_SIZE) {
                // Store the new tiles to memory.

                __syncwarp();
#if MAX_BITS_FOR_PAIRS > 0
                neighborsInBuffer = saveSinglePairs(x, buffer, flagsBuffer, neighborsInBuffer, maxSinglePairs, &interactionCount[1], singlePairs, buffer, pairStartIndex);
#endif
                unsigned int tilesToStore = neighborsInBuffer/TILE_SIZE;
                if (tilesToStore > 0) {
                    if (indexInWarp == 0)
                        tileStartIndex = atomicAdd(&interactionCount[0], tilesToStore);
                    __syncwarp();
                    unsigned int newTileStartIndex = tileStartIndex;
                    if (newTileStartIndex+tilesToStore <= maxTiles) {
                        if (indexInWarp < tilesToStore)
                            interactingTiles[newTileStartIndex+indexInWarp] = x;
                        #pragma unroll 8 // (GROUP_SIZE / TILE_SIZE)
                        for (int j = 0; j < tilesToStore; j++)
                            interactingAtoms[(newTileStartIndex+j)*TILE_SIZE+indexInWarp] = buffer[indexInWarp+j*TILE_SIZE];
                    }
                    __syncwarp();
                    if (indexInWarp+TILE_SIZE*tilesToStore < BUFFER_SIZE)
                        buffer[indexInWarp] = buffer[indexInWarp+TILE_SIZE*tilesToStore];
                    neighborsInBuffer -= TILE_SIZE*tilesToStore;
                }
                __syncwarp();
            }
        }

        // Store whatever is left in the buffer.

        __syncwarp();
#if MAX_BITS_FOR_PAIRS > 0
        if (neighborsInBuffer > 32)
            neighborsInBuffer = saveSinglePairs(x, buffer, flagsBuffer, neighborsInBuffer, maxSinglePairs, &interactionCount[1], singlePairs, buffer, pairStartIndex);
#endif
        if (neighborsInBuffer > 0) {
            unsigned int tilesToStore = (neighborsInBuffer+TILE_SIZE-1)/TILE_SIZE;
            if (indexInWarp == 0)
                tileStartIndex = atomicAdd(&interactionCount[0], tilesToStore);
            __syncwarp();
            unsigned int newTileStartIndex = tileStartIndex;
            if (newTileStartIndex+tilesToStore <= maxTiles) {
                if (indexInWarp < tilesToStore)
                    interactingTiles[newTileStartIndex+indexInWarp] = x;
                #pragma unroll 8 // (GROUP_SIZE / TILE_SIZE)
                for (int j = 0; j < tilesToStore; j++)
                    interactingAtoms[(newTileStartIndex+j)*TILE_SIZE+indexInWarp] = (indexInWarp+j*TILE_SIZE < neighborsInBuffer ? buffer[indexInWarp+j*TILE_SIZE] : NUM_ATOMS);
            }
        }
        __syncwarp();
        if (indexInWarp == 0)
            survivorCounts[x] = 0;
    }

    // Record the positions the pruned list is based on.

    for (int i = threadIdx.x+blockIdx.x*blockDim.x; i < NUM_ATOMS; i += blockDim.x*gridDim.x)
        prunePositions[i] = posq[i];
}
#endif
