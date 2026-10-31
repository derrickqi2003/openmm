/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2009-2025 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "openmm/OpenMMException.h"
#include "CudaNonbondedUtilities.h"
#include "CudaArray.h"
#include "CudaContext.h"
#include "CudaKernelSources.h"
#include "CudaExpressionUtilities.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <utility>

using namespace OpenMM;
using namespace std;

#define CHECK_RESULT(result) \
    if (result != CUDA_SUCCESS) { \
        std::stringstream m; \
        m<<errorMessage<<": "<<context.getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
        throw OpenMMException(m.str());\
    }


class CudaNonbondedUtilities::BlockSortTrait : public ComputeSortImpl::SortTrait {
public:
    BlockSortTrait() {}
    int getDataSize() const {return sizeof(int);}
    int getKeySize() const {return sizeof(int);}
    const char* getDataType() const {return "unsigned int";}
    const char* getKeyType() const {return "unsigned int";}
    const char* getMinKey() const {return "0";}
    const char* getMaxKey() const {return "0xFFFFFFFFu";}
    const char* getMaxValue() const {return "0xFFFFFFFFu";}
    const char* getSortKey() const {return "value";}
};

// With pruning, the search for each block's neighbors is split into pieces of this many blocks, and the pruning of its
// outer list into pieces of this many atoms, each handled by its own warp.
static const int SegmentBlocks = 128;
static const int PruneSegment = 128;

static double getEnvironmentDouble(const char* name, double defaultValue) {
    const char* value = getenv(name);
    if (value == NULL)
        return defaultValue;
    char* end;
    double result = strtod(value, &end);
    if (end == value || *end != 0 || result < 0)
        throw OpenMMException(string("Illegal value for ")+name+": "+value);
    return result;
}

CudaNonbondedUtilities::CudaNonbondedUtilities(CudaContext& context) : context(context), useCutoff(false), usePeriodic(false), useNeighborList(false), anyExclusions(false), usePadding(true),
        pinnedCountBuffer(NULL), forceRebuildNeighborList(true), groupFlags(0), canUsePairList(true), tilesAfterReorder(0),
        useLazyOverflowCheck(false), lazyCheckActive(false), countDownloadPending(false), useGatedBlockSort(false), useWarpBlockBounds(false), checkRebuildInBounds(false), useBuildPrefilter(true), usePruning(false),
        anchoringRequested(false), useAnchors(false), buildTimingRequested(false), splitSegment(0), buildTimingCalls(0), buildTimingReports(0),
        ordinaryPadding(-1.0), outerPadding(0.0), innerPadding(0.0), maxAnchorMargin(0.0), outerCapacity(0) {
    // A pruned neighbor list builds an outer list with a large padding, which only needs rebuilding once an atom has
    // moved half of it, and prunes it to the list used for forces whenever an atom has moved half of a much smaller
    // inner padding.  With an inner padding near zero that is every step, and the force kernel only gets atoms that
    // are really within the cutoff.  It is experimental, so it must be requested with an environment variable.  The
    // outer padding applies from the start, because other forces ask for the padded cutoff before initialize().
    //
    // Hydrogens constrained to heavy atoms move several times farther per step than the heavy atoms, so by default the
    // outer list follows each of them through its heavy atom (see findOuterBlockBounds).  That keeps it valid much
    // longer for the same size, so it gets a smaller default padding.

    // Settings for the ordinary list:
    //  - OPENMM_CUDA_PADDING: its padding in nm.  The default is 0.12 times the cutoff (upstream OpenMM uses 0.08).
    //    The list is rebuilt once any atom has moved half the padding, and with hydrogen mass repartitioning at 4 fs
    //    the fastest hydrogens move about 0.027 nm per step, so 0.08 nm means a rebuild every 2 steps and 0.12 nm one
    //    every 3.  That saves more than the extra pairs cost (measured on a 33,000 atom box on an A40).
    //  - OPENMM_CUDA_SPLIT_BUILD: the number of blocks each warp searches for a block's neighbors (default 256; 0 has
    //    one warp search all of them, as upstream does).  Each block only looks at the blocks after it, so the first
    //    blocks have far more work than the last, and the search lasts as long as the warp with the most.
    //  - OPENMM_CUDA_BUILD_TIMING=1: diagnostic timing of the search.

    if (getenv("OPENMM_CUDA_PADDING") != NULL)
        ordinaryPadding = getEnvironmentDouble("OPENMM_CUDA_PADDING", 0.0);
    splitSegment = (int) getEnvironmentDouble("OPENMM_CUDA_SPLIT_BUILD", 256.0);
    const char* growSetting = getenv("OPENMM_CUDA_GROW_NEIGHBOR_LIST");
    growNeighborListDisabled = (growSetting != NULL && string(growSetting) == "0");
    const char* timingSetting = getenv("OPENMM_CUDA_BUILD_TIMING");
    buildTimingRequested = (timingSetting != NULL && string(timingSetting) == "1");
    // In tiles from the neighbor list, each lane of the interaction kernel takes two atoms of the block (see
    // nonbonded.cu), unless OPENMM_CUDA_TWO_ATOMS_PER_LANE=0.
    const char* twoAtomsSetting = getenv("OPENMM_CUDA_TWO_ATOMS_PER_LANE");
    twoAtomsPerLane = (twoAtomsSetting == NULL || string(twoAtomsSetting) != "0");
    const char* pruneSetting = getenv("OPENMM_CUDA_PRUNED_NEIGHBOR_LIST");
    pruningRequested = (pruneSetting != NULL && string(pruneSetting) == "1");
    if (pruningRequested) {
        const char* anchorSetting = getenv("OPENMM_CUDA_ANCHORED_NEIGHBOR_LIST");
        anchoringRequested = (anchorSetting == NULL || string(anchorSetting) != "0");
        outerPadding = getEnvironmentDouble("OPENMM_CUDA_OUTER_PADDING", anchoringRequested ? 0.16 : 0.24);
        innerPadding = getEnvironmentDouble("OPENMM_CUDA_INNER_PADDING", 0.002);
    }

    // Decide how many thread blocks to use.

    string errorMessage = "Error initializing nonbonded utilities";
    int multiprocessors;
    CHECK_RESULT(cuDeviceGetAttribute(&multiprocessors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, context.getDevice()));
    CHECK_RESULT(cuEventCreate(&downloadCountEvent, context.getEventFlags()));
    CHECK_RESULT(cuMemHostAlloc((void**) &pinnedCountBuffer, 4*sizeof(unsigned int), CU_MEMHOSTALLOC_PORTABLE));
    numForceThreadBlocks = 4*multiprocessors;
    forceThreadBlockSize = (context.getComputeCapability() < 2.0 ? 128 : 256);
    
    // When building the neighbor list, we can optionally use large blocks (1024 atoms) to
    // accelerate the process.  This makes building the neighbor list faster, but it prevents
    // us from sorting atom blocks by size, which leads to a slightly less efficient neighbor
    // list.  We guess based on system size which will be faster.

    useLargeBlocks = (context.getNumAtoms() > 90000);
    setKernelSource(CudaKernelSources::nonbonded);
}

CudaNonbondedUtilities::~CudaNonbondedUtilities() {
    if (pinnedCountBuffer != NULL)
        cuMemFreeHost(pinnedCountBuffer);
    cuEventDestroy(downloadCountEvent);
}

void CudaNonbondedUtilities::addInteraction(bool usesCutoff, bool usesPeriodic, bool usesExclusions, double cutoffDistance,
        const vector<vector<int> >& exclusionList, const string& kernel, int forceGroup, bool useNeighborList, bool supportsPairList) {
    if (groupCutoff.size() > 0) {
        if (usesCutoff != useCutoff)
            throw OpenMMException("All Forces must agree on whether to use a cutoff");
        if (usesPeriodic != usePeriodic)
            throw OpenMMException("All Forces must agree on whether to use periodic boundary conditions");
        if (usesCutoff && groupCutoff.find(forceGroup) != groupCutoff.end() && groupCutoff[forceGroup] != cutoffDistance)
            throw OpenMMException("All Forces in a single force group must use the same cutoff distance");
    }
    if (usesExclusions)
        requestExclusions(exclusionList);
    useCutoff = usesCutoff;
    usePeriodic = usesPeriodic;
    this->useNeighborList |= (useNeighborList && useCutoff);
    groupCutoff[forceGroup] = cutoffDistance;
    groupFlags |= 1<<forceGroup;
    canUsePairList &= supportsPairList;
    if (kernel.size() > 0) {
        if (groupKernelSource.find(forceGroup) == groupKernelSource.end())
            groupKernelSource[forceGroup] = "";
        map<string, string> replacements;
        replacements["CUTOFF"] = "CUTOFF_"+context.intToString(forceGroup);
        replacements["CUTOFF_SQUARED"] = "CUTOFF_"+context.intToString(forceGroup)+"_SQUARED";
        groupKernelSource[forceGroup] += context.replaceStrings(kernel, replacements)+"\n";
    }
}

void CudaNonbondedUtilities::addParameter(ComputeParameterInfo parameter) {
    parameters.push_back(parameter);
}

void CudaNonbondedUtilities::addArgument(ComputeParameterInfo parameter) {
    arguments.push_back(parameter);
}

string CudaNonbondedUtilities::addEnergyParameterDerivative(const string& param) {
    // See if the parameter has already been added.
    
    int index;
    for (index = 0; index < energyParameterDerivatives.size(); index++)
        if (param == energyParameterDerivatives[index])
            break;
    if (index == energyParameterDerivatives.size())
        energyParameterDerivatives.push_back(param);
    context.addEnergyParameterDerivative(param);
    return string("energyParamDeriv")+context.intToString(index);
}

void CudaNonbondedUtilities::requestExclusions(const vector<vector<int> >& exclusionList) {
    if (anyExclusions) {
        bool sameExclusions = (exclusionList.size() == atomExclusions.size());
        for (int i = 0; i < (int) exclusionList.size() && sameExclusions; i++) {
             if (exclusionList[i].size() != atomExclusions[i].size())
                 sameExclusions = false;
            set<int> expectedExclusions;
            expectedExclusions.insert(atomExclusions[i].begin(), atomExclusions[i].end());
            for (int j = 0; j < (int) exclusionList[i].size(); j++)
                if (expectedExclusions.find(exclusionList[i][j]) == expectedExclusions.end())
                     sameExclusions = false;
        }
        if (!sameExclusions)
            throw OpenMMException("All Forces must have identical exceptions");
    }
    else {
        atomExclusions = exclusionList;
        anyExclusions = true;
    }
}

static bool compareInt2(int2 a, int2 b) {
    return ((a.y < b.y) || (a.y == b.y && a.x < b.x));
}

void CudaNonbondedUtilities::initialize(const System& system) {
    string errorMessage = "Error initializing nonbonded utilities";    
    if (atomExclusions.size() == 0) {
        // No exclusions were specifically requested, so just mark every atom as not interacting with itself.
        
        atomExclusions.resize(context.getNumAtoms());
        for (int i = 0; i < (int) atomExclusions.size(); i++)
            atomExclusions[i].push_back(i);
    }

    // Create the list of tiles.

    numAtoms = context.getNumAtoms();
    int numAtomBlocks = context.getNumAtomBlocks();
    int numContexts = context.getPlatformData().contexts.size();
    setAtomBlockRange(context.getContextIndex()/(double) numContexts, (context.getContextIndex()+1)/(double) numContexts);

    // Build a list of tiles that contain exclusions.

    set<pair<int, int> > tilesWithExclusions;
    for (int atom1 = 0; atom1 < (int) atomExclusions.size(); ++atom1) {
        int x = atom1/CudaContext::TileSize;
        for (int j = 0; j < (int) atomExclusions[atom1].size(); ++j) {
            int atom2 = atomExclusions[atom1][j];
            int y = atom2/CudaContext::TileSize;
            tilesWithExclusions.insert(make_pair(max(x, y), min(x, y)));
        }
    }
    vector<int2> exclusionTilesVec;
    for (set<pair<int, int> >::const_iterator iter = tilesWithExclusions.begin(); iter != tilesWithExclusions.end(); ++iter)
        exclusionTilesVec.push_back(make_int2(iter->first, iter->second));
    sort(exclusionTilesVec.begin(), exclusionTilesVec.end(), compareInt2);
    exclusionTiles.initialize<int2>(context, exclusionTilesVec.size(), "exclusionTiles");
    exclusionTiles.upload(exclusionTilesVec);
    map<pair<int, int>, int> exclusionTileMap;
    for (int i = 0; i < (int) exclusionTilesVec.size(); i++) {
        int2 tile = exclusionTilesVec[i];
        exclusionTileMap[make_pair(tile.x, tile.y)] = i;
    }
    vector<vector<int> > exclusionBlocksForBlock(numAtomBlocks);
    for (set<pair<int, int> >::const_iterator iter = tilesWithExclusions.begin(); iter != tilesWithExclusions.end(); ++iter) {
        exclusionBlocksForBlock[iter->first].push_back(iter->second);
        if (iter->first != iter->second)
            exclusionBlocksForBlock[iter->second].push_back(iter->first);
    }
    vector<unsigned int> exclusionRowIndicesVec(numAtomBlocks+1, 0);
    vector<unsigned int> exclusionIndicesVec;
    for (int i = 0; i < numAtomBlocks; i++) {
        exclusionIndicesVec.insert(exclusionIndicesVec.end(), exclusionBlocksForBlock[i].begin(), exclusionBlocksForBlock[i].end());
        exclusionRowIndicesVec[i+1] = exclusionIndicesVec.size();
    }
    maxExclusions = 0;
    for (int i = 0; i < (int) exclusionBlocksForBlock.size(); i++)
        maxExclusions = (maxExclusions > exclusionBlocksForBlock[i].size() ? maxExclusions : exclusionBlocksForBlock[i].size());
    exclusionIndices.initialize<unsigned int>(context, exclusionIndicesVec.size(), "exclusionIndices");
    exclusionRowIndices.initialize<unsigned int>(context, exclusionRowIndicesVec.size(), "exclusionRowIndices");
    exclusionIndices.upload(exclusionIndicesVec);
    exclusionRowIndices.upload(exclusionRowIndicesVec);

    // Record the exclusion data.

    exclusions.initialize<tileflags>(context, tilesWithExclusions.size()*CudaContext::TileSize, "exclusions");
    tileflags allFlags = (tileflags) -1;
    vector<tileflags> exclusionVec(exclusions.getSize(), allFlags);
    for (int atom1 = 0; atom1 < (int) atomExclusions.size(); ++atom1) {
        int x = atom1/CudaContext::TileSize;
        int offset1 = atom1-x*CudaContext::TileSize;
        for (int j = 0; j < (int) atomExclusions[atom1].size(); ++j) {
            int atom2 = atomExclusions[atom1][j];
            int y = atom2/CudaContext::TileSize;
            int offset2 = atom2-y*CudaContext::TileSize;
            if (x > y) {
                int index = exclusionTileMap[make_pair(x, y)]*CudaContext::TileSize;
                exclusionVec[index+offset1] &= allFlags-(1<<offset2);
            }
            else {
                int index = exclusionTileMap[make_pair(y, x)]*CudaContext::TileSize;
                exclusionVec[index+offset2] &= allFlags-(1<<offset1);
            }
        }
    }
    atomExclusions.clear(); // We won't use this again, so free the memory it used
    exclusions.upload(exclusionVec);

    // Create data structures for the neighbor list.

    maxCutoff = getMaxCutoffDistance();
    // Deferring the check for neighbor list overflow relies on a fallback kernel that enumerates every tile, so only
    // allow it when that is cheap enough to be an acceptable cost on the rare steps that need it.  The multi-GPU
    // code checks the counts itself.
    useLazyOverflowCheck = (useNeighborList && context.getPlatformData().contexts.size() == 1 && numAtomBlocks <= 4096);

    // Without pruning, a requested outer padding is still used for the ordinary list, so it stays consistent with the
    // padded cutoff other forces were given.

    usePruning = (pruningRequested && useNeighborList && usePadding && numContexts == 1 && numAtomBlocks <= 4096);
    if (usePruning || !useNeighborList || numContexts != 1 || numAtomBlocks <= splitSegment)
        splitSegment = 0;
    if (useCutoff) {
        // Select a size for the arrays that hold the neighbor list.  We have to make a fairly
        // arbitrary guess, but if this turns out to be too small we'll increase it later.

        maxTiles = 20*numAtomBlocks;
        if (maxTiles > numTiles)
            maxTiles = numTiles;
        if (maxTiles < 1)
            maxTiles = 1;
        maxSinglePairs = 5*numAtoms;
        interactingTiles.initialize<int>(context, maxTiles, "interactingTiles");
        interactingAtoms.initialize<int>(context, CudaContext::TileSize*maxTiles, "interactingAtoms");
        interactionCount.initialize<unsigned int>(context, 2, "interactionCount");
        singlePairs.initialize<int2>(context, maxSinglePairs, "singlePairs");
        int elementSize = (context.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
        blockCenter.initialize(context, numAtomBlocks, 4*elementSize, "blockCenter");
        blockBoundingBox.initialize(context, numAtomBlocks, 4*elementSize, "blockBoundingBox");
        sortedBlocks.initialize<unsigned int>(context, numAtomBlocks, "sortedBlocks");
        // For lists short enough that the generic sort would use a single rank sort kernel anyway, use one that
        // skips the work when the neighbor list doesn't need to be rebuilt.
        useGatedBlockSort = (numAtomBlocks <= CudaContext::ThreadBlockSize*context.getNumThreadBlocks());
        if (useGatedBlockSort)
            blockSortKeys.initialize<unsigned int>(context, numAtomBlocks, "blockSortKeys");
        // Bounding boxes are found with one warp per block, unless OPENMM_CUDA_WARP_BLOCK_BOUNDS=0 (for comparison).
        const char* warpSetting = getenv("OPENMM_CUDA_WARP_BLOCK_BOUNDS");
        useWarpBlockBounds = (warpSetting == NULL || string(warpSetting) != "0");
        // findBlockBoundsByWarp then also checks whether to rebuild the list, which saves launching
        // checkRebuildNeighborList.  The pruned list has more to check, so it keeps the separate kernel.
        checkRebuildInBounds = (useWarpBlockBounds && !usePruning);
        // findBlocksWithInteractions tests atoms against each block's bounding box before comparing them to all its
        // atoms, unless OPENMM_CUDA_BUILD_PREFILTER=0 (for comparison).  The list is the same either way.
        const char* prefilterSetting = getenv("OPENMM_CUDA_BUILD_PREFILTER");
        useBuildPrefilter = (prefilterSetting == NULL || string(prefilterSetting) != "0");
        sortedBlockCenter.initialize(context, numAtomBlocks+1, 4*elementSize, "sortedBlockCenter");
        sortedBlockBoundingBox.initialize(context, numAtomBlocks+1, 4*elementSize, "sortedBlockBoundingBox");
        // One range of sizes is recorded for each work group of findBlockBounds (64 threads, one per block) or
        // findBlockBoundsByWarp (128 threads, one warp per block).

        if (useWarpBlockBounds)
            numBlockSizes = min((context.getNumAtomBlocks()+3)/4, context.getNumThreadBlocks());
        else
            numBlockSizes = min((context.getNumAtomBlocks()+63)/64, context.getNumThreadBlocks());
        blockSizeRange.initialize(context, numBlockSizes, 2*elementSize, "blockSizeRange");
        if (checkRebuildInBounds)
            rebuildFlags.initialize<int>(context, numBlockSizes, "rebuildFlags");
        largeBlockCenter.initialize(context, numAtomBlocks, 4*elementSize, "largeBlockCenter");
        largeBlockBoundingBox.initialize(context, numAtomBlocks, 4*elementSize, "largeBlockBoundingBox");
        oldPositions.initialize(context, numAtoms, 4*elementSize, "oldPositions");
        rebuildNeighborList.initialize<int>(context, 2, "rebuildNeighborList");
        blockSorter = context.createSort(new BlockSortTrait(), numAtomBlocks, false);
        vector<unsigned int> count(2, 0);
        interactionCount.upload(count);
        rebuildNeighborList.upload(&count[0]);
        if (usePruning) {
            // Make a guess at the space each block needs in the outer list.  If it is too small, it is enlarged later.

            outerCapacity = min(numAtoms, 1024);
            outerAtoms.initialize<unsigned int>(context, (size_t) numAtomBlocks*outerCapacity, "outerAtoms");
            outerBlockCounts.initialize<int>(context, numAtomBlocks, "outerBlockCounts");
            outerCount.initialize<unsigned int>(context, 2, "outerCount");
            prunePositions.initialize(context, numAtoms, 4*elementSize, "prunePositions");
            survivorAtoms.initialize<unsigned int>(context, (size_t) numAtomBlocks*outerCapacity, "survivorAtoms");
            survivorFlags.initialize<int>(context, (size_t) numAtomBlocks*outerCapacity, "survivorFlags");
            survivorCounts.initialize<int>(context, numAtomBlocks, "survivorCounts");
            outerCount.upload(count);
            vector<int> zeros(numAtomBlocks, 0);
            survivorCounts.upload(zeros);
        }
        if (usePruning && anchoringRequested) {
            // Each atom follows the heaviest atom it is constrained to, if that is heavier than itself and does not
            // follow another atom.  Identical molecules have identical constraints, so reordering atoms does not change
            // the offsets.  Constraints are only satisfied to within a tolerance, so the margins are slightly larger
            // than the constrained distances.  An atom outside its margin is simply not anchored.

            vector<int> anchor(numAtoms);
            vector<double> distance(numAtoms, 0.0), anchorMass(numAtoms, 0.0);
            for (int i = 0; i < numAtoms; i++)
                anchor[i] = i;
            for (int i = 0; i < system.getNumConstraints(); i++) {
                int particle1, particle2;
                double dist;
                system.getConstraintParameters(i, particle1, particle2, dist);
                for (int k = 0; k < 2; k++) {
                    int atom = (k == 0 ? particle1 : particle2);
                    int other = (k == 0 ? particle2 : particle1);
                    double mass = system.getParticleMass(other);
                    if (mass > system.getParticleMass(atom) && mass > anchorMass[atom]) {
                        anchor[atom] = other;
                        distance[atom] = dist;
                        anchorMass[atom] = mass;
                    }
                }
            }
            vector<int> offsets(numAtoms, 0);
            vector<float> margins(numAtoms, 0.0f);
            for (int i = 0; i < numAtoms; i++) {
                if (anchor[i] != i && anchor[anchor[i]] == anchor[i]) {
                    offsets[i] = anchor[i]-i;
                    margins[i] = (float) (1.001*distance[i]);
                    maxAnchorMargin = max(maxAnchorMargin, (double) margins[i]);
                }
            }
            useAnchors = (maxAnchorMargin > 0.0);
            if (useAnchors) {
                anchorOffsets.initialize<int>(context, numAtoms, "anchorOffsets");
                anchorMargins.initialize<float>(context, numAtoms, "anchorMargins");
                outerPositions.initialize(context, context.getPaddedNumAtoms(), 4*elementSize, "outerPositions");
                outerBlockCenter.initialize(context, numAtomBlocks, 4*elementSize, "outerBlockCenter");
                outerBlockBoundingBox.initialize(context, numAtomBlocks, 4*elementSize, "outerBlockBoundingBox");
                anchorOffsets.upload(offsets);
                anchorMargins.upload(margins);
            }
        }
    }

    // Record arguments for kernels.

    forceArgs.push_back(&context.unwrap(context.getLongForceBuffer()).getDevicePointer());
    forceArgs.push_back(&context.unwrap(context.getEnergyBuffer()).getDevicePointer());
    forceArgs.push_back(&context.unwrap(context.getPosq()).getDevicePointer());
    forceArgs.push_back(&exclusions.getDevicePointer());
    forceArgs.push_back(&exclusionTiles.getDevicePointer());
    forceArgs.push_back(&startTileIndex);
    forceArgs.push_back(&numTiles);
    if (useCutoff) {
        forceArgs.push_back(&interactingTiles.getDevicePointer());
        forceArgs.push_back(&interactionCount.getDevicePointer());
        forceArgs.push_back(context.getPeriodicBoxSizePointer());
        forceArgs.push_back(context.getInvPeriodicBoxSizePointer());
        forceArgs.push_back(context.getPeriodicBoxVecXPointer());
        forceArgs.push_back(context.getPeriodicBoxVecYPointer());
        forceArgs.push_back(context.getPeriodicBoxVecZPointer());
        forceArgs.push_back(&maxTiles);
        forceArgs.push_back(&blockCenter.getDevicePointer());
        forceArgs.push_back(&blockBoundingBox.getDevicePointer());
        forceArgs.push_back(&interactingAtoms.getDevicePointer());
        forceArgs.push_back(&maxSinglePairs);
        forceArgs.push_back(&singlePairs.getDevicePointer());
    }
    hasInitializedParams = false;
    paramStartIndex = forceArgs.size();
    for (int i = 0; i < parameters.size()+arguments.size(); i++)
        forceArgs.push_back(NULL);
    if (energyParameterDerivatives.size() > 0)
        forceArgs.push_back(&context.unwrap(context.getEnergyParamDerivBuffer()).getDevicePointer());
    if (useCutoff) {
        findBlockBoundsArgs.push_back(&numAtoms);
        findBlockBoundsArgs.push_back(context.getPeriodicBoxSizePointer());
        findBlockBoundsArgs.push_back(context.getInvPeriodicBoxSizePointer());
        findBlockBoundsArgs.push_back(context.getPeriodicBoxVecXPointer());
        findBlockBoundsArgs.push_back(context.getPeriodicBoxVecYPointer());
        findBlockBoundsArgs.push_back(context.getPeriodicBoxVecZPointer());
        findBlockBoundsArgs.push_back(&context.unwrap(context.getPosq()).getDevicePointer());
        findBlockBoundsArgs.push_back(&blockCenter.getDevicePointer());
        findBlockBoundsArgs.push_back(&blockBoundingBox.getDevicePointer());
        findBlockBoundsArgs.push_back(&rebuildNeighborList.getDevicePointer());
        findBlockBoundsArgs.push_back(&blockSizeRange.getDevicePointer());
        if (checkRebuildInBounds) {
            findBlockBoundsArgs.push_back(&oldPositions.getDevicePointer());
            findBlockBoundsArgs.push_back(&rebuildFlags.getDevicePointer());
        }
        checkRebuildArgs.push_back(&context.unwrap(context.getPosq()).getDevicePointer());
        checkRebuildArgs.push_back(&oldPositions.getDevicePointer());
        checkRebuildArgs.push_back(&interactionCount.getDevicePointer());
        checkRebuildArgs.push_back(&rebuildNeighborList.getDevicePointer());
        checkRebuildArgs.push_back(&forceRebuildNeighborList);
        if (usePruning) {
            checkRebuildArgs.push_back(&prunePositions.getDevicePointer());
            checkRebuildArgs.push_back(&outerCount.getDevicePointer());
        }
        if (useAnchors) {
            checkRebuildArgs.push_back(context.getPeriodicBoxSizePointer());
            checkRebuildArgs.push_back(context.getInvPeriodicBoxSizePointer());
            checkRebuildArgs.push_back(context.getPeriodicBoxVecXPointer());
            checkRebuildArgs.push_back(context.getPeriodicBoxVecYPointer());
            checkRebuildArgs.push_back(context.getPeriodicBoxVecZPointer());
            checkRebuildArgs.push_back(&anchorOffsets.getDevicePointer());
        }
        computeSortKeysArgs.push_back(&blockBoundingBox.getDevicePointer());
        computeSortKeysArgs.push_back(useGatedBlockSort ? &blockSortKeys.getDevicePointer() : &sortedBlocks.getDevicePointer());
        computeSortKeysArgs.push_back(&blockSizeRange.getDevicePointer());
        computeSortKeysArgs.push_back(&numBlockSizes);
        computeSortKeysArgs.push_back(&rebuildNeighborList.getDevicePointer());
        if (checkRebuildInBounds) {
            computeSortKeysArgs.push_back(&rebuildFlags.getDevicePointer());
            computeSortKeysArgs.push_back(&interactionCount.getDevicePointer());
            computeSortKeysArgs.push_back(&forceRebuildNeighborList);
        }
        if (useAnchors) {
            outerBoundsArgs.push_back(context.getPeriodicBoxSizePointer());
            outerBoundsArgs.push_back(context.getInvPeriodicBoxSizePointer());
            outerBoundsArgs.push_back(context.getPeriodicBoxVecXPointer());
            outerBoundsArgs.push_back(context.getPeriodicBoxVecYPointer());
            outerBoundsArgs.push_back(context.getPeriodicBoxVecZPointer());
            outerBoundsArgs.push_back(&context.unwrap(context.getPosq()).getDevicePointer());
            outerBoundsArgs.push_back(&anchorOffsets.getDevicePointer());
            outerBoundsArgs.push_back(&anchorMargins.getDevicePointer());
            outerBoundsArgs.push_back(&outerPositions.getDevicePointer());
            outerBoundsArgs.push_back(&outerBlockCenter.getDevicePointer());
            outerBoundsArgs.push_back(&outerBlockBoundingBox.getDevicePointer());
            outerBoundsArgs.push_back(&rebuildNeighborList.getDevicePointer());
        }
        if (useGatedBlockSort) {
            sortBlockKeysArgs.push_back(&blockSortKeys.getDevicePointer());
            sortBlockKeysArgs.push_back(&sortedBlocks.getDevicePointer());
            sortBlockKeysArgs.push_back(&rebuildNeighborList.getDevicePointer());
        }
        sortBoxDataArgs.push_back(&sortedBlocks.getDevicePointer());
        // With anchors, the outer list is built from the bounding boxes of the anchored positions.

        sortBoxDataArgs.push_back(useAnchors ? &outerBlockCenter.getDevicePointer() : &blockCenter.getDevicePointer());
        sortBoxDataArgs.push_back(useAnchors ? &outerBlockBoundingBox.getDevicePointer() : &blockBoundingBox.getDevicePointer());
        sortBoxDataArgs.push_back(&sortedBlockCenter.getDevicePointer());
        sortBoxDataArgs.push_back(&sortedBlockBoundingBox.getDevicePointer());
        if (useLargeBlocks) {
            sortBoxDataArgs.push_back(&largeBlockCenter.getDevicePointer());
            sortBoxDataArgs.push_back(&largeBlockBoundingBox.getDevicePointer());
            sortBoxDataArgs.push_back(context.getPeriodicBoxSizePointer());
            sortBoxDataArgs.push_back(context.getInvPeriodicBoxSizePointer());
            sortBoxDataArgs.push_back(context.getPeriodicBoxVecXPointer());
            sortBoxDataArgs.push_back(context.getPeriodicBoxVecYPointer());
            sortBoxDataArgs.push_back(context.getPeriodicBoxVecZPointer());
        }
        sortBoxDataArgs.push_back(&rebuildNeighborList.getDevicePointer());
        if (usePruning)
            sortBoxDataArgs.push_back(&outerBlockCounts.getDevicePointer());
        findInteractingBlocksArgs.push_back(context.getPeriodicBoxSizePointer());
        findInteractingBlocksArgs.push_back(context.getInvPeriodicBoxSizePointer());
        findInteractingBlocksArgs.push_back(context.getPeriodicBoxVecXPointer());
        findInteractingBlocksArgs.push_back(context.getPeriodicBoxVecYPointer());
        findInteractingBlocksArgs.push_back(context.getPeriodicBoxVecZPointer());
        // With pruning, that kernel builds the outer list, so these arguments have the meanings described there.

        findInteractingBlocksArgs.push_back(usePruning ? &outerCount.getDevicePointer() : &interactionCount.getDevicePointer());
        findInteractingBlocksArgs.push_back(usePruning ? &outerBlockCounts.getDevicePointer() : &interactingTiles.getDevicePointer());
        findInteractingBlocksArgs.push_back(usePruning ? &outerAtoms.getDevicePointer() : &interactingAtoms.getDevicePointer());
        findInteractingBlocksArgs.push_back(&singlePairs.getDevicePointer());
        findInteractingBlocksArgs.push_back(useAnchors ? &outerPositions.getDevicePointer() : &context.unwrap(context.getPosq()).getDevicePointer());
        findInteractingBlocksArgs.push_back(usePruning ? &outerCapacity : &maxTiles);
        findInteractingBlocksArgs.push_back(&maxSinglePairs);
        findInteractingBlocksArgs.push_back(&startBlockIndex);
        findInteractingBlocksArgs.push_back(&numBlocks);
        findInteractingBlocksArgs.push_back(&sortedBlocks.getDevicePointer());
        findInteractingBlocksArgs.push_back(&sortedBlockCenter.getDevicePointer());
        findInteractingBlocksArgs.push_back(&sortedBlockBoundingBox.getDevicePointer());
        if (useLargeBlocks) {
            findInteractingBlocksArgs.push_back(&largeBlockCenter.getDevicePointer());
            findInteractingBlocksArgs.push_back(&largeBlockBoundingBox.getDevicePointer());
        }
        findInteractingBlocksArgs.push_back(&exclusionIndices.getDevicePointer());
        findInteractingBlocksArgs.push_back(&exclusionRowIndices.getDevicePointer());
        findInteractingBlocksArgs.push_back(&oldPositions.getDevicePointer());
        findInteractingBlocksArgs.push_back(&rebuildNeighborList.getDevicePointer());
        if (splitSegment > 0) {
            int numSegments = (numAtomBlocks+splitSegment-1)/splitSegment;
            leftoverAtoms.initialize<unsigned int>(context, (size_t) numAtomBlocks*numSegments*32, "leftoverAtoms");
            leftoverCount.initialize<int>(context, numAtomBlocks, "leftoverCount");
            segmentsDone.initialize<unsigned int>(context, numAtomBlocks, "segmentsDone");
            vector<int> zeros(numAtomBlocks, 0);
            leftoverCount.upload(zeros);
            segmentsDone.upload(zeros.data());
            findInteractingBlocksArgs.push_back(&leftoverAtoms.getDevicePointer());
            findInteractingBlocksArgs.push_back(&leftoverCount.getDevicePointer());
            findInteractingBlocksArgs.push_back(&segmentsDone.getDevicePointer());
        }
        if (buildTimingRequested) {
            int segment = (usePruning ? SegmentBlocks : splitSegment);
            int items = numAtomBlocks*(segment > 0 ? (numAtomBlocks+segment-1)/segment : 1);
            buildTiming.initialize<long long>(context, 4*items, "buildTiming");
            findInteractingBlocksArgs.push_back(&buildTiming.getDevicePointer());
        }
        if (usePruning) {
            // The indices of these arguments are used in updateNeighborListSize().

            pruneFilterArgs.push_back(context.getPeriodicBoxSizePointer());
            pruneFilterArgs.push_back(context.getInvPeriodicBoxSizePointer());
            pruneFilterArgs.push_back(context.getPeriodicBoxVecXPointer());
            pruneFilterArgs.push_back(context.getPeriodicBoxVecYPointer());
            pruneFilterArgs.push_back(context.getPeriodicBoxVecZPointer());
            pruneFilterArgs.push_back(&interactionCount.getDevicePointer());
            pruneFilterArgs.push_back(&context.unwrap(context.getPosq()).getDevicePointer());
            pruneFilterArgs.push_back(&blockCenter.getDevicePointer());
            pruneFilterArgs.push_back(&blockBoundingBox.getDevicePointer());
            pruneFilterArgs.push_back(&outerCount.getDevicePointer());
            pruneFilterArgs.push_back(&outerBlockCounts.getDevicePointer());
            pruneFilterArgs.push_back(&outerAtoms.getDevicePointer());          // 11
            pruneFilterArgs.push_back(&outerCapacity);
            pruneFilterArgs.push_back(&survivorAtoms.getDevicePointer());       // 13
            pruneFilterArgs.push_back(&survivorFlags.getDevicePointer());       // 14
            pruneFilterArgs.push_back(&survivorCounts.getDevicePointer());
            pruneFilterArgs.push_back(&rebuildNeighborList.getDevicePointer());
            prunePackArgs.push_back(&interactionCount.getDevicePointer());
            prunePackArgs.push_back(&interactingTiles.getDevicePointer());      // 1
            prunePackArgs.push_back(&interactingAtoms.getDevicePointer());      // 2
            prunePackArgs.push_back(&singlePairs.getDevicePointer());           // 3
            prunePackArgs.push_back(&context.unwrap(context.getPosq()).getDevicePointer());
            prunePackArgs.push_back(&maxTiles);
            prunePackArgs.push_back(&maxSinglePairs);
            prunePackArgs.push_back(&outerCount.getDevicePointer());
            prunePackArgs.push_back(&outerCapacity);
            prunePackArgs.push_back(&survivorAtoms.getDevicePointer());         // 9
            prunePackArgs.push_back(&survivorFlags.getDevicePointer());         // 10
            prunePackArgs.push_back(&survivorCounts.getDevicePointer());
            prunePackArgs.push_back(&prunePositions.getDevicePointer());
            prunePackArgs.push_back(&rebuildNeighborList.getDevicePointer());
        }
    }
}

double CudaNonbondedUtilities::getMaxCutoffDistance() {
    double cutoff = 0.0;
    for (map<int, double>::const_iterator iter = groupCutoff.begin(); iter != groupCutoff.end(); ++iter)
        cutoff = max(cutoff, iter->second);
    return cutoff;
}

double CudaNonbondedUtilities::padCutoff(double cutoff) {
    double padding = (usePadding ? (pruningRequested ? outerPadding : (ordinaryPadding >= 0 ? ordinaryPadding : 0.12*cutoff)) : 0.0);
    return cutoff+padding;
}

void CudaNonbondedUtilities::prepareInteractions(int forceGroups) {
    if ((forceGroups&groupFlags) == 0)
        return;
    if (groupKernels.find(forceGroups) == groupKernels.end())
        createKernelsForGroups(forceGroups);
    KernelSet& kernels = groupKernels[forceGroups];
    if (useCutoff && usePeriodic) {
        double4 box = context.getPeriodicBoxSize();
        double minAllowedSize = 1.999999*maxCutoff;
        if (box.x < minAllowedSize || box.y < minAllowedSize || box.z < minAllowedSize)
            throw OpenMMException("The periodic box size has decreased to less than twice the nonbonded cutoff.");
    }
    if (!useNeighborList)
        return;
    if (numTiles == 0)
        return;

    // Compute the neighbor list.

    if (useWarpBlockBounds)
        context.executeKernel(kernels.findBlockBoundsKernel, &findBlockBoundsArgs[0], numBlockSizes*128, 128);
    else
        context.executeKernel(kernels.findBlockBoundsKernel, &findBlockBoundsArgs[0], context.getNumAtomBlocks());
    if (!checkRebuildInBounds)
        context.executeKernel(kernels.checkRebuildKernel, &checkRebuildArgs[0], context.getNumAtoms());
    if (useAnchors)
        context.executeKernel(kernels.outerBoundsKernel, &outerBoundsArgs[0], context.getNumAtomBlocks()*32, 128);
    context.executeKernel(kernels.computeSortKeysKernel, &computeSortKeysArgs[0], context.getNumAtomBlocks());
    if (useGatedBlockSort)
        context.executeKernel(kernels.sortBlockKeysKernel, &sortBlockKeysArgs[0], context.getNumAtomBlocks()*32, 128);
    else
        blockSorter->sort(sortedBlocks);
    context.executeKernel(kernels.sortBoxDataKernel, &sortBoxDataArgs[0], context.getNumAtoms());
    if (usePruning) {
        // These kernels split each block's work over several warps (see the kernels), so launch enough warps for
        // all the pieces.

        int numSegments = (context.getNumAtomBlocks()+SegmentBlocks-1)/SegmentBlocks;
        int numPruneSegments = (outerCapacity+PruneSegment-1)/PruneSegment;
        context.executeKernel(kernels.findInteractingBlocksKernel, &findInteractingBlocksArgs[0], context.getNumAtomBlocks()*numSegments*32, 256);
        context.executeKernel(kernels.pruneFilterKernel, &pruneFilterArgs[0], context.getNumAtomBlocks()*numPruneSegments*32, 256);
        context.executeKernel(kernels.prunePackKernel, &prunePackArgs[0], context.getNumAtomBlocks()*32, 256);
    }
    else if (splitSegment > 0) {
        int numSegments = (context.getNumAtomBlocks()+splitSegment-1)/splitSegment;
        context.executeKernel(kernels.findInteractingBlocksKernel, &findInteractingBlocksArgs[0], context.getNumAtomBlocks()*numSegments*32, 256);
    }
    else
        context.executeKernel(kernels.findInteractingBlocksKernel, &findInteractingBlocksArgs[0], context.getNumAtoms(), 256);
    if (buildTimingRequested && buildTimingReports < 3 && ++buildTimingCalls%500 == 0)
        reportBuildTiming();
    forceRebuildNeighborList = false;
    if (!lazyCheckActive || !countDownloadPending) {
        interactionCount.download(pinnedCountBuffer, false);
        if (usePruning)
            outerCount.download(pinnedCountBuffer+2, false);
        cuEventRecord(downloadCountEvent, context.getCurrentStream());
        countDownloadPending = true;
    }
}

void CudaNonbondedUtilities::initParamArgs() {
    int index = paramStartIndex;
    for (ComputeParameterInfo& param : parameters)
        forceArgs[index++] = &context.unwrap(param.getArray()).getDevicePointer();
    for (ComputeParameterInfo& arg : arguments)
        forceArgs[index++] = &context.unwrap(arg.getArray()).getDevicePointer();
    hasInitializedParams = true;
}

void CudaNonbondedUtilities::computeInteractions(int forceGroups, bool includeForces, bool includeEnergy) {
    if ((forceGroups&groupFlags) == 0)
        return;
    KernelSet& kernels = groupKernels[forceGroups];
    if (kernels.hasForces && (includeForces || includeEnergy)) {
        CUfunction& kernel = (includeForces ? (includeEnergy ? kernels.forceEnergyKernel : kernels.forceKernel) : kernels.energyKernel);
        if (kernel == NULL)
            kernel = createInteractionKernel(kernels.source, parameters, arguments, true, true, forceGroups, includeForces, includeEnergy);
        if (!hasInitializedParams)
            initParamArgs();
        context.executeKernel(kernel, &forceArgs[0], numForceThreadBlocks*forceThreadBlockSize, forceThreadBlockSize);
        if (lazyCheckActive && numTiles > 0) {
            // Computes the interactions on the GPU if the neighbor list overflowed, and otherwise returns immediately.
            CUfunction& fallback = (includeForces ? (includeEnergy ? kernels.fallbackForceEnergyKernel : kernels.fallbackForceKernel) : kernels.fallbackEnergyKernel);
            if (fallback == NULL)
                fallback = createInteractionKernel(kernels.source, parameters, arguments, true, true, forceGroups, includeForces, includeEnergy, true);
            context.executeKernel(fallback, &forceArgs[0], numForceThreadBlocks*forceThreadBlockSize, forceThreadBlockSize);
        }
    }
    if (useNeighborList && numTiles > 0) {
        if (lazyCheckActive) {
            // The forces are correct even if the neighbor list overflowed, so there is no need to wait for the GPU.
            // Check the counts whenever a download has completed, and enlarge the arrays if necessary.
            if (countDownloadPending && cuEventQuery(downloadCountEvent) == CUDA_SUCCESS) {
                countDownloadPending = false;
                updateNeighborListSize();
            }
        }
        else {
            cuEventSynchronize(downloadCountEvent);
            countDownloadPending = false;
            if (!updateNeighborListSize() && useLazyOverflowCheck)
                lazyCheckActive = true; // The neighbor list fits, so later checks can be deferred.
        }
    }
}

bool CudaNonbondedUtilities::updateNeighborListSize() {
    if (!useCutoff)
        return false;

    // If the outer list overflowed, the pruned list was not built and its counts mean nothing.

    bool outerOverflow = (usePruning && pinnedCountBuffer[2] > outerCapacity);
    if (!outerOverflow) {
        if (context.getStepsSinceReorder() == 0 || tilesAfterReorder == 0)
            tilesAfterReorder = pinnedCountBuffer[0];
        else if (context.getStepsSinceReorder() > 25 && pinnedCountBuffer[0] > 1.1*tilesAfterReorder)
            context.forceReorder();
        if (pinnedCountBuffer[0] <= maxTiles && pinnedCountBuffer[1] <= maxSinglePairs) {
            // The list fits.  With the lazy check, an overflow is only noticed when the counts for some later step
            // have been downloaded, and the GPU computes every tile for each step queued until then, which costs far
            // more than synchronizing once.  So when the list is nearly full, enlarge the arrays before it overflows.

            bool nearlyFull = (pinnedCountBuffer[0] > 0.9*maxTiles || pinnedCountBuffer[1] > 0.9*maxSinglePairs);
            if (!lazyCheckActive || !nearlyFull || growNeighborListDisabled)
                return false;
            string errorMessage = "Error synchronizing before enlarging the neighbor list";
            CHECK_RESULT(cuCtxSynchronize());
            enlargeNeighborList((unsigned int) (1.5*pinnedCountBuffer[0]), (unsigned int) (1.5*pinnedCountBuffer[1]));
            forceRebuildNeighborList = true;
            return false;
        }
    }

    // The most recent timestep had too many interactions to fit in the arrays.  Make the arrays bigger to prevent
    // this from happening in the future.  If we didn't wait for the counts, kernels using the current arrays may
    // still be queued, so let them finish before replacing the arrays.

    if (lazyCheckActive) {
        string errorMessage = "Error synchronizing before resizing the neighbor list";
        CHECK_RESULT(cuCtxSynchronize());
    }

    if (outerOverflow) {
        outerCapacity = (unsigned int) min((double) numAtoms, 1.2*pinnedCountBuffer[2]);
        outerAtoms.resize((size_t) context.getNumAtomBlocks()*outerCapacity);
        survivorAtoms.resize((size_t) context.getNumAtomBlocks()*outerCapacity);
        survivorFlags.resize((size_t) context.getNumAtomBlocks()*outerCapacity);
        findInteractingBlocksArgs[7] = &outerAtoms.getDevicePointer();
        pruneFilterArgs[11] = &outerAtoms.getDevicePointer();
        pruneFilterArgs[13] = &survivorAtoms.getDevicePointer();
        pruneFilterArgs[14] = &survivorFlags.getDevicePointer();
        prunePackArgs[9] = &survivorAtoms.getDevicePointer();
        prunePackArgs[10] = &survivorFlags.getDevicePointer();
    }
    if (!outerOverflow)
        enlargeNeighborList((unsigned int) (1.2*pinnedCountBuffer[0]), (unsigned int) (1.2*pinnedCountBuffer[1]));
    forceRebuildNeighborList = true;
    if (lazyCheckActive) {
        // The fallback kernel already computed correct forces.  Check synchronously again until the neighbor
        // list has been built without overflowing.
        lazyCheckActive = false;
    }
    else
        context.setForcesValid(false);
    return true;
}

void CudaNonbondedUtilities::enlargeNeighborList(unsigned int tiles, unsigned int pairs) {
    // Make the arrays for tiles and single pairs at least this large.  CudaArray::resize() keeps the address of the
    // device pointer, which is what the kernel arguments hold, but they are set again for clarity.

    if (tiles > maxTiles) {
        maxTiles = tiles;
        unsigned int numBlocks = context.getNumAtomBlocks();
        int totalTiles = numBlocks*(numBlocks+1)/2;
        if (maxTiles > totalTiles)
            maxTiles = totalTiles;
        interactingTiles.resize(maxTiles);
        interactingAtoms.resize(CudaContext::TileSize*(size_t) maxTiles);
        if (forceArgs.size() > 0)
            forceArgs[7] = &interactingTiles.getDevicePointer();
        if (forceArgs.size() > 0)
            forceArgs[17] = &interactingAtoms.getDevicePointer();
        if (usePruning) {
            prunePackArgs[1] = &interactingTiles.getDevicePointer();
            prunePackArgs[2] = &interactingAtoms.getDevicePointer();
        }
        else {
            findInteractingBlocksArgs[6] = &interactingTiles.getDevicePointer();
            findInteractingBlocksArgs[7] = &interactingAtoms.getDevicePointer();
        }
    }
    if (pairs > maxSinglePairs) {
        maxSinglePairs = pairs;
        singlePairs.resize(maxSinglePairs);
        if (forceArgs.size() > 0)
            forceArgs[19] = &singlePairs.getDevicePointer();
        findInteractingBlocksArgs[8] = &singlePairs.getDevicePointer();
        if (usePruning)
            prunePackArgs[3] = &singlePairs.getDevicePointer();
    }
}

void CudaNonbondedUtilities::reportBuildTiming() {
    // Diagnostic: summarize the cycles each warp spent in findBlocksWithInteractions.  The array only has data if the
    // list was rebuilt on this step, so otherwise try again on the next one.

    vector<long long> t;
    buildTiming.download(t);
    vector<int> items;
    for (int i = 0; i < (int) t.size()/4; i++)
        if (t[4*i] > 0)
            items.push_back(i);
    if (items.size() == 0) {
        buildTimingCalls--;
        return;
    }
    int khz;
    cuDeviceGetAttribute(&khz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, context.getDevice());
    double usPerCycle = 1000.0/khz;
    double sumTotal = 0, sumCompare = 0, sumCand = 0, sxx = 0, sxy = 0, maxTotal = 0;
    int heaviest = items[0], numLarge = 0;
    for (int i : items) {
        double total = t[4*i]*usPerCycle, cand = (double) t[4*i+2];
        sumTotal += total;
        sumCompare += t[4*i+1]*usPerCycle;
        sumCand += cand;
        if (total > maxTotal) {
            maxTotal = total;
            heaviest = i;
        }
        if (t[4*i+3] < 0)
            numLarge++;
    }
    double n = items.size(), meanCand = sumCand/n, meanTotal = sumTotal/n;
    for (int i : items) {
        double dc = t[4*i+2]-meanCand;
        sxx += dc*dc;
        sxy += dc*(t[4*i]*usPerCycle-meanTotal);
    }
    double slope = (sxx > 0 ? sxy/sxx : 0);
    vector<double> totals;
    for (int i : items)
        totals.push_back(t[4*i]*usPerCycle);
    sort(totals.begin(), totals.end());
    printf("BUILD TIMING: %d warp items (%d too large for one periodic copy); per item mean %.1f us, 90th %.1f, 99th %.1f, max %.1f us\n",
            (int) n, numLarge, meanTotal, totals[(int) (0.9*(n-1))], totals[(int) (0.99*(n-1))], maxTotal);
    printf("BUILD TIMING: mean comparing queued atoms %.1f us, mean candidates %.1f; fit total = %.1f us + %.3f us per candidate\n",
            sumCompare/n, meanCand, meanTotal-slope*meanCand, slope);
    printf("BUILD TIMING: heaviest item: %.1f us, comparing %.1f us, %lld candidates, sorted position %lld%s\n",
            maxTotal, t[4*heaviest+1]*usPerCycle, t[4*heaviest+2], t[4*heaviest+3] < 0 ? -1-t[4*heaviest+3] : t[4*heaviest+3],
            t[4*heaviest+3] < 0 ? " (too large for one periodic copy)" : "");
    fflush(stdout);
    buildTimingReports++;
}

void CudaNonbondedUtilities::setUsePadding(bool padding) {
    usePadding = padding;
}

void CudaNonbondedUtilities::setAtomBlockRange(double startFraction, double endFraction) {
    int numAtomBlocks = context.getNumAtomBlocks();
    startBlockIndex = (int) (startFraction*numAtomBlocks);
    numBlocks = (int) (endFraction*numAtomBlocks)-startBlockIndex;
    long long totalTiles = context.getNumAtomBlocks()*((long long)context.getNumAtomBlocks()+1)/2;
    startTileIndex = (int) (startFraction*totalTiles);
    numTiles = (long long) (endFraction*totalTiles)-startTileIndex;
    forceRebuildNeighborList = true;
}

void CudaNonbondedUtilities::createKernelsForGroups(int groups) {
    KernelSet kernels;
    string source;
    for (int i = 0; i < 32; i++) {
        if ((groups&(1<<i)) != 0) {
            source += groupKernelSource[i];
        }
    }
    kernels.hasForces = (source.size() > 0);
    kernels.source = source;
    kernels.forceKernel = kernels.energyKernel = kernels.forceEnergyKernel = NULL;
    kernels.fallbackForceKernel = kernels.fallbackEnergyKernel = kernels.fallbackForceEnergyKernel = NULL;
    kernels.pruneFilterKernel = kernels.prunePackKernel = kernels.outerBoundsKernel = NULL;
    if (useCutoff) {
        double paddedCutoff = padCutoff(maxCutoff);
        map<string, string> defines;
        defines["TILE_SIZE"] = context.intToString(CudaContext::TileSize);
        defines["NUM_BLOCKS"] = context.intToString(context.getNumAtomBlocks());
        defines["NUM_ATOMS"] = context.intToString(context.getNumAtoms());
        defines["PADDING"] = context.doubleToString(paddedCutoff-maxCutoff);
        defines["PADDED_CUTOFF"] = context.doubleToString(paddedCutoff);
        defines["PADDED_CUTOFF_SQUARED"] = context.doubleToString(paddedCutoff*paddedCutoff);
        defines["NUM_TILES_WITH_EXCLUSIONS"] = context.intToString(exclusionTiles.getSize());
        if (usePruning) {
            double innerCutoff = maxCutoff+innerPadding;
            defines["USE_PRUNING"] = "1";
            defines["INNER_PADDING"] = context.doubleToString(innerPadding);
            defines["INNER_PADDED_CUTOFF"] = context.doubleToString(innerCutoff);
            defines["INNER_PADDED_CUTOFF_SQUARED"] = context.doubleToString(innerCutoff*innerCutoff);
            defines["SEGMENT_BLOCKS"] = context.intToString(SegmentBlocks);
            defines["PRUNE_SEGMENT"] = context.intToString(PruneSegment);
        }
        if (splitSegment > 0) {
            defines["SPLIT_BUILD"] = "1";
            defines["SEGMENT_BLOCKS"] = context.intToString(splitSegment);
            defines["LEFTOVER_CAPACITY"] = context.intToString(32*((context.getNumAtomBlocks()+splitSegment-1)/splitSegment));
        }
        if (buildTimingRequested)
            defines["BUILD_TIMING"] = "1";
        if (checkRebuildInBounds)
            defines["CHECK_REBUILD_IN_BOUNDS"] = "1";
        if (!useBuildPrefilter)
            defines["DISABLE_BUILD_PREFILTER"] = "1";
        if (useAnchors) {
            defines["USE_ANCHORS"] = "1";
            defines["MAX_ANCHOR_MARGIN"] = context.doubleToString(maxAnchorMargin);
        }
        if (usePeriodic)
            defines["USE_PERIODIC"] = "1";
        if (context.getBoxIsTriclinic())
            defines["TRICLINIC"] = "1";
        if (useLargeBlocks)
            defines["USE_LARGE_BLOCKS"] = "1";
        defines["MAX_EXCLUSIONS"] = context.intToString(maxExclusions);
        defines["MAX_BITS_FOR_PAIRS"] = (canUsePairList ? (context.getComputeCapability() < 8.0 ? "2" : "3") : "0");
        int binShift = 1;
        while (1<<binShift <= context.getNumAtomBlocks())
            binShift++;
        defines["BIN_SHIFT"] = context.intToString(binShift);
        defines["BLOCK_INDEX_MASK"] = context.intToString((1<<binShift)-1);
        CUmodule interactingBlocksProgram = context.createModule(CudaKernelSources::vectorOps+CudaKernelSources::findInteractingBlocks, defines);
        kernels.findBlockBoundsKernel = context.getKernel(interactingBlocksProgram, useWarpBlockBounds ? "findBlockBoundsByWarp" : "findBlockBounds");
        kernels.checkRebuildKernel = context.getKernel(interactingBlocksProgram, "checkRebuildNeighborList");
        kernels.computeSortKeysKernel = context.getKernel(interactingBlocksProgram, "computeSortKeys");
        kernels.sortBlockKeysKernel = context.getKernel(interactingBlocksProgram, "sortBlockKeys");
        kernels.sortBoxDataKernel = context.getKernel(interactingBlocksProgram, "sortBoxData");
        kernels.findInteractingBlocksKernel = context.getKernel(interactingBlocksProgram, "findBlocksWithInteractions");
        if (usePruning) {
            kernels.pruneFilterKernel = context.getKernel(interactingBlocksProgram, "pruneFilter");
            kernels.prunePackKernel = context.getKernel(interactingBlocksProgram, "prunePack");
        }
        if (useAnchors)
            kernels.outerBoundsKernel = context.getKernel(interactingBlocksProgram, "findOuterBlockBoundsKernel");
    }
    groupKernels[groups] = kernels;
}

CUfunction CudaNonbondedUtilities::createInteractionKernel(const string& source, vector<ComputeParameterInfo>& params, vector<ComputeParameterInfo>& arguments, bool useExclusions, bool isSymmetric, int groups, bool includeForces, bool includeEnergy, bool overflowFallback) {
    map<string, string> replacements;
    replacements["COMPUTE_INTERACTION"] = source;
    const string suffixes[] = {"x", "y", "z", "w"};
    stringstream localData;
    int localDataSize = 0;
    for (const ComputeParameterInfo& param : params) {
        if (param.getNumComponents() == 1)
            localData<<param.getType()<<" "<<param.getName()<<";\n";
        else {
            for (int j = 0; j < param.getNumComponents(); ++j)
                localData<<param.getComponentType()<<" "<<param.getName()<<"_"<<suffixes[j]<<";\n";
        }
        localDataSize += param.getSize();
    }
    replacements["ATOM_PARAMETER_DATA"] = localData.str();
    stringstream args;
    for (const ComputeParameterInfo& param : params) {
        args << ", ";
        if (param.isConstant())
            args << "const ";
        args << param.getType();
        args << "* __restrict__ global_";
        args << param.getName();
    }
    for (const ComputeParameterInfo& arg : arguments) {
        args << ", ";
        if (arg.isConstant())
            args << "const ";
        args << arg.getType();
        args << "* __restrict__ ";
        args << arg.getName();
    }
    if (energyParameterDerivatives.size() > 0)
        args << ", mixed* __restrict__ energyParamDerivs";
    replacements["PARAMETER_ARGUMENTS"] = args.str();

    stringstream load1;
    for (const ComputeParameterInfo& param : params) {
        load1 << param.getType();
        load1 << " ";
        load1 << param.getName();
        load1 << "1 = global_";
        load1 << param.getName();
        load1 << "[atom1];\n";
    }
    replacements["LOAD_ATOM1_PARAMETERS"] = load1.str();

    // Part 1. Defines for on diagonal exclusion tiles

    stringstream broadcastWarpData;
    broadcastWarpData << "posq2.x = real_shfl(shflPosq.x, j);\n";
    broadcastWarpData << "posq2.y = real_shfl(shflPosq.y, j);\n";
    broadcastWarpData << "posq2.z = real_shfl(shflPosq.z, j);\n";
    broadcastWarpData << "posq2.w = real_shfl(shflPosq.w, j);\n";
    for (const ComputeParameterInfo& param : params) {
        broadcastWarpData << param.getType() << " shfl" << param.getName() << ";\n";
        for (int j = 0; j < param.getNumComponents(); j++) {
            if (param.getNumComponents() == 1)
                broadcastWarpData << "shfl" << param.getName() << "=real_shfl(" << param.getName() <<"1,j);\n";
            else
                broadcastWarpData << "shfl" << param.getName()+"."+suffixes[j] << "=real_shfl(" << param.getName()+"1."+suffixes[j] <<",j);\n";
        }
    }
    replacements["BROADCAST_WARP_DATA"] = broadcastWarpData.str();
    
    // Part 2. Defines for off-diagonal exclusions, and neighborlist tiles. 
    stringstream declareLocal2;
    for (const ComputeParameterInfo& param : params)
        declareLocal2<<param.getType()<<" shfl"<<param.getName()<<";\n";
    replacements["DECLARE_LOCAL_PARAMETERS"] = declareLocal2.str();

    stringstream loadLocal2;
    for (const ComputeParameterInfo& param : params)
        loadLocal2<<"shfl"<<param.getName()<<" = global_"<<param.getName()<<"[j];\n";
    replacements["LOAD_LOCAL_PARAMETERS_FROM_GLOBAL"] = loadLocal2.str();
   
    stringstream load2j;
    for (const ComputeParameterInfo& param : params)
        load2j<<param.getType()<<" "<<param.getName()<<"2 = shfl"<<param.getName()<<";\n";
    replacements["LOAD_ATOM2_PARAMETERS"] = load2j.str();

    stringstream load2g;
    for (const ComputeParameterInfo& param : params)
        load2g<<param.getType()<<" "<<param.getName()<<"2 = global_"<<param.getName()<<"[atom2];\n";
    replacements["LOAD_ATOM2_PARAMETERS_FROM_GLOBAL"] = load2g.str();
    
    stringstream clearLocal;
    for (const ComputeParameterInfo& param : params) {
        clearLocal<<"shfl"<<param.getName()<<" = ";
        if (param.getNumComponents() == 1)
            clearLocal<<"0;\n";
        else
            clearLocal<<"make_"<<param.getType()<<"(0);\n";
    }
    replacements["CLEAR_LOCAL_PARAMETERS"] = clearLocal.str();

    stringstream initDerivs;
    for (int i = 0; i < energyParameterDerivatives.size(); i++)
        initDerivs<<"mixed energyParamDeriv"<<i<<" = 0;\n";
    replacements["INIT_DERIVATIVES"] = initDerivs.str();
    stringstream saveDerivs;
    const vector<string>& allParamDerivNames = context.getEnergyParamDerivNames();
    int numDerivs = allParamDerivNames.size();
    for (int i = 0; i < energyParameterDerivatives.size(); i++)
        for (int index = 0; index < numDerivs; index++)
            if (allParamDerivNames[index] == energyParameterDerivatives[i])
                saveDerivs<<"energyParamDerivs[GLOBAL_ID*"<<numDerivs<<"+"<<index<<"] += energyParamDeriv"<<i<<";\n";
    replacements["SAVE_DERIVATIVES"] = saveDerivs.str();

    stringstream shuffleWarpData;
    shuffleWarpData << "shflPosq.x = real_shfl(shflPosq.x, tgx+1);\n";
    shuffleWarpData << "shflPosq.y = real_shfl(shflPosq.y, tgx+1);\n";
    shuffleWarpData << "shflPosq.z = real_shfl(shflPosq.z, tgx+1);\n";
    shuffleWarpData << "shflPosq.w = real_shfl(shflPosq.w, tgx+1);\n";
    shuffleWarpData << "shflForce.x = real_shfl(shflForce.x, tgx+1);\n";
    shuffleWarpData << "shflForce.y = real_shfl(shflForce.y, tgx+1);\n";
    shuffleWarpData << "shflForce.z = real_shfl(shflForce.z, tgx+1);\n";
    for (const ComputeParameterInfo& param : params) {
        if (param.getNumComponents() == 1)
            shuffleWarpData<<"shfl"<<param.getName()<<"=real_shfl(shfl"<<param.getName()<<", tgx+1);\n";
        else {
            for (int j = 0; j < param.getNumComponents(); j++) {
                // looks something like shflsigmaEpsilon.x = real_shfl(shflsigmaEpsilon.x,tgx+1);
                shuffleWarpData<<"shfl"<<param.getName()
                    <<"."<<suffixes[j]<<"=real_shfl(shfl"
                    <<param.getName()<<"."<<suffixes[j]
                    <<", tgx+1);\n";
            }
        }
    }
    replacements["SHUFFLE_WARP_DATA"] = shuffleWarpData.str();

    // The same, passing data around each half of the warp, and the parameters of the atom held by the lane 16 away
    // (see TWO_ATOMS_PER_LANE in nonbonded.cu).

    string shuffleHalfWarpData = shuffleWarpData.str();
    size_t start;
    while ((start = shuffleHalfWarpData.find("tgx+1)")) != string::npos)
        shuffleHalfWarpData.replace(start, 6, "rotateSource)");
    replacements["SHUFFLE_HALF_WARP_DATA"] = shuffleHalfWarpData;
    stringstream loadPartner, usePartner;
    for (const ComputeParameterInfo& param : params) {
        loadPartner<<param.getType()<<" "<<param.getName()<<"Partner;\n";
        if (param.getNumComponents() == 1)
            loadPartner<<param.getName()<<"Partner = real_shfl("<<param.getName()<<"1, partnerLane);\n";
        else
            for (int j = 0; j < param.getNumComponents(); j++)
                loadPartner<<param.getName()<<"Partner."<<suffixes[j]<<" = real_shfl("<<param.getName()<<"1."<<suffixes[j]<<", partnerLane);\n";
        usePartner<<param.getType()<<" "<<param.getName()<<"1 = "<<param.getName()<<"Partner;\n";
    }
    replacements["LOAD_PARTNER_PARAMETERS"] = loadPartner.str();
    replacements["USE_PARTNER_PARAMETERS"] = usePartner.str();

    map<string, string> defines;
    if (useCutoff)
        defines["USE_CUTOFF"] = "1";
    if (usePeriodic)
        defines["USE_PERIODIC"] = "1";
    if (useExclusions)
        defines["USE_EXCLUSIONS"] = "1";
    if (isSymmetric)
        defines["USE_SYMMETRIC"] = "1";
    if (overflowFallback)
        defines["OVERFLOW_FALLBACK"] = "1";
    else if (useNeighborList)
        defines["USE_NEIGHBOR_LIST"] = "1";
    if (twoAtomsPerLane)
        defines["TWO_ATOMS_PER_LANE"] = "1";
    defines["ENABLE_SHUFFLE"] = "1";
    if (includeForces)
        defines["INCLUDE_FORCES"] = "1";
    if (includeEnergy)
        defines["INCLUDE_ENERGY"] = "1";
    defines["THREAD_BLOCK_SIZE"] = context.intToString(forceThreadBlockSize);
    double maxCutoff = 0.0;
    for (int i = 0; i < 32; i++) {
        if ((groups&(1<<i)) != 0) {
            double cutoff = groupCutoff[i];
            maxCutoff = max(maxCutoff, cutoff);
            defines["CUTOFF_"+context.intToString(i)+"_SQUARED"] = context.doubleToString(cutoff*cutoff);
            defines["CUTOFF_"+context.intToString(i)] = context.doubleToString(cutoff);
        }
    }
    defines["MAX_CUTOFF"] = context.doubleToString(maxCutoff);
    defines["NUM_ATOMS"] = context.intToString(context.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = context.intToString(context.getPaddedNumAtoms());
    defines["NUM_BLOCKS"] = context.intToString(context.getNumAtomBlocks());
    defines["TILE_SIZE"] = context.intToString(CudaContext::TileSize);
    int numExclusionTiles = exclusionTiles.getSize();
    defines["NUM_TILES_WITH_EXCLUSIONS"] = context.intToString(numExclusionTiles);
    int numContexts = context.getPlatformData().contexts.size();
    int startExclusionIndex = context.getContextIndex()*numExclusionTiles/numContexts;
    int endExclusionIndex = (context.getContextIndex()+1)*numExclusionTiles/numContexts;
    defines["FIRST_EXCLUSION_TILE"] = context.intToString(startExclusionIndex);
    defines["LAST_EXCLUSION_TILE"] = context.intToString(endExclusionIndex);
    if ((localDataSize/4)%2 == 0 && !context.getUseDoublePrecision())
        defines["PARAMETER_SIZE_IS_EVEN"] = "1";
    CUmodule program = context.createModule(CudaKernelSources::vectorOps+context.replaceStrings(kernelSource, replacements), defines);
    CUfunction kernel = context.getKernel(program, "computeNonbonded");
    return kernel;
}

void CudaNonbondedUtilities::setKernelSource(const string& source) {
    kernelSource = source;
}
