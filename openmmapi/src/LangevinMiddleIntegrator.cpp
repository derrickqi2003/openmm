/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2021 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/LangevinMiddleIntegrator.h"
#include "openmm/ConstantPotentialForce.h"
#include "openmm/Context.h"
#include "openmm/CustomCVForce.h"
#include "openmm/NonbondedForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/kernels.h"
#include <string>
#include <typeinfo>

using namespace OpenMM;
using std::string;
using std::vector;

LangevinMiddleIntegrator::LangevinMiddleIntegrator(double temperature, double frictionCoeff, double stepSize) {
    setTemperature(temperature);
    setFriction(frictionCoeff);
    setStepSize(stepSize);
    setConstraintTolerance(1e-5);
    setRandomNumberSeed(0);
    reciprocalSpaceInterval = 1;
    reciprocalSpaceIntervalChecked = false;
}

void LangevinMiddleIntegrator::initialize(ContextImpl& contextRef) {
    if (owner != NULL && &contextRef.getOwner() != owner)
        throw OpenMMException("This Integrator is already bound to a context");
    context = &contextRef;
    owner = &contextRef.getOwner();
    reciprocalSpaceIntervalChecked = false;
    kernel = context->getPlatform().createKernel(IntegrateLangevinMiddleStepKernel::Name(), contextRef);
    kernel.getAs<IntegrateLangevinMiddleStepKernel>().initialize(contextRef.getSystem(), *this);
}

void LangevinMiddleIntegrator::setTemperature(double temp) {
    if (temp < 0)
        throw OpenMMException("Temperature cannot be negative");
    temperature = temp;
}

void LangevinMiddleIntegrator::setFriction(double coeff) {
    if (coeff < 0)
        throw OpenMMException("Friction cannot be negative");
    friction = coeff;
}

void LangevinMiddleIntegrator::setReciprocalSpaceInterval(int interval) {
    if (interval < 1)
        throw OpenMMException("The reciprocal space interval must be at least 1");
    reciprocalSpaceInterval = interval;
    reciprocalSpaceIntervalChecked = false;
}

void LangevinMiddleIntegrator::checkReciprocalSpaceIntervalSupported() {
    // Only the NonbondedForce kernels of these platforms apply the reciprocal space scale.  Any other force with a
    // reciprocal space part would keep applying it every step, so the dynamics would be silently wrong.

    string platform = context->getPlatform().getName();
    if (platform != "Reference" && platform != "CUDA" && platform != "OpenCL" && platform != "HIP" && platform != "Metal")
        throw OpenMMException("LangevinMiddleIntegrator: a reciprocal space interval greater than 1 is not supported on the "+platform+" platform");
    const System& system = context->getSystem();
    for (int i = 0; i < system.getNumForces(); i++) {
        const Force& force = system.getForce(i);
        string type = typeid(force).name();
        bool unsupported = (dynamic_cast<const ConstantPotentialForce*>(&force) != NULL ||
                            type.find("AmoebaMultipoleForce") != string::npos || type.find("HippoNonbondedForce") != string::npos);
        const CustomCVForce* cv = dynamic_cast<const CustomCVForce*>(&force);
        if (cv != NULL) {
            for (int j = 0; j < cv->getNumCollectiveVariables(); j++) {
                const NonbondedForce* nb = dynamic_cast<const NonbondedForce*>(&cv->getCollectiveVariable(j));
                if (nb != NULL && (nb->getNonbondedMethod() == NonbondedForce::Ewald || nb->getNonbondedMethod() == NonbondedForce::PME || nb->getNonbondedMethod() == NonbondedForce::LJPME))
                    unsupported = true;
            }
        }
        if (unsupported)
            throw OpenMMException("LangevinMiddleIntegrator: a reciprocal space interval greater than 1 only supports reciprocal space computed by NonbondedForce");
    }
    reciprocalSpaceIntervalChecked = true;
}

void LangevinMiddleIntegrator::cleanup() {
    kernel = Kernel();
}

vector<string> LangevinMiddleIntegrator::getKernelNames() {
    return {IntegrateLangevinMiddleStepKernel::Name()};
}

double LangevinMiddleIntegrator::computeKineticEnergy() {
    return kernel.getAs<IntegrateLangevinMiddleStepKernel>().computeKineticEnergy(*context, *this);
}

bool LangevinMiddleIntegrator::kineticEnergyRequiresForce() const {
    return false;
}

void LangevinMiddleIntegrator::step(int steps) {
    if (context == NULL)
        throw OpenMMException("This Integrator is not bound to a context!");  
    if (reciprocalSpaceInterval > 1 && !reciprocalSpaceIntervalChecked)
        checkReciprocalSpaceIntervalSupported();
    for (int i = 0; i < steps; ++i) {
        context->updateContextState();
        if (reciprocalSpaceInterval > 1) {
            // Multiple time stepping: the slow forces act as an impulse on every n'th step.  The phase comes from the
            // step count, so it is the same however the steps are split between calls and across checkpoints.

            long long phase = context->getStepCount()%reciprocalSpaceInterval;
            context->setReciprocalSpaceForceScale(phase == 0 ? reciprocalSpaceInterval : 0.0);
            try {
                context->calcForcesAndEnergy(true, false, getIntegrationForceGroups());
            }
            catch (...) {
                context->setReciprocalSpaceForceScale(1.0);
                throw;
            }
            context->setReciprocalSpaceForceScale(1.0);
        }
        else
            context->calcForcesAndEnergy(true, false, getIntegrationForceGroups());
        kernel.getAs<IntegrateLangevinMiddleStepKernel>().execute(*context, *this);
    }
}
