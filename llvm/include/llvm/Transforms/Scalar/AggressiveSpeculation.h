//===------- AggressiveSpeculation.h - Aggressive Speculation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides the interface for the AggressiveSpeculation pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_AGGRESSIVESPECULATION_H
#define LLVM_TRANSFORMS_SCALAR_AGGRESSIVESPECULATION_H

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class AggressiveSpeculationPass
    : public PassInfoMixin<AggressiveSpeculationPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_AGGRESSIVESPECULATION_H
