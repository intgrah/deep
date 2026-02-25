import Deep.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

noncomputable section

open Real (sqrt)

class Weights (W : Type) extends AddCommGroup W, Module ℝ W where
  zipWith : (ℝ → ℝ → ℝ) → W → W → W

structure Optimizer (W S : Type) where
  init : S
  step : W → W → StateM S W

def gd [AddCommGroup W] [Module ℝ W] (η : ℝ) : Optimizer W Unit where
  init := ()
  step g w := do
    return w - η • g

def sgd [AddCommGroup W] [Module ℝ W] (η μ : ℝ) : Optimizer W W where
  init := 0
  step g w := do
    let v ← get
    let v' := μ • v + g
    set v'
    return w - η • v'

structure AdamState (W : Type) where
  m : W
  v : W
  t : ℕ

open Weights in
def adam [Weights W] (η β₁ β₂ ε : ℝ) : Optimizer W (AdamState W) where
  init := ⟨0, 0, 0⟩
  step g w := do
    let ⟨m, v, t⟩ ← get
    let m' := β₁ • m + (1 - β₁) • g
    let v' := β₂ • v + (1 - β₂) • zipWith (· * ·) g g
    let t' := t + 1
    let mCorr := (1 / (1 - β₁ ^ t')) • m'
    let vCorr := (1 / (1 - β₂ ^ t')) • v'
    set (⟨m', v', t'⟩ : AdamState W)
    return w - η • zipWith (fun a b => a / (sqrt b + ε)) mCorr vCorr

open Weights in
def adamW [Weights W] (η β₁ β₂ ε wd : ℝ) : Optimizer W (AdamState W) where
  init := ⟨0, 0, 0⟩
  step g w := do
    let ⟨m, v, t⟩ ← get
    let m' := β₁ • m + (1 - β₁) • g
    let v' := β₂ • v + (1 - β₂) • zipWith (· * ·) g g
    let t' := t + 1
    let mCorr := (1 / (1 - β₁ ^ t')) • m'
    let vCorr := (1 / (1 - β₂ ^ t')) • v'
    set (⟨m', v', t'⟩ : AdamState W)
    return (1 - η * wd) • w - η • zipWith (fun a b => a / (sqrt b + ε)) mCorr vCorr

end
