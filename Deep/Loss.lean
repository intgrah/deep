import Deep.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic

noncomputable section

open Real (log)

def mse {n : ℕ} [NeZero n] (pred target : Vec n) : ℝ :=
  (∑ i, (pred i - target i) ^ 2) / n

def crossEntropy {n : ℕ} (pred target : Vec n) : ℝ :=
  -∑ i, target i * log (pred i)

end
