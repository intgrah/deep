import Deep.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

noncomputable section

open Real (sqrt)

def mean {n : ℕ} [NeZero n] (x : Vec n) : ℝ := (∑ i, x i) / n

def variance {n : ℕ} [NeZero n] (x : Vec n) : ℝ :=
  (∑ i, (x i - mean x) ^ 2) / n

structure LayerNorm (n : ℕ) where
  γ : Vec n
  β : Vec n
  ε : ℝ

def LayerNorm.apply {n : ℕ} [NeZero n] (ln : LayerNorm n) (x : Vec n) : Vec n :=
  fun i => ln.γ i * (x i - mean x) / sqrt (variance x + ln.ε) + ln.β i

def LayerNorm.applySeq {n seqLen : ℕ} [NeZero n] (ln : LayerNorm n)
    (X : Mat seqLen n) : Mat seqLen n :=
  Matrix.of fun i => ln.apply (X i)

def LayerNorm.numParams (n : ℕ) : ℕ := 2 * n

end
