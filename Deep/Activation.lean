import Deep.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic

noncomputable section

open Real (sqrt tanh pi exp)

def relu (x : ℝ) : ℝ := max 0 x

def Φ (x : ℝ) : ℝ := 1/2 + ∫ t in 0..x, exp (-t^2 / 2) / sqrt (2 * pi)

def gelu (x : ℝ) : ℝ := x * Φ x

def gelu' (x : ℝ) : ℝ := x / 2 * (1 + tanh (sqrt (2 / pi) * (x + 0.044715 * x ^ 3)))

def sigmoid (x : ℝ) : ℝ := 1 / (1 + exp (-x))

def softmax {n : ℕ} (x : Vec n) : Vec n := fun i => exp (x i) / ∑ j, exp (x j)

end
