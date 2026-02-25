import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Data.Real.Basic

noncomputable section

open scoped Matrix

abbrev Mat (m n : ℕ) := Matrix (Fin m) (Fin n) ℝ
abbrev Vec (n : ℕ) := Fin n → ℝ

def res {α : Type*} [Add α] (f : α → α) (x : α) : α := x + f x

def mean {n : ℕ} [NeZero n] (x : Vec n) : ℝ := (∑ i, x i) / n

def variance {n : ℕ} [NeZero n] (x : Vec n) : ℝ :=
  (∑ i, (x i - mean x) ^ 2) / n

end
