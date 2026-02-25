import Deep.Basic
import Mathlib.Data.Real.Sqrt

noncomputable section

open Real (sqrt)

structure BatchNorm (n : ℕ) where
  γ : Vec n
  β : Vec n
  ε : ℝ

def BatchNorm.apply {batchSize n : ℕ} [NeZero batchSize] (bn : BatchNorm n)
    (X : Mat batchSize n) : Mat batchSize n :=
  let μ : Vec n := fun j => mean (fun i => X i j)
  let v : Vec n := fun j => variance (fun i => X i j)
  Matrix.of fun i j => bn.γ j * (X i j - μ j) / sqrt (v j + bn.ε) + bn.β j

def BatchNorm.numParams (n : ℕ) : ℕ := 2 * n

end
