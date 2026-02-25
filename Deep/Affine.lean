import Deep.Basic

noncomputable section

open scoped Matrix

structure Affine (m n : ℕ) where
  W : Mat m n
  b : Vec m

def Affine.apply {m n : ℕ} (A : Affine m n) (x : Vec n) : Vec m :=
  A.W *ᵥ x + A.b

def Affine.applySeq {m n seqLen : ℕ} (A : Affine m n) (X : Mat seqLen n) : Mat seqLen m :=
  Matrix.of (A.apply ∘ X)

def Affine.numParams (m n : ℕ) : ℕ := m * n + m

end
