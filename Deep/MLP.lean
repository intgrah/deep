import Deep.Affine

noncomputable section

open Matrix (of)

inductive MLP (σ : ℝ → ℝ) : ℕ → ℕ → Type where
  | single {input output : ℕ} : Affine output input → MLP σ input output
  | cons {input hidden output : ℕ} : Affine hidden input → MLP σ hidden output → MLP σ input output

def MLP.apply {σ : ℝ → ℝ} {input output : ℕ} : MLP σ input output → Vec input → Vec output
  | .single A => A.apply
  | .cons A rest => rest.apply ∘ (σ ∘ ·) ∘ A.apply

def MLP.applySeq {σ : ℝ → ℝ} {input output seqLen : ℕ} (mlp : MLP σ input output)
    (X : Mat seqLen input) : Mat seqLen output :=
  of (mlp.apply ∘ X)

def MLP.numParams {σ : ℝ → ℝ} {input output : ℕ} : MLP σ input output → ℕ
  | .single (input := i) (output := o) _ => Affine.numParams o i
  | .cons (input := i) (hidden := h) _ rest => Affine.numParams h i + rest.numParams

end
