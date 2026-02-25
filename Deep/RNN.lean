import Deep.Basic
import Deep.Activation

noncomputable section

open scoped Matrix
open Real (tanh)

structure RNNCell (input hidden : ℕ) where
  σ : ℝ → ℝ
  Wh : Mat hidden hidden
  Wx : Mat hidden input
  b : Vec hidden

def RNNCell.step {input hidden : ℕ} (cell : RNNCell input hidden)
    (h : Vec hidden) (x : Vec input) : Vec hidden :=
  cell.σ ∘ (cell.Wh *ᵥ h + cell.Wx *ᵥ x + cell.b)

def RNNCell.fold {input hidden seqLen : ℕ}
    (cell : RNNCell input hidden) (h₀ : Vec hidden) (X : Mat seqLen input) : Vec hidden :=
  Fin.foldl seqLen (fun h i => cell.step h (X i)) h₀

def RNNCell.numParams (input hidden : ℕ) : ℕ :=
  hidden * hidden + hidden * input + hidden

structure LSTMState (hidden : ℕ) where
  h : Vec hidden
  c : Vec hidden

structure LSTMCell (input hidden : ℕ) where
  Wf : Mat hidden hidden
  Uf : Mat hidden input
  bf : Vec hidden
  Wi : Mat hidden hidden
  Ui : Mat hidden input
  bi : Vec hidden
  Wc : Mat hidden hidden
  Uc : Mat hidden input
  bc : Vec hidden
  Wo : Mat hidden hidden
  Uo : Mat hidden input
  bo : Vec hidden

def LSTMCell.step {input hidden : ℕ} (cell : LSTMCell input hidden)
    (s : LSTMState hidden) (x : Vec input) : LSTMState hidden :=
  let f := sigmoid ∘ (cell.Wf *ᵥ s.h + cell.Uf *ᵥ x + cell.bf)
  let i := sigmoid ∘ (cell.Wi *ᵥ s.h + cell.Ui *ᵥ x + cell.bi)
  let g := tanh ∘ (cell.Wc *ᵥ s.h + cell.Uc *ᵥ x + cell.bc)
  let o := sigmoid ∘ (cell.Wo *ᵥ s.h + cell.Uo *ᵥ x + cell.bo)
  let c' := f * s.c + i * g
  let h' := o * (tanh ∘ c')
  ⟨h', c'⟩

def LSTMCell.fold {input hidden seqLen : ℕ} (cell : LSTMCell input hidden)
    (s₀ : LSTMState hidden) (X : Mat seqLen input) : LSTMState hidden :=
  Fin.foldl seqLen (fun s i => cell.step s (X i)) s₀

def LSTMCell.numParams (input hidden : ℕ) : ℕ :=
  4 * (hidden * hidden + hidden * input + hidden)

end
