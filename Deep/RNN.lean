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

def RNNCell.scan {input hidden seqLen : ℕ}
    (cell : RNNCell input hidden) (h₀ : Vec hidden) (X : Mat seqLen input) : Mat seqLen hidden :=
  (Fin.foldl seqLen (fun (acc : Vec hidden × Mat seqLen hidden) i =>
    let h' := cell.step acc.1 (X i)
    (h', Function.update acc.2 i h'))
    (h₀, 0)).2

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

def LSTMCell.scan {input hidden seqLen : ℕ} (cell : LSTMCell input hidden)
    (s₀ : LSTMState hidden) (X : Mat seqLen input) : Mat seqLen hidden :=
  (Fin.foldl seqLen (fun (acc : LSTMState hidden × Mat seqLen hidden) i =>
    let s' := cell.step acc.1 (X i)
    (s', Function.update acc.2 i s'.h))
    (s₀, 0)).2

def LSTMCell.numParams (input hidden : ℕ) : ℕ :=
  4 * (hidden * hidden + hidden * input + hidden)

structure GRUCell (input hidden : ℕ) where
  Wz : Mat hidden hidden
  Uz : Mat hidden input
  bz : Vec hidden
  Wr : Mat hidden hidden
  Ur : Mat hidden input
  br : Vec hidden
  Wh : Mat hidden hidden
  Uh : Mat hidden input
  bh : Vec hidden

def GRUCell.step {input hidden : ℕ} (cell : GRUCell input hidden)
    (h : Vec hidden) (x : Vec input) : Vec hidden :=
  let z := sigmoid ∘ (cell.Wz *ᵥ h + cell.Uz *ᵥ x + cell.bz)
  let r := sigmoid ∘ (cell.Wr *ᵥ h + cell.Ur *ᵥ x + cell.br)
  let g := tanh ∘ (cell.Wh *ᵥ (r * h) + cell.Uh *ᵥ x + cell.bh)
  (1 - z) * h + z * g

def GRUCell.fold {input hidden seqLen : ℕ}
    (cell : GRUCell input hidden) (h₀ : Vec hidden) (X : Mat seqLen input) : Vec hidden :=
  Fin.foldl seqLen (fun h i => cell.step h (X i)) h₀

def GRUCell.scan {input hidden seqLen : ℕ}
    (cell : GRUCell input hidden) (h₀ : Vec hidden) (X : Mat seqLen input) : Mat seqLen hidden :=
  (Fin.foldl seqLen (fun (acc : Vec hidden × Mat seqLen hidden) i =>
    let h' := cell.step acc.1 (X i)
    (h', Function.update acc.2 i h'))
    (h₀, 0)).2

def GRUCell.numParams (input hidden : ℕ) : ℕ :=
  3 * (hidden * hidden + hidden * input + hidden)

structure RNN (input hidden : ℕ) (numLayers : ℕ) where
  first : RNNCell input hidden
  rest : Fin numLayers → RNNCell hidden hidden

def RNN.forward {input hidden numLayers seqLen : ℕ}
    (rnn : RNN input hidden numLayers) (h₀ : Fin (numLayers + 1) → Vec hidden)
    (X : Mat seqLen input) : Mat seqLen hidden :=
  Fin.foldl numLayers
    (fun H i => (rnn.rest i).scan (h₀ i.succ) H)
    (rnn.first.scan (h₀ 0) X)

def RNN.numParams (input hidden numLayers : ℕ) : ℕ :=
  RNNCell.numParams input hidden + numLayers * RNNCell.numParams hidden hidden

structure LSTM (input hidden : ℕ) (numLayers : ℕ) where
  first : LSTMCell input hidden
  rest : Fin numLayers → LSTMCell hidden hidden

def LSTM.forward {input hidden numLayers seqLen : ℕ}
    (lstm : LSTM input hidden numLayers) (s₀ : Fin (numLayers + 1) → LSTMState hidden)
    (X : Mat seqLen input) : Mat seqLen hidden :=
  Fin.foldl numLayers
    (fun H i => (lstm.rest i).scan (s₀ i.succ) H)
    (lstm.first.scan (s₀ 0) X)

def LSTM.numParams (input hidden numLayers : ℕ) : ℕ :=
  LSTMCell.numParams input hidden + numLayers * LSTMCell.numParams hidden hidden

structure GRU (input hidden : ℕ) (numLayers : ℕ) where
  first : GRUCell input hidden
  rest : Fin numLayers → GRUCell hidden hidden

def GRU.forward {input hidden numLayers seqLen : ℕ}
    (gru : GRU input hidden numLayers) (h₀ : Fin (numLayers + 1) → Vec hidden)
    (X : Mat seqLen input) : Mat seqLen hidden :=
  Fin.foldl numLayers
    (fun H i => (gru.rest i).scan (h₀ i.succ) H)
    (gru.first.scan (h₀ 0) X)

def GRU.numParams (input hidden numLayers : ℕ) : ℕ :=
  GRUCell.numParams input hidden + numLayers * GRUCell.numParams hidden hidden

end
