import Deep.Affine
import Deep.Activation

noncomputable section

open scoped Matrix
open Real (tanh)

structure RNNCell (input hidden : ℕ) where
  σ : ℝ → ℝ
  A : Affine hidden (hidden + input)

def RNNCell.step {input hidden : ℕ} (cell : RNNCell input hidden)
    (h : Vec hidden) (x : Vec input) : Vec hidden :=
  cell.σ ∘ cell.A.apply (Fin.addCases h x)

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
  Affine.numParams hidden (hidden + input)

structure LSTMState (hidden : ℕ) where
  h : Vec hidden
  c : Vec hidden

structure LSTMCell (input hidden : ℕ) where
  Af : Affine hidden (hidden + input)
  Ai : Affine hidden (hidden + input)
  Ac : Affine hidden (hidden + input)
  Ao : Affine hidden (hidden + input)

def LSTMCell.step {input hidden : ℕ} (cell : LSTMCell input hidden)
    (s : LSTMState hidden) (x : Vec input) : LSTMState hidden :=
  let hx := Fin.addCases s.h x
  let f := sigmoid ∘ cell.Af.apply hx
  let i := sigmoid ∘ cell.Ai.apply hx
  let g := tanh ∘ cell.Ac.apply hx
  let o := sigmoid ∘ cell.Ao.apply hx
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
  4 * Affine.numParams hidden (hidden + input)

structure GRUCell (input hidden : ℕ) where
  Az : Affine hidden (hidden + input)
  Ar : Affine hidden (hidden + input)
  Ah : Affine hidden (hidden + input)

def GRUCell.step {input hidden : ℕ} (cell : GRUCell input hidden)
    (h : Vec hidden) (x : Vec input) : Vec hidden :=
  let hx := Fin.addCases h x
  let z := sigmoid ∘ cell.Az.apply hx
  let r := sigmoid ∘ cell.Ar.apply hx
  let g := tanh ∘ cell.Ah.apply (Fin.addCases (r * h) x)
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
  3 * Affine.numParams hidden (hidden + input)

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
