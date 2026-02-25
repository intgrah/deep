import Deep.Affine
import Deep.Activation

noncomputable section

open scoped Matrix
open Real (sqrt)

structure AttentionHead (embedDim keyDim : ℕ) where
  Aq : Affine keyDim embedDim
  Ak : Affine keyDim embedDim
  Av : Affine keyDim embedDim

def AttentionHead.apply {embedDim keyDim seqLen : ℕ}
    (head : AttentionHead embedDim keyDim) (X : Mat seqLen embedDim) : Mat seqLen keyDim :=
  let Q := head.Aq.applySeq X
  let K := head.Ak.applySeq X
  let V := head.Av.applySeq X
  let causalMask : Mat seqLen seqLen :=
    Matrix.of fun i j => if j ≤ i then 0 else -1e100
  let scores := (1 / sqrt keyDim) • (Q * Kᵀ)
  let logits := scores + causalMask
  let attn : Mat seqLen seqLen := Matrix.of (softmax ∘ logits)
  attn * V

def AttentionHead.numParams (embedDim keyDim : ℕ) : ℕ :=
  3 * Affine.numParams keyDim embedDim

structure MultiHeadAttention (embedDim : ℕ) where
  numHeads : ℕ
  keyDim : ℕ
  heads : Vector (AttentionHead embedDim keyDim) numHeads
  Ao : Affine embedDim (numHeads * keyDim)

def MultiHeadAttention.apply {embedDim seqLen : ℕ}
    (mha : MultiHeadAttention embedDim)
    (X : Mat seqLen embedDim) :
    Mat seqLen embedDim :=
  let headOutputs : Vector (Mat seqLen mha.keyDim) mha.numHeads :=
    mha.heads.map (·.apply X)
  let concat : Mat seqLen (mha.numHeads * mha.keyDim) :=
    Matrix.of fun i j => headOutputs[j.divNat] i j.modNat
  mha.Ao.applySeq concat

def MultiHeadAttention.numParams {embedDim : ℕ} (mha : MultiHeadAttention embedDim) : ℕ :=
  mha.numHeads * AttentionHead.numParams embedDim mha.keyDim +
  Affine.numParams embedDim (mha.numHeads * mha.keyDim)

end
