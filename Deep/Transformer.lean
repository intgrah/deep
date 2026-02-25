import Deep.Attention
import Deep.LayerNorm
import Deep.MLP

noncomputable section

open Matrix (of)

structure TransformerBlock (embedDim : ℕ) where
  ln₁ : LayerNorm embedDim
  attn : MultiHeadAttention embedDim
  ln₂ : LayerNorm embedDim
  ffn : MLP gelu embedDim embedDim

def TransformerBlock.apply {embedDim seqLen : ℕ} [NeZero embedDim]
    (block : TransformerBlock embedDim)
    (X : Mat seqLen embedDim) :
    Mat seqLen embedDim :=
  X
  |> res (block.attn.apply ∘ block.ln₁.applySeq)
  |> res (block.ffn.applySeq ∘ block.ln₂.applySeq)

def TransformerBlock.numParams {embedDim : ℕ} (block : TransformerBlock embedDim) : ℕ :=
  LayerNorm.numParams embedDim + block.attn.numParams +
  LayerNorm.numParams embedDim + block.ffn.numParams

end
