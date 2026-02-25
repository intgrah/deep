import Deep.GPT

noncomputable section

namespace GPT3

def vocabSize : ℕ := 50257
def ctxLen : ℕ := 2048
def embedDim : ℕ := 12288
def keyDim : ℕ := 128
def numHeads : ℕ := 96
def hiddenDim : ℕ := embedDim * 4
def numLayers : ℕ := 96

def head : AttentionHead embedDim keyDim where
  Aq := ⟨0, 0⟩
  Ak := ⟨0, 0⟩
  Av := ⟨0, 0⟩

def mha : MultiHeadAttention embedDim where
  numHeads := numHeads
  keyDim := keyDim
  heads := Vector.ofFn fun _ => head
  Ao := ⟨0, 0⟩

def ffn : MLP gelu embedDim embedDim :=
  .cons (hidden := hiddenDim) ⟨0, 0⟩ (.single ⟨0, 0⟩)

def block : TransformerBlock embedDim where
  ln₁ := ⟨default, default, default⟩
  attn := mha
  ln₂ := ⟨default, default, default⟩
  ffn := ffn

def model : GPT vocabSize ctxLen where
  embedDim := embedDim
  numLayers := numLayers
  embed := default
  posEmbed := default
  blocks := Vector.ofFn fun _ => GPT3.block
  lnFinal := ⟨default, default, default⟩

def numParams : ℕ :=
  vocabSize * embedDim +
  ctxLen * embedDim +
  numLayers * (
    2 * LayerNorm.numParams embedDim +
    numHeads * AttentionHead.numParams embedDim keyDim +
    Affine.numParams embedDim (numHeads * keyDim) +
    Affine.numParams hiddenDim embedDim +
    Affine.numParams embedDim hiddenDim) +
  LayerNorm.numParams embedDim

#eval numParams

end GPT3

end
