import Deep.Transformer

noncomputable section

open scoped Matrix

structure GPT (vocabSize ctxLen : ℕ) where
  embedDim : ℕ
  numLayers : ℕ
  embed : Mat vocabSize embedDim
  posEmbed : Mat ctxLen embedDim
  blocks : Vector (TransformerBlock embedDim) numLayers
  lnFinal : LayerNorm embedDim

def GPT.apply {vocabSize ctxLen : ℕ} (gpt : GPT vocabSize ctxLen) [NeZero gpt.embedDim]
    (tokens : Vector (Fin vocabSize) ctxLen) : Mat ctxLen vocabSize :=
  let tokenEmbeddings := Matrix.of fun i => gpt.embed tokens[i]
  tokenEmbeddings + gpt.posEmbed
  |> gpt.blocks.foldl (fun X block => block.apply X)
  |> gpt.lnFinal.applySeq
  |> (· * gpt.embedᵀ)

def GPT.numParams {vocabSize ctxLen : ℕ} (gpt : GPT vocabSize ctxLen) : ℕ :=
  vocabSize * gpt.embedDim +
  ctxLen * gpt.embedDim +
  gpt.blocks.foldl (· + ·.numParams) 0 +
  LayerNorm.numParams gpt.embedDim

end
