# Pre Training

## Model Trends
- Encoder Only
  - BERT
    - Trained with 
      - Masked language modeling, where randomly masked tokens are fed in and the encoder needs to predict the masked tokens.
      - Sentence order, although this was less useful when pretraining RoBERTa. 
  - ALBERT (enforced parameter sharing)
  - SpanBERT (representing and predicting spans of text)
  - DistilBERT (lightweight via knowledge distillation) 
  - ELECTRA (replaced token detection)
- Encoder-Decoder
  - BART
  - T5
    - Trained with
      - Replacing consecutive spans with a special token.
- Decoder Only
  - \*GPT*
    - 