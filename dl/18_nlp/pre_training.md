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
    - Pretraining T5 by predicting consecutive spans. 
    - The original sentence is “I”, “love”, “this”, “red”, “car”
    - Input is masked to “I”, “<X>”, “this”, “<Y>”
    - Target sequence is “<X>”, “love”, “<Y>”, “red”, “car”, “<Z>”
- Decoder Only
  - \*GPT*
    - 