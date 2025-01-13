# Post Training

- Pretrained LLMs allow us to tackle a wide range of NLP tasks with minimal architectural changes.

## Encoder-Only
- BERT
  - Single Text Classification
    - ![bert_single_classification.png](bert_single_classification.png)[Source](http://d2l.ai/chapter_natural-language-processing-applications/finetuning-bert.html)
  - Text Pair Classification or Regression
    - ![bert_pair_classification.png](bert_pair_classification.png)[Source](http://d2l.ai/chapter_natural-language-processing-applications/finetuning-bert.html)
  - Text Tagging
    - ![bert_text_tagging.png](bert_text_tagging.png)[Source](http://d2l.ai/chapter_natural-language-processing-applications/finetuning-bert.html)
  - Question Answering
    - ![bert_qna.png](bert_qna.png)[Source](http://d2l.ai/chapter_natural-language-processing-applications/finetuning-bert.html)
    - For the Stanford Question Answering Dataset, the answer to every question is a text span from the input passage.
    - The goal is to predict the start and end of the text span.

## Decoder-Only
- Classification
  - We usually amend the final linear layer, and because of the causal self-attention mask, look at the last token.
- Following instructions
  - Predicting the next word is insufficient to for more specialized use cases, e.g. answering (certain) prompts
  - Guidance
    - Prompt-tuning (see [Post-Training](../22_post_training/notes.md))
    - Retrieval Augmented Generation (RAG) provides relevant context in the input
      - Relevant context is drawn from a larger database of text
      - Database text is first embedded. At inference time, the query is embedded and similarity search is used to find relevant text pieces
  - Fine-Tuning
    - Parameter Efficient Fine Tuning
      - See [Post-Training](../22_post_training/notes.md).
    - Reinforcement Learning with Human Feedback (RLHF)
      - See [Post-Training](../22_post_training/notes.md)
      - InstructGPT/ChatGPT used RLHF to generate "human-like" responses
      - Anthropic has a [paper](../23_safety/03_alignment.md) where they do this with an AI
    - Direct Preference Optimization (DPO)
      - See [Post-Training](../22_post_training/notes.md)
    - Retrieval Augmented Generation
    - [Agents](./agents.md)
    
## Task-specific Architectures

- The ability of pretrained LLMs to tackle a wide range of NLP tasks with minimal architectural changes largely reduces the need for crafting task-specific architectures. 
- With space and time constraints, however, one may still consider building task-specific architectures on top of pre-trained, "frozen" embeddings. For example,
  - Sentiment analysis
    - Use a [bidirectional RNN + GloVe embeddings](http://d2l.ai/chapter_natural-language-processing-applications/sentiment-analysis-rnn.html)
    - Use a [textCNN + GloVe embeddings](http://d2l.ai/chapter_natural-language-processing-applications/sentiment-analysis-cnn.html)
      - 1D Convolutions capture sequential information
  - Natural Language Inference 
    - Use an [attention-based model + GloVe embeddings](http://d2l.ai/chapter_natural-language-processing-applications/natural-language-inference-attention.html)