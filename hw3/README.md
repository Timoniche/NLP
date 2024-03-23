

1. Train generative model
2. Take decoder-only/encoder-decoder model as a backbone
3. Train on
   1. https://tlk.s3.yandex.net/dataset/TlkPersonaChatRus.zip
   2. https://huggingface.co/datasets/zjkarina/matreshka
   3. https://github.com/Koziev/NLP_Datasets/tree/master/Conversations/Data
4. Add dialogue cycle (history context)
5. Configure generating parameters
6. Evaluate on metrics (BLEU):
   1. https://lightning.ai/docs/torchmetrics/stable/text/bleu_score.html
   2. https://huggingface.co/spaces/evaluate-metric/meteor
   3. https://lightning.ai/docs/torchmetrics/stable/text/rouge_score.html