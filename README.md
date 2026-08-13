# Flipkart Support Assistant — Parts 1–3

This repository implements the supplied 100-mark project brief as one integrated support-assistant demo. The brief requires a single public GitHub repository containing Parts 1–3, their code, saved model artifacts, README instructions, and transcripts. fileciteturn0file0L5-L16

## Repository layout

```text
generate_orders.py
orders_dataset.csv
part1/train_return_risk.py
part2/train_product_classifier.py
part2/predict_product.py
part3/policy_kb.json
part3/build_index.py
part3/agent.py
part3/retrieval_eval.py
part3/run_transcripts.py
models/return_risk_model.pkl
models/return_risk_threshold.json
models/product_classifier.pt
results/
data/sample_images/
transcripts/
```

## Part 1 — Return-risk model

The exact seeded generator from the brief is in `generate_orders.py`; run it from the repository root:

```bash
python generate_orders.py
python part1/train_return_risk.py
```

The generator produces exactly 6,000 rows and 13 columns. The observed return rate is **22.75%** and `rating_given` is missing on **13.05%** of rows. Missingness is **MAR**, because the missingness probability is conditioned on the observed `payment_method`: COD is **22.83%** missing versus **6.06%** for non-COD. This is not MCAR because the rates differ by an observed variable, and it is not MNAR because the generator does not use the unobserved rating value to determine whether it is missing. fileciteturn0file0L97-L100

### Current Part 1 run

| Metric | Result |
|---|---:|
| Rows / columns | 6,000 / 13 |
| Return rate | 22.75% |
| Rating missing | 13.05% |
| Dummy accuracy | 77.25% |
| Dummy F1 (return=1) | 0.00 |
| Logistic ROC-AUC | 0.6253 |
| Logistic F1 @ 0.5 | 0.3921 |
| Logistic best threshold | 0.44 |
| Logistic recall @ 0.44 | 75.82% |
| Logistic precision @ 0.44 | 28.01% |
| RF best CV ROC-AUC | 0.6178 |
| RF test ROC-AUC | 0.6143 |
| RF t*_rf | 0.46 |
| RF F1 @ t*_rf | 0.3962 |

The DummyClassifier illustrates the required “high accuracy, zero recall” trap: predicting the majority class can look accurate when returns are the minority class, while completely failing to identify returns. The brief explicitly requires this interpretation. fileciteturn0file0L88-L103

The final saved model is the tuned Random Forest pipeline, not the Logistic Regression. Its Part 3 buckets are anchored to **t*_rf = 0.46**: Low `< 0.46`, Medium `0.46–<0.61`, High `>= 0.61`. This follows the brief's requirement that the threshold come from the Random Forest's own `predict_proba` output. fileciteturn0file0L93-L106

## Part 2 — Fashion-MNIST transfer learning

Install requirements, then:

```bash
python part2/train_product_classifier.py
```

The implementation uses Fashion-MNIST, a pretrained ResNet-18, 3-channel replication, 224×224 resizing, ImageNet normalization, a frozen-backbone feature-extraction phase, and late-layer fine-tuning only if validation accuracy is below 80%. It caches frozen-backbone features to make CPU execution practical. These choices follow the project brief. fileciteturn0file0L115-L121

The script exports at least five actual test-set PNG files into `data/sample_images/` and saves `models/product_classifier.pt`.

## Part 3 — LangGraph support agent

Build the local vector index first:

```bash
python part3/build_index.py
```

Then evaluate retrieval:

```bash
python part3/retrieval_eval.py
```

The agent is in `part3/agent.py`. It contains four graph nodes:

- `intent` — policy / return-risk / product-category routing
- `retrieve` — sentence-transformer + Faiss policy retrieval
- `tools` — calls the real saved Part 1 or Part 2 artifact
- `response` — deterministic `MOCK_LLM`-style structured response generation

There is a conditional edge after intent classification. Return-risk buckets use the saved Random Forest's `t*_rf`, and image classification loads the actual saved product-classifier artifact. The required architecture and artifact-loading behavior are specified in the brief. fileciteturn0file0L136-L160

Run transcripts with:

```bash
python part3/run_transcripts.py
```

The project requires eight or more test conversations, including policy RAG, return-risk, product classification, multi-turn state, a fresh conversation, prompt injection, and an ungrounded policy question. The transcript runner provides these scenarios once the local Part 2 artifact and Part 3 index have been built. fileciteturn0file0L148-L161

## Git workflow required by the brief

Use a real feature branch with at least two commits and merge it into `main`:

```bash
git checkout -b feature/flipkart-support-agent
git add . && git commit -m "feat: implement support assistant parts 1 and 2"
# make another meaningful change
git add . && git commit -m "feat: add LangGraph support agent"
git checkout main
git merge --no-ff feature/flipkart-support-agent -m "merge: flipkart support assistant"
git log --graph --all --oneline
```

The final submission is **one public GitHub repository URL**, not separate links for the three parts. fileciteturn0file0L5-L8

## Important execution note

Part 1 has been generated and trained in this working environment, so its reported metrics and saved Random Forest artifact are real outputs. Part 2 requires downloading Fashion-MNIST and pretrained ResNet-18 weights; Part 3 requires downloading the local sentence-transformer model. This execution environment currently has no package/model-download access for those external assets, so their scripts are provided but their final numerical results must be generated on a machine with the listed packages and dataset/model downloads available. Do **not** invent those results; the brief explicitly requires real model predictions. fileciteturn0file0L126-L132

Project status: Part 1 artifact generated; Part 2/3 scripts ready for dependency-backed execution.
