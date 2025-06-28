# Contract Risk Classification Using BERT

## Overview
This project focuses on building an end-to-end machine learning pipeline to classify legal contract text into specific risk categories using advanced Natural Language Processing (NLP) techniques. It leverages BERT, a powerful transformer-based language model, to understand and classify complex legal language. The model predicts risks into one of the following classes:

- `Privacy`
- `IP` (Intellectual Property)
- `Compliance`

The project also includes a clean and scalable deployment architecture using **Google Cloud Platform (GCP)** for data storage, training, and inference, along with a **FastAPI-based user interface** for public or internal access.

---

## Business Use Cases

### 1. Legal and Business Firms
- **Automated Contract Review:** Automatically identify clauses that may pose privacy, intellectual property, or compliance risks.
- **Due Diligence:** Flag risk elements in M&A contracts and vendor agreements.
- **Policy Audits:** Ensure compliance with internal standards and external regulations (GDPR, HIPAA, etc).

### 2. Healthcare & Insurance Companies
- **Privacy Monitoring:** Identify clauses where patient or client data may be mishandled or exposed.
- **Regulatory Auditing:** Check if agreements comply with healthcare laws.
- **Vendor Contracts:** Analyze risks in third-party contracts.

### 3. General Public or Individual Professionals
- **Freelancers & Startups:** Individuals signing service agreements or NDAs can upload text and receive risk classification.
- **Legal Aid:** Self-represented parties can gain insight into potentially risky terms.

---

## Technical Architecture

### Data
- **Input Format:** CSV file with at least one column: `text` (contract clauses) and a label `risk`.
- **GCP Storage:** Input and output files stored in a bucket: `contract-risk-classifier`.

### Preprocessing
- Lemmatization
- Noise removal (emails, URLs, HTML tags)
- Legal term standardization
- Stopword filtering

### Model
- **Transformer Used:** `bert-base-uncased`
- **Custom Classifier:** A simple linear head on top of BERT CLS embedding
- **Training:**
  - 1 Epoch
  - Batch size: 8
  - Loss: CrossEntropyLoss
  - Optimizer: Adam (lr=2e-5)
  - Freeze all BERT layers except last encoder layer (partial fine-tuning)

### Artifacts
- `model.pth` — Trained model weights
- `tokenizer.pkl` — BERT tokenizer
- `label_encoder.pkl` — LabelEncoder for inverse transformation

Stored in `gs://contract-risk-classifier/artifacts/`

### Inference
- New texts are cleaned, tokenized, vectorized, and passed to the model
- Predicted label is converted back using `label_encoder`

---

## Deployment

### FastAPI Web UI (Planned)
- **Goal:** A web interface where users can paste contract clauses and see predictions in real-time.
- **Tools:** FastAPI + HTML/CSS frontend (not yet implemented)
- **Access:** Can be containerized and deployed via GCP Cloud Run or Vertex AI endpoints.

---

## Vectorization & Storage
- **Embeddings:** The CLS token outputs (768-d vector) are stored for each contract clause.
- **Use Cases:**
  - Searchable vector database
  - Similarity clustering
  - Contract fingerprinting for retrieval tasks

---

## Workflow Summary
1. Upload data to GCS
2. Train model in Vertex AI Workbench with GPU
3. Store all artifacts (model, tokenizer, label encoder) in GCS
4. Run inference with a new text input
5. (Planned) Deploy FastAPI UI for public interaction

---

## Sample Output
**Input Clause:**
> "The contractor shall be liable for any breach of confidentiality during the term of this agreement."

**Predicted Risk:** `Privacy`

---

## Future Improvements
- Add FastAPI UI for real-time access
- Support multi-label classification
- Include explainability (e.g., attention weights)
- Use vector embeddings for document search

---

## Authors & Credits
- Chandrakanth Naidu — Data Engineering & Model Development
- GCP — Training, Storage, Deployment
- HuggingFace Transformers — Model
- Spacy & TextBlob — NLP cleaning utilities

---



