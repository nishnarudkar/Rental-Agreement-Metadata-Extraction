# Rental Agreement Metadata Extraction Using AI

An AI-powered document information extraction system designed to identify and extract structured metadata from rental agreements in different formats, including DOCX documents and scanned PNG images.

The project combines **Optical Character Recognition (OCR), Natural Language Processing (NLP), and Large Language Models (LLMs)** to understand legal agreements and extract relevant contractual information without relying on regular expressions or hardcoded extraction rules.

## Project Overview

Rental agreements contain important information such as contracting parties, agreement dates, monetary values, and notice periods. However, these details are often embedded in lengthy, unstructured documents with varying layouts and terminology.

Traditional rule-based extraction methods can struggle with these variations.

This project implements a document-processing pipeline that converts rental agreements into machine-readable text and uses a locally hosted language model to extract structured metadata.

### Objectives

- Extract metadata from DOCX and scanned PNG rental agreements.
- Process scanned documents using OCR.
- Identify contractual information using an LLM.
- Generate structured JSON predictions.
- Handle missing or ambiguous information.
- Evaluate extraction accuracy using field-level recall.
- Provide a modular, reproducible Python implementation.

## Metadata Fields

The system extracts six fields:

| Field | Description |
|---|---|
| Agreement Value | Monetary value associated with the agreement |
| Agreement Start Date | Commencement date of the agreement |
| Agreement End Date | Expiration date of the agreement |
| Renewal Notice (Days) | Required advance notice for renewal |
| Party One | First contracting party |
| Party Two | Second contracting party |

Missing or unsupported information should be represented as `null` rather than fabricated.

## System Architecture

```text
Rental Agreement Documents
         |
         v
Document Format Detection
         |
    +----+----+
    |         |
    v         v
  DOCX       PNG
    |         |
    v         v
python-docx  PaddleOCR
    |         |
    +----+----+
         |
         v
  Extracted Document Text
         |
         v
  Local LLM (Qwen2.5)
         |
         v
  Structured JSON Output
         |
         v
  Schema Validation
         |
         v
  Predictions CSV
         |
         v
  Field-Level Evaluation
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| python-docx | DOCX text extraction |
| PaddleOCR | Text recognition from scanned images |
| Ollama | Local language model serving |
| Qwen2.5 Instruct | Semantic metadata extraction |
| Pydantic | Structured output validation |
| pandas | Dataset processing and evaluation |
| pytest | Automated testing |
| Jupyter | Exploration and experimentation |

The local model and OCR runtime can be adjusted according to available hardware.

## Project Structure

```text
Rental-Agreement-Metadata-Extraction/
|
├── data/
│   ├── train/
│   ├── test/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
│   ├── __init__.py
│   ├── document_reader.py
│   ├── ocr.py
│   ├── extractor.py
│   ├── schemas.py
│   └── evaluate.py
│
├── outputs/
│   ├── extracted_text/
│   ├── predictions.csv
│   └── evaluation.json
│
├── tests/
│
├── requirements.txt
├── .gitignore
└── README.md
```

*The structure above represents the intended modular organization. File names should be updated to match the final implementation.*

## Dataset

The provided dataset consists of rental agreement documents and corresponding metadata annotations.

| Dataset | Documents |
|---|---:|
| Training | 10 |
| Test | 4 |
| Total | 14 |

The dataset includes different document layouts, agreement terminology, date representations, and monetary clauses.

### Data Challenges

- Inconsistent formatting across agreements
- Scanned documents with OCR errors
- Multiple dates within the same agreement
- Monthly rent versus security deposits
- Ambiguous notice-period clauses
- Multiple individuals or organizations mentioned in a document
- Missing or inconsistent reference annotations

The original documents and annotations are preserved during processing.

## Installation

### Prerequisites

- Python 3.11
- Conda or another Python virtual environment
- Ollama installed locally
- PaddlePaddle runtime compatible with the system
- NVIDIA GPU recommended for faster local LLM inference

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Rental-Agreement-Metadata-Extraction
```

### 2. Create a Python Environment

```bash
conda create -n rental-extraction python=3.11 -y
conda activate rental-extraction
```

### 3. Install Python Dependencies

```bash
python -m pip install -r requirements.txt
```

Install the appropriate PaddlePaddle runtime separately according to the official installation instructions.

### 4. Install the Local Language Model

Install Ollama from:

https://ollama.com/download

Download the model:

```bash
ollama pull qwen2.5:7b
```

Confirm that Ollama is running before executing the extraction pipeline.

## Extraction Methodology

### 1. Document Ingestion

The pipeline accepts rental agreements in supported formats and routes them to the appropriate text extraction component.

### 2. Text Extraction

**DOCX documents:** Text is extracted using `python-docx`, including paragraphs and supported tables.

**Scanned images:** PaddleOCR recognizes text from PNG images and converts it into machine-readable form.

### 3. AI-Based Metadata Extraction

The extracted text is provided to a locally hosted instruction-tuned language model.

The model identifies the requested contractual fields by interpreting the document's content and clauses.

The extraction approach is designed to avoid regular expressions and document-specific hardcoded extraction rules.

### 4. Structured Output Validation

Model responses are validated using Pydantic to enforce the expected output structure and handle missing fields.

### 5. Evaluation

Predictions are compared against reference annotations using field-level exact-match recall.

## Example Output

The following is an illustrative example using fictional data, not an actual model prediction:

```json
{
  "Agreement Value": 12000,
  "Agreement Start Date": "01.04.2024",
  "Agreement End Date": "31.03.2025",
  "Renewal Notice (Days)": null,
  "Party One": "Example Property Owner",
  "Party Two": "Example Tenant"
}
```

The final output formatting should follow the original dataset's expected schema.

## Evaluation Metrics

The primary evaluation metric is **field-level recall**, as required by the assignment.

For each metadata field:

\[
\text{Recall} = \frac{\text{Correctly Extracted Values}}{\text{Eligible Reference Values}}
\]

Exact-match scoring compares extracted values with reference annotations.

The evaluation should also document how missing labels, invalid dates, and formatting differences are handled.

### Results

| Metadata Field | Exact-Match Recall |
|---|---|
| Agreement Value | Pending evaluation |
| Agreement Start Date | Pending evaluation |
| Agreement End Date | Pending evaluation |
| Renewal Notice (Days) | Pending evaluation |
| Party One | Pending evaluation |
| Party Two | Pending evaluation |
| **Macro Average** | **Pending evaluation** |

Actual metrics will be added after running the finalized extraction pipeline on the designated test set.

## Design Considerations

**Local inference:** Rental agreement text can be processed locally without requiring a paid cloud LLM API.

**Modularity:** Document reading, OCR, extraction, validation, and evaluation are separated into independent components.

**Explainability:** Supporting text excerpts can be retained to help inspect extracted fields.

**Reproducibility:** Dependencies, model configuration, and evaluation procedures should be documented.

**Error handling:** Missing information should not be replaced with unsupported guesses.

## Limitations

- OCR errors may affect extraction quality.
- Small local language models can misinterpret complex legal clauses.
- Agreement dates may be ambiguous or inconsistent.
- Renewal and termination notice periods may be expressed differently.
- The small dataset limits the reliability of performance estimates.
- The system is intended for information extraction and does not provide legal advice.

## Future Improvements

- Support for PDF documents and additional file formats
- Layout-aware document understanding
- Confidence estimation and human review
- More extensive evaluation on diverse rental agreements
- Improved processing of long documents
- Optional REST API for integration with other applications

## Disclaimer

This project was developed as a machine learning and document understanding assignment. Extracted metadata should be verified before being used in legal, financial, or contractual decisions.

## Author

Developed as part of an AI/ML internship technical assignment.

## License

No license has been specified. All rights are reserved unless a license is added to the repository.