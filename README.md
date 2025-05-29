# SS25AIR\_Group14 (Machinen Results in the Test Batch 4)

This repository was used to structure our **pipeline for the BioASQ competition**. Please note that some large files (e.g., models and large retrieval results) are not included here due to repository size limitations.
**To ensure full reproducibility, we have provided these files on Google Drive:**
👉 [Access all models and large retrieval files here.](https://drive.google.com/drive/u/0/folders/1_BjWUujPHd3s0l7Y2eoCNc7QC5u4wipg)



## Repository Structure & Project Organization

```
SS25AI-Project/
│
├── api_retrieval/              # API retrieval with a pool of 1000 documents
│
├── bm_25/                      # BM25-based document ranking (filters from 1000 to 500)
│
├── NLP/                        # NLP models and utilities
│   ├── Model_In_Drive/         # Trained model storage (see Google Drive)
│   └── finetune.ipynb          # Fine-tuning code for the NLP module
│
├── Neural_Reranking/           # Neural network based reranking
│   ├── Model_In_Drive/         # Trained model storage (see Google Drive)
│   └── Test_Batches/           # Submission file for Phase A Test Batch 4
│
├── Phase_B/                    # Phase B: Question-specific agent modules & outputs
│   └── answers_phase_b-...json # Example output for Phase B
│
├── .gitignore                  # Git ignore file
├── AIR_Code.ipynb              # Main notebook (orchestration)
├── BioASQ-task13bPhase...json  # Test dataset for the 4th batch (BioASQ challenge)
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── training13b.json            # Training data for model fine-tuning
```

### Architectural Overview

This repository is organized following a two-phase pipeline:

* **Phase A:** Retrieval and ranking (BM25, NLP representations, neural reranking)
* **Phase B:** Question-specific agents for answer generation (Yes/No, Factoid/List, Summary)

See the main notebook (`AIR_Code.ipynb`) for a pipeline overview and usage instructions.
