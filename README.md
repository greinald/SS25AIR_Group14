# SS25AIR_Group14(Machinen Results in the Test Batch 4)

This repository was used to structure our **pipeline for the BioASQ competition**. Please note that some large files (e.g., models and large retrieval results) are not included here due to repository size limitations. **To ensure full reproducibility, we have provided these files on Google Drive:**
👉 \[Access all models and large retrieval files here.]\([https://drive.google.com/drive/u/0/folders/](https://drive.google.com/drive/u/0/folders/)

# Repository Structure & Project Organization


SS25AI-Project/
│
├── api_retrieval/               # API retrieval with a pool of 1000 Documents. The retrival document can be found in the Drive
├── bm_25/                       # BM25-based document ranking. Filtering from 1000 to 500 documents
│
├── NLP/                         # NLP models 
│   ├── Model_In_Drive/          # Trained model storage (Can be found in Drive)
│   ├── finetune.ipynb/          # Fine-Tuning code for the NLP Modul
│
│
│
├── Neural_Reranking/            # Neural network based reranking
│   ├── Model_In_Drive/          # Trained model storage (Can be found in Drive)
│   └── Test_Batches/            # Subbmition file for Phase A Test Batch 4
│
│
├── Phase_B/                     # Phase B: Question-specific agent modules & outputs
│   └── answers_phase_b-...json  # Example output for Phase B
│
├── .gitignore                   # Git ignore file
├── AIR_Code.ipynb               # Main notebook
├── BioASQ-task13bPhase...json   # Test Dataset for the 4^{th} batch (BioASQ challenge)
├── README.md                    # This file that you are reading now.
├── requirements.txt             # Python dependencies
└── training13b.json             # Training data for model fine-tuning
