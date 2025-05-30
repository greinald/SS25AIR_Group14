This repository was used to structure our **pipeline for the BioASQ competition**. Please note that some large files (e.g., models and large retrieval results) are not included here due to repository size limitations.
**To ensure full reproducibility, we have provided these files on Google Drive:**
👉 [Access all models and large retrieval files here.](https://drive.google.com/drive/folders/1_BjWUujPHd3s0l7Y2eoCNc7QC5u4wipg?usp=share_link)


## Repository Structure & Project Organization

```
SS25AI-Project/
│
├── api_retrieval/                    # API retrieval with a pool of 1000 documents
│
├── bm_25/                            # BM25-based document ranking (filters from 1000 to 500)
│
├── NLP/                              # NLP models and utilities
│   ├── Model_In_Drive/               # Trained model storage (see Google Drive)
│   └── finetune.ipynb                # Fine-tuning code for the NLP module
│
├── Neural_Reranking/                 # Neural network based reranking
│   ├── Model_In_Drive/               # Trained model storage (see Google Drive)
│   └── Test_Batches/                 # Submission file for Phase A Test Batch 4
│
├── Phase_B/                          # Phase B: Question-specific agent modules & outputs
│   └── answers_phase_b-...json       # Example output for Phase B
│
├── .gitignore                        # Git ignore file
├── AIR_Code.ipynb                    # Main notebook. Fully developed Pipeline. 
├── SS25__BIOASQ_Phase_A_Protocol.pdf # A protocol for Phase A containing a description of the evaluation metrics as well as tutorial for official testing
├── BioASQ-task13bPhase...json        # Test dataset for the 4th batch (BioASQ challenge)
├── README.md                         # The file you are reading
├── requirements.txt                  # Python dependencies
└── training13b.json                  # Training data for model fine-tuning
```

### Architectural Overview

This repository is organized following a two-phase pipeline:

<img width="691" alt="System_Architecture" src="https://github.com/user-attachments/assets/4e28d00f-fc60-4150-93e4-342986dd9975" /> <p align="center" style="color:gray"> <i>Figure: High-level architecture of the BioASQ pipeline implemented in this repository.<br> Phase A handles document retrieval and ranking. Phase B processes the ranked results through question-specific agents to generate answers.</i> </p>


Please see the main notebook (`AIR_Code.ipynb`) for a pipeline overview and usage instructions.




## Final Remarks

With this project our main goal was not only pure performance but rather create a pipeline which is easly replicatable and workable from others in future work. We tried to create a system which is easly understandable and tried to visualize that with the diagram above. This should make it easy for anyone to follow our work or even make it better.

**Note:** The API was non functional on the day of the subbmition(29.05.2025) but we hope that it will work when you try to run it. Thank you.
