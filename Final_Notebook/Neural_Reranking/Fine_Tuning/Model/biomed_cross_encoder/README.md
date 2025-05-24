---
tags:
- sentence-transformers
- cross-encoder
- generated_from_trainer
- dataset_size:441019
- loss:BinaryCrossEntropyLoss
base_model: microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract) <!-- at revision d673b8835373c6fa116d6d8006b33d48734e305d -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ['How do CYP1A2 polymorphisms affect the habitual coffee consumption effect on apetite?', 'The coffee fermentation microflora were rich and mainly constituted of aerobic Gram-negative bacilli, with Erwinia and Klebsiella genuses at the highest frequencies. The best population increase was observed with lactic acid bacteria and yeasts, whereas those microorganisms that counted on a pectin medium remained constant during the fermentation step. Qualitatively, lactic acid bacteria belonged mainly to Leuconostoc mesenteroides species but the others microflora were relatively heterogeneous. The microorganisms isolated on pectin medium were Enterobacteriaceae, identified as Erwinia herbicola and Klebsiella pneumoniae, not reported as strong pectolytic strains. Throughout coffee fermentation, 60% of the simple sugars were degraded by the total microflora and not specifically by pectolytic microorganisms.'],
    ['What is the function of lncRNA?', "Mammalian genomes include many maternally and paternally imprinted genes. Most of these are also expressed in the brain, and several have been implicated in regulating specific behavioral traits. Here, we have used a knockout approach to study the function of <i>Peg13</i>, a gene that codes for a fast-evolving lncRNA (long noncoding RNA) and is part of a complex of imprinted genes on chromosome 15 in mice and chromosome 8 in humans. Mice lacking the 3' half of the transcript look morphologically wild-type but show distinct behavioral differences. They lose interest in the opposite sex, instead displaying a preference for wild-type animals of the same sex. Further, they show a higher level of anxiety, lowered activity and curiosity, and a deficiency in pup retrieval behavior. Brain RNA expression analysis reveals that genes involved in the serotonergic system, formation of glutamatergic synapses, olfactory processing, and estrogen signaling-as well as more than half of the other known imprinted genes-show significant expression changes in <i>Peg13</i>-deficient mice. Intriguingly, these pathways are differentially affected in the sexes, resulting in male and female brains of <i>Peg13</i>-deficient mice differing more from each other than those of wild-type mice. We conclude that <i>Peg13</i> is part of a developmental pathway that regulates the neurobiology of social and sexual interactions."],
    ['What is Tarlov Cyst?', 'Enterogenous cyst is a rare congenital lesion generally located in the mediastinum or the abdominal cavity. We reported the first case of testicular enterogenous cyst in a 55-year-old white male presented with testicular pain and a gradually enlarging left scrotal mass with a 2-week duration.'],
    ['Which application is the backbone of BioPAXViz?', 'The catheter straight advancement rate for introduction into the epidural space was investigated using a radiopaque catheter. One hundred patients were divided into two groups and underwent thoracic or lumbar epidural punctures, with one of two different puncture methods: the median approach or paramedian approach. Two different angles of epidural puncture needle insertion, 50-60 degrees and 90 degrees to skin surface plane, were used. A catheter was inserted into the epidural space about 5 cm cephalad and the course of the inserted catheter was ascertained by radiography. The results have shown that punctures performed at an insertion angle of 50-60 degrees yielded higher catheter straight advancement rates than those performed at an angle of 90 degrees in both thoracic and lumbar epidural punctures.'],
    ['What is Aortitis?', 'In this paper the clinical condition of two male patients, aged 58 and 65 years are presented, after being admitted as a consequence of a rare complication of an inflammatory aneurysm of the abdominal aorta, which is an ureteral compression, with hydronephrosis, anuria and acute renal failure. After having an urgent haemodialysis session, the etiology of the process was diagnosed by echography and abdominal CT-scans, followed by ureteral catheterization, restoration of diuresis and normalization of renal function. Conventional surgery was performed later, in elective conditions, and the post operative course was normal, without complications. The main features of this clinical entity, its diagnosis and multidisciplinary management are presented and discussed.'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'How do CYP1A2 polymorphisms affect the habitual coffee consumption effect on apetite?',
    [
        'The coffee fermentation microflora were rich and mainly constituted of aerobic Gram-negative bacilli, with Erwinia and Klebsiella genuses at the highest frequencies. The best population increase was observed with lactic acid bacteria and yeasts, whereas those microorganisms that counted on a pectin medium remained constant during the fermentation step. Qualitatively, lactic acid bacteria belonged mainly to Leuconostoc mesenteroides species but the others microflora were relatively heterogeneous. The microorganisms isolated on pectin medium were Enterobacteriaceae, identified as Erwinia herbicola and Klebsiella pneumoniae, not reported as strong pectolytic strains. Throughout coffee fermentation, 60% of the simple sugars were degraded by the total microflora and not specifically by pectolytic microorganisms.',
        "Mammalian genomes include many maternally and paternally imprinted genes. Most of these are also expressed in the brain, and several have been implicated in regulating specific behavioral traits. Here, we have used a knockout approach to study the function of <i>Peg13</i>, a gene that codes for a fast-evolving lncRNA (long noncoding RNA) and is part of a complex of imprinted genes on chromosome 15 in mice and chromosome 8 in humans. Mice lacking the 3' half of the transcript look morphologically wild-type but show distinct behavioral differences. They lose interest in the opposite sex, instead displaying a preference for wild-type animals of the same sex. Further, they show a higher level of anxiety, lowered activity and curiosity, and a deficiency in pup retrieval behavior. Brain RNA expression analysis reveals that genes involved in the serotonergic system, formation of glutamatergic synapses, olfactory processing, and estrogen signaling-as well as more than half of the other known imprinted genes-show significant expression changes in <i>Peg13</i>-deficient mice. Intriguingly, these pathways are differentially affected in the sexes, resulting in male and female brains of <i>Peg13</i>-deficient mice differing more from each other than those of wild-type mice. We conclude that <i>Peg13</i> is part of a developmental pathway that regulates the neurobiology of social and sexual interactions.",
        'Enterogenous cyst is a rare congenital lesion generally located in the mediastinum or the abdominal cavity. We reported the first case of testicular enterogenous cyst in a 55-year-old white male presented with testicular pain and a gradually enlarging left scrotal mass with a 2-week duration.',
        'The catheter straight advancement rate for introduction into the epidural space was investigated using a radiopaque catheter. One hundred patients were divided into two groups and underwent thoracic or lumbar epidural punctures, with one of two different puncture methods: the median approach or paramedian approach. Two different angles of epidural puncture needle insertion, 50-60 degrees and 90 degrees to skin surface plane, were used. A catheter was inserted into the epidural space about 5 cm cephalad and the course of the inserted catheter was ascertained by radiography. The results have shown that punctures performed at an insertion angle of 50-60 degrees yielded higher catheter straight advancement rates than those performed at an angle of 90 degrees in both thoracic and lumbar epidural punctures.',
        'In this paper the clinical condition of two male patients, aged 58 and 65 years are presented, after being admitted as a consequence of a rare complication of an inflammatory aneurysm of the abdominal aorta, which is an ureteral compression, with hydronephrosis, anuria and acute renal failure. After having an urgent haemodialysis session, the etiology of the process was diagnosed by echography and abdominal CT-scans, followed by ureteral catheterization, restoration of diuresis and normalization of renal function. Conventional surgery was performed later, in elective conditions, and the post operative course was normal, without complications. The main features of this clinical entity, its diagnosis and multidisciplinary management are presented and discussed.',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 441,019 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                      | sentence_1                                                                                         | label                                                          |
  |:--------|:------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                             | float                                                          |
  | details | <ul><li>min: 14 characters</li><li>mean: 55.77 characters</li><li>max: 189 characters</li></ul> | <ul><li>min: 95 characters</li><li>mean: 1207.39 characters</li><li>max: 4524 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.09</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                         | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:---------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>How do CYP1A2 polymorphisms affect the habitual coffee consumption effect on apetite?</code> | <code>The coffee fermentation microflora were rich and mainly constituted of aerobic Gram-negative bacilli, with Erwinia and Klebsiella genuses at the highest frequencies. The best population increase was observed with lactic acid bacteria and yeasts, whereas those microorganisms that counted on a pectin medium remained constant during the fermentation step. Qualitatively, lactic acid bacteria belonged mainly to Leuconostoc mesenteroides species but the others microflora were relatively heterogeneous. The microorganisms isolated on pectin medium were Enterobacteriaceae, identified as Erwinia herbicola and Klebsiella pneumoniae, not reported as strong pectolytic strains. Throughout coffee fermentation, 60% of the simple sugars were degraded by the total microflora and not specifically by pectolytic microorganisms.</code>                                                                                                                                                                                          | <code>0.0</code> |
  | <code>What is the function of lncRNA?</code>                                                       | <code>Mammalian genomes include many maternally and paternally imprinted genes. Most of these are also expressed in the brain, and several have been implicated in regulating specific behavioral traits. Here, we have used a knockout approach to study the function of <i>Peg13</i>, a gene that codes for a fast-evolving lncRNA (long noncoding RNA) and is part of a complex of imprinted genes on chromosome 15 in mice and chromosome 8 in humans. Mice lacking the 3' half of the transcript look morphologically wild-type but show distinct behavioral differences. They lose interest in the opposite sex, instead displaying a preference for wild-type animals of the same sex. Further, they show a higher level of anxiety, lowered activity and curiosity, and a deficiency in pup retrieval behavior. Brain RNA expression analysis reveals that genes involved in the serotonergic system, formation of glutamatergic synapses, olfactory processing, and estrogen signaling-as well as more than half of the other known i...</code> | <code>0.0</code> |
  | <code>What is Tarlov Cyst?</code>                                                                  | <code>Enterogenous cyst is a rare congenital lesion generally located in the mediastinum or the abdominal cavity. We reported the first case of testicular enterogenous cyst in a 55-year-old white male presented with testicular pain and a gradually enlarging left scrotal mass with a 2-week duration.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `fp16`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `tp_size`: 0
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional

</details>

### Training Logs
| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 0.0181 | 500   | 0.3642        |
| 0.0363 | 1000  | 0.2504        |
| 0.0544 | 1500  | 0.2234        |
| 0.0726 | 2000  | 0.2192        |
| 0.0907 | 2500  | 0.1653        |
| 0.1088 | 3000  | 0.1449        |
| 0.1270 | 3500  | 0.1341        |
| 0.1451 | 4000  | 0.1274        |
| 0.1633 | 4500  | 0.1172        |
| 0.1814 | 5000  | 0.1187        |
| 0.1995 | 5500  | 0.0978        |
| 0.2177 | 6000  | 0.1074        |
| 0.2358 | 6500  | 0.1074        |
| 0.2540 | 7000  | 0.092         |
| 0.2721 | 7500  | 0.1028        |
| 0.2902 | 8000  | 0.0919        |
| 0.3084 | 8500  | 0.0786        |
| 0.3265 | 9000  | 0.0682        |
| 0.3447 | 9500  | 0.0772        |
| 0.3628 | 10000 | 0.0705        |
| 0.3809 | 10500 | 0.0595        |
| 0.3991 | 11000 | 0.0557        |
| 0.4172 | 11500 | 0.067         |
| 0.4354 | 12000 | 0.054         |
| 0.4535 | 12500 | 0.0518        |
| 0.4716 | 13000 | 0.054         |
| 0.4898 | 13500 | 0.0552        |
| 0.5079 | 14000 | 0.0415        |
| 0.5260 | 14500 | 0.0487        |
| 0.5442 | 15000 | 0.0447        |
| 0.5623 | 15500 | 0.043         |
| 0.5805 | 16000 | 0.0462        |
| 0.5986 | 16500 | 0.0428        |
| 0.6167 | 17000 | 0.0426        |
| 0.6349 | 17500 | 0.0509        |
| 0.6530 | 18000 | 0.0426        |
| 0.6712 | 18500 | 0.0469        |
| 0.6893 | 19000 | 0.0375        |
| 0.7074 | 19500 | 0.0367        |
| 0.7256 | 20000 | 0.0357        |
| 0.7437 | 20500 | 0.0379        |
| 0.7619 | 21000 | 0.035         |
| 0.7800 | 21500 | 0.036         |
| 0.7981 | 22000 | 0.0346        |
| 0.8163 | 22500 | 0.0307        |
| 0.8344 | 23000 | 0.0368        |
| 0.8526 | 23500 | 0.031         |
| 0.8707 | 24000 | 0.0363        |
| 0.8888 | 24500 | 0.0435        |
| 0.9070 | 25000 | 0.0353        |
| 0.9251 | 25500 | 0.0332        |
| 0.9433 | 26000 | 0.0322        |
| 0.9614 | 26500 | 0.0345        |
| 0.9795 | 27000 | 0.0399        |
| 0.9977 | 27500 | 0.0317        |


### Framework Versions
- Python: 3.10.10
- Sentence Transformers: 4.1.0
- Transformers: 4.51.3
- PyTorch: 2.7.0+cu128
- Accelerate: 1.6.0
- Datasets: 3.6.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->