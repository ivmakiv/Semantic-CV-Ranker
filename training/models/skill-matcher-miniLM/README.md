---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:4440
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Veille concurrentielle; Logistique; Rédaction de communiqués de
    presse; Brand Content; Clôture comptable; Espagnol (B1); Organisation d’événements;
    TOEIC en cours); Collaborations inter-marques; Marketing d’influence
  sentences:
  - Pack Office; Marketing d’influence; Reporting; Réseaux Sociaux; Brand Content;
    Anglais (B2; Développement de campagnes marketing; Coordination d’équipes
  - Graphic Design; English (Very Good); Outlook; IT Hardware Support; Troubleshooting
  - États Financiers; Arabe (Langue Maternelle); PowerPoint; Microsoft Office (Word;
    Banque d'Investissement; Leadership
- source_sentence: Arabic (Native); MySQL; Flask; Airflow; Agile Development; Data
    Analysis
  sentences:
  - Design; Suédois (intermédiaire); Espagnol (B1); Créativité
  - Comptabilité (Comptes Débiteurs/Créditeurs); Marketing (Réseaux Sociaux; Anglais
    (C2); Leadership; Techniques d'Évaluation; Finance Internationale; Planification
    Financière
  - Brand Content; Clôture comptable; Gestion de bases de données
- source_sentence: Stress Management; Business Law; English (Fluent); Optimization;
    Learning Ability; VBA; Written/Oral Communication; Corporate Finance; Excel; MATLAB;
    Industrial Engineering; Communication-Psychosocial-Leadership; Word
  sentences:
  - Arabe; Gestion de projet; Chimie; Mécanique des milieux continus; SQL; Programmation
    informatique; Optimisation; Analyse numérique; Physique quantique; MATLAB
  - Python; R; Anglais (C1; Turbomachines (hydrauliques/éoliennes); SQL (bases)
  - Python; Django; PostgreSQL; Figma; Flask; React Js; C++; Nginx; Teamwork
- source_sentence: Relationnel; Production de plastique biodégradable; Service clientèle;
    Physique-Chimie; Garage Band; Adaptabilité; Allemand (B1); Analyse R&D; Microsoft
    Office; Mathématiques; Anglais (B2); Assemblage et contrôle qualité; Python; Gestion
    des stocks
  sentences:
  - LinkedIn; Sens du relationnel; Montage vidéo; Photoshop; Autonomie; TikTok; PowerPoint
  - Turc (langue maternelle); Xfoil); Écoulements compressibles/visqueux; Gestion
    d’équipe (10 personnes); Thermodynamique; Design aérodynamique (VGK; MFN (Mécanique
    des Fluides Numériques); Ébavurage et finition de produits
  - Aide à la décision (bases de données); Gestion d’équipe (10 personnes); Espagnol
    (C1; MFN (Mécanique des Fluides Numériques); Montage vidéo (Adobe Premiere); Anglais
    (C1; Négociation de partenariats; MATLAB; Propulsion aéronautique; Pointwise
- source_sentence: Communication; Python; Gestion de projet; Fortran; R Studio; Japonais
    Basique; Interprétation des courbes d’essais mécaniques; Autonomie; Travail d’équipe;
    Sciences des matériaux; Lecture et interprétation des diagrammes de phase; Logiciel
    de cristallographie Vesta; Montage et prémontage de chalumeaux industriels; Modélisation
    thermique
  sentences:
  - Agile Methodologies; Neural Networks; C++; Jira; SQL
  - R; MLflow; Machine Learning; Excel; Django; Spark; Scenario Analysis; Power BI;
    Laravel
  - Time Management; Python; SQL
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Communication; Python; Gestion de projet; Fortran; R Studio; Japonais Basique; Interprétation des courbes d’essais mécaniques; Autonomie; Travail d’équipe; Sciences des matériaux; Lecture et interprétation des diagrammes de phase; Logiciel de cristallographie Vesta; Montage et prémontage de chalumeaux industriels; Modélisation thermique',
    'Time Management; Python; SQL',
    'R; MLflow; Machine Learning; Excel; Django; Spark; Scenario Analysis; Power BI; Laravel',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.1504, 0.0696],
#         [0.1504, 1.0000, 0.3678],
#         [0.0696, 0.3678, 1.0000]])
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

* Size: 4,440 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                          | sentence_1                                                                        | label                                                          |
  |:--------|:------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                              | string                                                                            | float                                                          |
  | details | <ul><li>min: 13 tokens</li><li>mean: 47.14 tokens</li><li>max: 127 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 30.25 tokens</li><li>max: 94 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.43</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                    | sentence_1                                                                                                                                                        | label              |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------|
  | <code>Oracle; Arabe (langue maternelle); InfluxDB; PL/SQL; JEE; Esprit d’équipe; NumPy; TensorFlow; GitHub; Time Series Forecasting</code>                                                                    | <code>Closing Skills; Voice Over; Arabic Language; Team Leadership; Active Listening; Mail; Mass Communication; Customer Service; Outlook; Public Speaking</code> | <code>0.16</code>  |
  | <code>English (B1); Maintenance Software (CMMS); Thermodynamics; Industrial; Word; PowerPoint); Electricity; Microsoft Office (Excel; Aeromodelling (Aerobatic Aircraft/Drone); Mechanics; Electronics</code> | <code>Linux (Ubuntu); Oracle; MATLAB; CNN; TensorFlow; Math; Figma</code>                                                                                         | <code>0.085</code> |
  | <code>VBA; Project Management; R; Python; Planning</code>                                                                                                                                                     | <code>R; VBA; Graphic Design (Canvas</code>                                                                                                                       | <code>0.928</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 5
- `multi_dataset_batch_sampler`: round_robin

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
- `num_train_epochs`: 5
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
- `bf16`: False
- `fp16`: False
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
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
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
- `hub_revision`: None
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
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 1.7986 | 500  | 0.0694        |
| 3.5971 | 1000 | 0.042         |


### Framework Versions
- Python: 3.13.1
- Sentence Transformers: 5.1.2
- Transformers: 4.57.2
- PyTorch: 2.9.1
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.1

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