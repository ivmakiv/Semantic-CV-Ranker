# Semantic CV Ranker

An automated AI system that helps recruiters **rank candidates for a specific job offer**.
Instead of manually reviewing dozens of irrelevant profiles, a recruiter can paste a job description into a simple web interface and instantly receive the **Top 10 most relevant CVs**, ranked by a transparent scoring logic.

This project is delivered as a **ready-to-use pipeline** you can adapt to your needs: plug in your own CV dataset, adjust section weights, swap models, or extend the scoring logic.

---

## 1) Problem & Motivation

Recruiters often receive a **large number of applications** for a single position.
A significant part of the process is spent reviewing CVs that are **not suitable** (wrong domain, wrong seniority, missing key skills, etc.). This creates:

* Slow hiring cycles
* High manual workload
* Risk of missing strong candidates in the noise

My goal is **not** to replace recruiters.
This tool accelerates the first-pass screening by automatically pushing the most relevant profiles to the top.

---

## 2) Project Overview

The system ranks candidates using **four core signals** extracted from both CVs and the job offer:

1. **Experience** (main recent position + years)
2. **Skills** (technical and non-technical skill list)
3. **Education** (highest degree + field of study)
4. **Location** (distance-based match)

Each signal is converted into a **score between 0 and 1**, then combined into a final weighted ranking score.

---

## 3) Data Source

I started from a real CV dataset provided via Kaggle:

* ~80 resumes
* Mostly **PDF files**, plus **scanned / image-based resumes**
* Languages: mostly **English and French**

The CV files are placed under:

* `data/CV/`

(See `data/README.md` for details.)

---

## 4) Why I Don’t Embed the Full CV

I initially experimented with embedding the **entire CV as one text block**.
The results were not reliable: CVs contain mixed information (education, projects, unrelated jobs, soft skills, hobbies), and a single embedding often loses the key hiring signals.

Instead, I extract the most informative sections and score them separately.

---

## 5) Structured Extraction with an LLM (CV → JSON)

After converting a CV into plain text, I use an LLM (Mistral) to extract a strict JSON profile with fixed keys:

* `Education` → highest degree (normalized) + field
* `Experience` → main recent professional position + total years for that position
* `Skills` → list of skills
* `Location` → "City, Country"

Degree normalization is enforced to one of:

* `High School`, `Bachelor`, `Master`, `PhD`

Implemented in:

* `pipelines/text_division_cv.py`
* `pipelines/text_division_offer.py`

---

## 6) Text Extraction (PDF + Scanned Documents)

CV files may be:

* Text-based PDFs (extractable with PyPDF2)
* Scanned/image documents (require OCR)

I use a hybrid approach:

* Try fast extraction first (PyPDF2)
* Fallback to OCR when needed (Mistral OCR)

Implemented in:

* `pipelines/text_extraction.py`

---

## 7) Embeddings & Fine-tuned Matching Models

Key fields are represented as embeddings and compared using cosine similarity.

### Why fine-tuning?

I tried generic embeddings first, but similarity was often not aligned with recruiting meaning.
So I fine-tuned specialized matchers on top of the same base model.

### Base model

* `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)

### Three specialized embedding matchers

1. **Skills matcher**

   * Compares CV skills ↔ job skills
2. **Education-field matcher**

   * Compares field of study ↔ job domain
3. **Position matcher**

   * Compares main job title / role ↔ offer role

Training resources:

* `training/datasets/`
* `training/scripts/`
* fine-tuned models in `training/models/`

---

## 8) Location (Geocoding)

I convert "City, Country" to coordinates and compute distance-based similarity.

* CV location → (lat, lon)
* Offer location → (lat, lon)
* Distance computed with Haversine formula
* Converted to a score in [0, 1] using an exponential decay

Implemented in:

* `pipelines/location_coor.py`

---

## 9) Storage (PostgreSQL + pgvector)

All processed CV profiles are stored in a local PostgreSQL database with pgvector.

Stored fields:

* `file_name`
* education: degree + vector(field)
* experience: years + vector(position)
* location: latitude + longitude
* vector(skills)

Schema:

* `db/schema.sql`

DB container:

* `db/docker-compose.yml`

Scripts:

* `scripts/clear_db.py` — clears the table
* `scripts/ingest_cvs.py` — processes and uploads all CVs to the DB

---

## 10) Scoring Logic

For every CV and a given job offer, the system computes four normalized scores:

* **Skills score**: cosine(skills_cv, skills_offer)
* **Education score**: mix(field similarity + degree match)
* **Experience score**: mix(position similarity + years match)
* **Location score**: function(distance_km)

Final ranking score (weighted sum, default):

* Experience: **0.4**
* Skills: **0.3**
* Education: **0.2**
* Location: **0.1**

Implemented in:

* `pipelines/comparision.py`

Ranking logic:

* `pipelines/rank_cvs.py`

---

## 11) How to Adapt This Pipeline

This project is designed to be reused and adjusted:

* **Use your own dataset:** replace `data/CV/` with your CV folder
* **Change weights:** update the weights in `pipelines/comparision.py`
* **Add/remove sections:** extend the LLM extraction JSON + scoring
* **Swap embedding models:** replace base model / fine-tuned models in `training/models/`
* **Improve matching:** expand training datasets and re-train matchers

---

## 12) Repository Structure

```
app/                  # Web interface (recruiter chat UI)
data/                 # CV dataset files
db/                   # docker-compose + schema.sql
pipelines/            # extraction, embedding, scoring, ranking
scripts/              # DB maintenance + ingestion scripts
training/             # datasets + training scripts + trained models
```

---

## 13) Setup & Run

### 13.1 Configure environment

Create `.env` from `.env.example` and fill:

* `MISTRAL_API_KEY=...`
* `DB_DSN=...`

### 13.2 Start Postgres (Docker)

```bash
docker compose -f db/docker-compose.yml up -d
```

### 13.3 Install dependencies

```bash
pip install -r requirements.txt
```

### 13.4 Initialize schema (if needed)

```bash
psql "$DB_DSN" -f db/schema.sql
```

### 13.5 Ingest CVs into DB

```bash
python scripts/ingest_cvs.py
```

### 13.6 Rank CVs for a job offer (CLI)

```bash
python pipelines/rank_cvs.py
```

### 13.7 Run the web app

```bash
python app/main.py
```

---

## 14) Notes & Limitations

* CV parsing quality depends on text extraction quality (OCR can introduce noise).
* This tool is designed to speed up screening; final hiring decisions remain human.
* Current dataset is small (~80 CVs); scaling is possible with batch processing and optimized indexing.
