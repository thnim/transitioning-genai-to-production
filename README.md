# transitioning-genai-to-production
# Replication Package: Transitioning From GenAI Tooling to Production: Exploring the Diversity of AI Projects on GitHub

This repository contains the data, scripts, and results necessary to replicate the study detailed in the paper, **"Transitioning From GenAI Tooling to Production: Exploring the Diversity of AI Projects on GitHub"**.

## 1. Structure of the Repository

The repository files are organized as follows in the main directory:

| File/Directory | Content | Description |
| :--- | :--- | :--- |
| `clean-reader_prompt_metadata.json` | Initial Dataset | The starting raw dataset of code prompts. |
| `filter_prompt_files.py` | Python Script | Executes the data filtering process (Step 1). |
| `extract_reposity_information.py` | Python Script | Executes the repository information extraction (Step 2). |
| `repository_dataset.csv` | Intermediate Data | Output of Step 2, containing all extracted repository metadata. |
| `association_rule.py` | Python Script | Executes the association rule analysis (Step 4.1). |
| `heatmap.py` | Python Script | Executes the heatmap visualization (Step 4.2). |
| `study_result/` | Output Directory | Contains final processed data from analysis. |
| `README.md` | This file | Provides an overview of the package and replication steps. |

## 1. Requirements

To run the scripts and replicate the study, you need the following:

### Software
* Python 3.8+

### Python Libraries
Install the necessary Python libraries using the following command:

```bash
pip install pandas requests tqdm beautifulsoup4 langdetect mlxtend seaborn matplotlib

### GitHub Authentication
The data extraction script (`extract_reposity_information.py`) relies heavily on the GitHub REST API. To prevent hitting rate limits, you **must** set your GitHub Personal Access Token as an environment variable:

```bash
export GITHUB_TOKEN="your_personal_access_token"

## 2. Data Collection and Processing Pipeline

The replication process is executed in three sequential steps.

### Step 1: Initial Data Filtering

**Purpose:** To filter the initial raw dataset of code prompts (`clean-reader_prompt_metadata.json`) down to high-quality, English-only prompts with substantial content.

| Input File | Script Used | Output File | Filter Criteria |
| :--- | :--- | :--- | :--- |
| `clean-reader_prompt_metadata.json` | `filter_prompt_files.py` | `filtered_prompt_files.json` | Prompts must be detected as **English** and contain **more than 15 words**. |

To run this step:
```bash
python filter_prompt_files.py

### Step 2: Repository Information Extraction

**Purpose:** To gather detailed metadata for all repositories identified in the filtered dataset, including LLM usage, README presence, size, and activity metrics (PRs/Issues).

| Input File | Script Used | Output File | Key Feature |
| :--- | :--- | :--- | :--- |
| `filtered_prompt_files.json` | `extract_reposity_information.py` | `repository_dataset.csv` | **One-pass extraction** to minimize API calls, with logic to record `"ERROR"` data instead of skipping failed repos. |

To run this step (requires `GITHUB_TOKEN`):
```bash
python extract_reposity_information.py

### Step 3: Final Dataset Preparation (Manual and Filtered)

The `repository_dataset.csv` was further refined to create the **Final Dataset** (293 repositories) used for analysis:

1.  **Filtering:** Repositories with a fundamental API fetching error (`Description` != "Error fetching repo"), detected LLM usage, and a confirming README (`Have README` = "Yes") were selected, resulting in **297 initial candidates**.
2.  **Manual Cleaning:** Four repositories were manually removed from this set because their README files were found to be non-English.
3.  **Anonymization and Final Count:** The remaining repositories were anonymized and resulted in the **293 repositories** used for the final study.

## 3. Analysis and Evaluation

The core analysis of the final dataset is performed using the scripts below.

### 4.1. Association Rule Mining

**Script:** `association_rule.py`

**Purpose:** To find meaningful relationships (association rules) of README doucumentation sections pattern.

### 4.2. Visualization (Heatmap)

**Script:** `heatmap.py`

**Purpose:** To generate a visualization (heatmap) showing README section coverage by project category

To run the analysis scripts:

```bash
python association_rule.py
python heatmap.py


