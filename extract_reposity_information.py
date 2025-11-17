import os
import json
import time
import re
from typing import Dict, List, Tuple, Optional
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm


# -------------------------------
# Configuration & Constants
# -------------------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") 

BASE_HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "GenAI-Repo-Study/1.0",
}
if GITHUB_TOKEN:
    BASE_HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"

RAW_README_HEADERS = {
    "User-Agent": "GenAI-Repo-Study/1.0",
}

RATE_LIMIT_SLEEP = 6 # Sleep time between fetching each complete repo dataset


# -------------------------------
# LLM Keyword Configuration (from getRepoWithLLM.py)
# -------------------------------
RAW_KEYWORD_TO_LABEL = {
     # OpenAI
    "openai": "openai", "openai.api_key": "openai", "openai.chatcompletion": "openai",
    "gpt": "openai", "gpt-1": "openai", "gpt1": "openai",
    "gpt-2": "openai", "gpt2": "openai",
    "gpt-3": "openai", "gpt3": "openai",
    "gpt-3.5": "openai", "gpt3.5": "openai",
    "gpt-4": "openai", "gpt4": "openai",
    "gpt-4-turbo": "openai", "gpt4-turbo": "openai",
    "gpt-4.5": "openai", "gpt4.5": "openai",
    "gpt-4o": "openai", "gpt4o": "openai",
    "chatgpt": "openai", "codex": "openai",

    # Anthropic
    "anthropic": "claude", "claude": "claude",
    "claude-1": "claude", "claude-2": "claude", "claude-2.1": "claude",
    "claude-3": "claude", "claude-3-opus": "claude", "claude-3-sonnet": "claude", "claude-3-haiku": "claude",

    # Google / DeepMind  
    "bard": "palm", 
    "palm": "palm", "palm-2": "palm", "text-bison": "palm",
    "gemini": "gemini", "gemini-1": "gemini", "gemini-1.5": "gemini",
    "gemini-pro": "gemini", "gemini-ultra": "gemini", "gemini-nano": "gemini",
    "chinchilla": "deepmind", "gopher": "deepmind",
    "lambda": "lamda", "lamda": "lamda", 
    "gemma": "gemma",

    # Meta
    "llama": "llama", "llama-2": "llama", "llama2": "llama", "llama-3": "llama", "llama3": "llama",
    "llama.cpp": "llama",
    "opt": "opt", "galactica": "meta", "chameleon": "meta",

    # Mistral
    "mistral": "mistral", "mistral-7b": "mistral", "mistral7b": "mistral",
    "mixtral": "mistral", "mixtral-8x7b": "mistral", "mixtral8x7b": "mistral",

    # EleutherAI
    "gpt-j": "eleutherai", "gptj": "eleutherai",
    "gpt-neo": "eleutherai", "gptneo": "eleutherai",
    "gpt-neo-2": "eleutherai", "gpt-neox": "eleutherai", "gpt-neox-20b": "eleutherai",

    # AI21
    "ai21": "ai21", "jurassic": "ai21", "jurassic-1": "ai21", "jurassic-2": "ai21",

    # Microsoft
    "phi": "microsoft", "phi-1": "microsoft", "phi-2": "microsoft", "phi-3": "microsoft",
    "orca": "microsoft", "orca-2": "microsoft", "orca-mini": "microsoft",

    # NVIDIA / Baidu 
    "megatron": "nvidia", "megatron-turing": "nvidia", "megatron-turing-nlg": "nvidia",
    "ernie": "baidu", "ernie-3.0": "baidu", "ernie-bot": "baidu",

    # IBM
    "granite": "ibm", "granite-13b": "ibm",

    # Amazon
    "titan": "amazon", "amazon titan": "amazon",
    "amazon bedrock": "amazon", "bedrock": "amazon",
    "amazon nova": "amazon", "nova": "amazon",

    # DeepSeek
    "deepseek": "deepseek", "deepseek-coder": "deepseek", "deepseek-v2": "deepseek", "deepseek-v3": "deepseek",

    # Bloomberg
    "bloomberggpt": "bloomberg", "bloomberg gpt": "bloomberg",

    # BLOOM
    "bloom": "bloom", "bloomz": "bloom",

    # Qwen 
    "qwen": "qwen", "qwen1": "qwen", "qwen2": "qwen", "qwen3": "qwen", "qwen-14b": "qwen",

    # xAI
    "grok": "xai", "grok-1": "xai", "grok-1.5": "xai",

    # Cohere
    "cohere": "cohere", "command-r": "cohere", "command-r+": "cohere",

    # Replit
    "replit": "replit", "replit-code-v1": "replit",

    # TII / Falcon 
    "falcon": "falcon", "falcon-7b": "falcon", "falcon-40b": "falcon",

    # OpenChat 
    "openchat": "openchat", "openchat-3.5": "openchat",

    # Tools / Frameworks / Platforms
    "huggingface": "platform:huggingface",
    "transformers": "platform:transformers",
    "langchain": "platform:langchain",
    "llama-index": "platform:llama-index",
    "autogen": "platform:autogen"
}

KEYWORD_TO_LABEL = {k.lower(): v for k, v in RAW_KEYWORD_TO_LABEL.items()}
KEYWORDS = sorted(KEYWORD_TO_LABEL.keys(), key=lambda k: -len(k))
LLM_PATTERN = re.compile(r"\b(" + "|".join(map(re.escape, KEYWORDS)) + r")\b", re.IGNORECASE)


# -------------------------------
# GitHub API Client (Unified Extraction)
# -------------------------------
class GitHubAPIClient:
    """Encapsulates all GitHub API interactions"""
    
    def __init__(self, session: requests.Session):
        self.session = session
    
    def fetch_repo_metadata(self, user: str, repo: str) -> Dict:
        """Fetch basic repo metadata (description, topics, size)"""
        api_url = f"https://api.github.com/repos/{user}/{repo}"
        try:
            response = self.session.get(api_url, headers=BASE_HEADERS, timeout=20)
            if response.status_code == 200:
                repo_info = response.json()
                return {
                    "description": repo_info.get("description") or "",
                    "topics": ", ".join(repo_info.get("topics", [])),
                    "size_kb": repo_info.get("size", 0),
                    "error": None
                }
        except requests.RequestException:
            pass
        
        # Return error placeholders
        return {"description": "Error fetching repo", "topics": "", "size_kb": "ERROR", "error": "fetch_failed"}
    
    def fetch_issue_pr_counts(self, user: str, repo: str) -> Dict:
        """Fetch PR and Issue counts"""
        counts = {
            "pr_total": self._search_count(user, repo, "type:pr"),
            "pr_open": self._search_count(user, repo, "type:pr+state:open"),
            "pr_closed": self._search_count(user, repo, "type:pr+state:closed"),
            "issue_total": self._search_count(user, repo, "type:issue"),
            "issue_open": self._search_count(user, repo, "type:issue+state:open"),
            "issue_closed": self._search_count(user, repo, "type:issue+state:closed"),
        }
        return counts
    
    def _search_count(self, user: str, repo: str, query: str) -> int:
        """Internal helper for search API calls"""
        url = f"https://api.github.com/search/issues?q=repo:{user}/{repo}+{query}"
        try:
            resp = self.session.get(url, headers=BASE_HEADERS, timeout=20)
            if resp.status_code == 200:
                return resp.json().get("total_count", 0)
        except requests.RequestException:
            pass
        return "ERROR" # Return 'ERROR' for failure

    def fetch_languages(self, user: str, repo: str) -> Dict[str, str]:
        """Fetch language breakdown with percentages"""
        url = f"https://api.github.com/repos/{user}/{repo}/languages"
        try:
            r = self.session.get(url, headers=BASE_HEADERS, timeout=10)
            if r.status_code == 200:
                lang_data = r.json()
                total = sum(lang_data.values()) or 1
                return {lang: f"{(count / total) * 100:.1f}%" for lang, count in lang_data.items()}
        except requests.RequestException:
            pass
        return {}

    def fetch_contributor_count(self, user: str, repo: str) -> int:
        """Fetch total contributor count"""
        url = f"https://api.github.com/repos/{user}/{repo}/contributors?anon=1&per_page=100"
        try:
            count = 0
            page = 1
            while True:
                r = self.session.get(f"{url}&page={page}", headers=BASE_HEADERS, timeout=10)
                if r.status_code != 200:
                    break
                data = r.json()
                if not data:
                    break
                count += len(data)
                if len(data) < 100:
                    break
                page += 1
            return count
        except requests.RequestException:
            return "ERROR" # Return 'ERROR' for failure
    
    def fetch_readme(self, user: str, repo: str) -> Optional[str]:
        """Fetch README content"""
        for branch in ("main", "master"):
            for fname in ("README.md", "readme.md", "README.MD"):
                url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{fname}"
                try:
                    r = self.session.get(url, headers=RAW_README_HEADERS, timeout=10)
                    if r.status_code == 200:
                        # Use BeautifulSoup to clean HTML/Markdown artifact
                        return BeautifulSoup(r.text, "html.parser").get_text().lower()
                except requests.RequestException:
                    continue
        return None


class LLMDetector:
    """Detects LLM mentions in text (from getRepoWithLLM.py)"""
    @staticmethod
    def detect_llm_from_text(text: str) -> str:
        if not text:
            return ""
        found = set()
        for match in LLM_PATTERN.findall(text):
            label = KEYWORD_TO_LABEL.get(match.lower())
            if label:
                found.add(label)
        return ",\n".join(sorted(found)) if found else ""


# -------------------------------
# Single-Pass Extraction Function
# -------------------------------
def fetch_complete_repo_data(client: GitHubAPIClient, user: str, repo: str) -> Dict:
    """
    Fetch ALL data for a single repo in one pass. 
    Returns a dictionary with error placeholders if fetch fails.
    """
    url = f"https://github.com/{user}/{repo}"
    
    # Fetch basic metadata (Includes size)
    metadata = client.fetch_repo_metadata(user, repo)
    
    # Fetch PR/Issue counts
    counts = client.fetch_issue_pr_counts(user, repo)
    
    # Check if metadata fetching failed. If so, skip
    if metadata["error"]:
        # Use placeholders for the detail fields
        readme_text = None
        langs = {}
        contribs = "ERROR"
        llm_used = ""
        has_readme = "No"
    else:
        # Fetch README
        readme_text = client.fetch_readme(user, repo)
        has_readme = "Yes" if readme_text else "No"
        
        # Detect LLM usage
        llm_used = LLMDetector.detect_llm_from_text(readme_text or "")
        
        # Fetch languages
        langs = client.fetch_languages(user, repo)
        
        # Fetch contributors
        contribs = client.fetch_contributor_count(user, repo)
    
    # Compile the complete result dictionary
    return {
        "Repo Name": f"{user}/{repo}",
        "URL": url,
        "LLM used": llm_used,
        "Have README": has_readme,
        "Description": metadata["description"],
        "Topics": metadata["topics"],
        "Languages": ",\n".join([f"{k}: {v}" for k, v in langs.items()]),
        "Contributors": contribs,
        "Total PRs": counts["pr_total"],
        "Open PRs": counts["pr_open"],
        "Closed PRs": counts["pr_closed"],
        "Total Issues": counts["issue_total"],
        "Open Issues": counts["issue_open"],
        "Closed Issues": counts["issue_closed"],
        "Size (KB)": metadata["size_kb"],
    }

# -------------------------------
# Sorting Logic 
# -------------------------------
def clean_value(x):
    """Clean empty/NaN values"""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return "" if not s or s.lower() == "nan" else s

def sort_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Sort dataset by quality: LLM repos first, then README repos, then others"""
    def sort_key(row):
        # Determine presence of key features
        has_llm = bool(str(row.get("LLM used", "")).strip())
        has_desc = bool(str(row.get("Description", "")).strip()) and str(row.get("Description")) != "Error fetching repo"
        has_topics = bool(str(row.get("Topics", "")).strip())
        has_readme = row.get("Have README", "") == "Yes"

        # Primary sort: LLM > README > No README
        if has_llm:
            category = 0
        elif has_readme:
            category = 1
        else:
            category = 2

        # Secondary sort: Full metadata > Partial > None
        if has_desc and has_topics:
            subcategory = 0
        elif has_desc:
            subcategory = 1
        elif has_topics:
            subcategory = 2
        else:
            subcategory = 3

        return (category, subcategory)

    df = df.copy()
    df["__sort_key__"] = df.apply(sort_key, axis=1)
    df = df.sort_values("__sort_key__").drop(columns="__sort_key__").reset_index(drop=True)
    return df


# -------------------------------
# Main Extraction Pipeline
# -------------------------------
def load_unique_repos(filtered_json_path: str) -> List[Tuple[str, str]]:
    """Load unique (user, repo) tuples (from extract_reposity_info.py)"""
    try:
        with open(filtered_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        unique = {(entry["user"], entry["repo"]) for entry in data}
        return sorted(unique)
    except FileNotFoundError:
        print(f"Error: Input file '{filtered_json_path}' not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filtered_json_path}'.")
        return []


def extract_repository_information(
    filtered_json_path: str = "filtered_prompt_files.json",
    repository_dataset_csv: str = "repository_dataset.csv",
):
    """
    The unified pipeline: Extract ALL data in one pass, then sort at the end.
    """
    if not GITHUB_TOKEN:
        print("[WARN] GITHUB_TOKEN is not set. API rate limits may be hit quickly.")

    session = requests.Session()
    client = GitHubAPIClient(session)

    # Load unique repos
    unique_repos = load_unique_repos(filtered_json_path)
    if not unique_repos:
        return
        
    print(f"Found {len(unique_repos)} unique repos. Extracting complete data...")

    # Extract ALL data 
    results = []
    
    for user, repo in tqdm(unique_repos, desc="Extracting repos"):
        # Fetch complete data.
        repo_data = fetch_complete_repo_data(client, user, repo)
        results.append(repo_data)
        time.sleep(RATE_LIMIT_SLEEP) 

    # Create DataFrame and clean values
    df = pd.DataFrame(results)
    df["Description"] = df["Description"].map(clean_value)
    df["Topics"] = df["Topics"].map(clean_value)

    # Sort once at the end
    print("Sorting dataset...")
    final_df = sort_final_dataset(df)
    
    # Save
    final_df.to_csv(repository_dataset_csv, index=False, encoding="utf-8")
    print(f"\n✓ Saved {len(final_df)} repositories to {repository_dataset_csv}")
    
    # Print summary statistics
    llm_count = (final_df["LLM used"] != "").sum()
    print(f"\nSummary: {len(final_df)} total repos, {llm_count} with LLM usage detected.")


if __name__ == "__main__":
    extract_repository_information()