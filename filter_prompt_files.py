import json
import re
from langdetect import detect

def count_english_words(text):
    return len(re.findall(r'\b[a-zA-Z]{2,}\b', text))

def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False

with open("clean-reader_prompt_metadata.json", "r", encoding="utf-8") as f:
    data = json.load(f)

filtered = []

for path, prompts in data.items():
    good_prompts = [p for p in prompts if count_english_words(p) > 15 and is_english(p)]
    if good_prompts:
        match = re.match(r"data/scraping/repos/([^~]+)~([^/]+)/(.+)", path)
        if match:
            user, repo, file_path = match.groups()
            filtered.append({
                "repo_path": path,
                "user": user,
                "repo": repo,
                "file_path": file_path.replace("~", "/"),
                "prompts": good_prompts
            })

with open("filtered_prompt_files.json", "w") as f:
    json.dump(filtered, f, indent=2)

print("Saved files completely")

