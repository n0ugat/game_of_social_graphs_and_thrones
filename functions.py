import os
import nltk
import re
import json

def load_text_files(file: str, path: str = 'GoT_files', newLine: bool = False) -> str:
    with open(os.path.join(path, file), 'r', encoding='utf-8', errors='ignore') as f:
        if newLine:
            text = f.read().splitlines()
        else:
            text = f.read()

    return text

def tokenize_text(text: str) -> list:
    tokens = nltk.word_tokenize(text)

    # Filter out punctuation and convert to lowercase in one pass
    tokens = [token.lower() for token in tokens]
    return tokens


def check_categories(text: str, unique_categories: list = []) -> list:
    """Extract wiki [[Category:...]] values from raw text and append to unique_categories.

    Uses a regex that handles single or double brackets and is case-insitive.
    Returns the updated unique_categories list.
    """

    # Match [[Category:...]] (captures the inner content up to the first closing ]])
    pattern = re.compile(r"\[\[\s*Category\s*:\s*(.+?)\s*\]\]", flags=re.IGNORECASE)
    matches = pattern.findall(text)

    for m in matches:
        cat = m.strip()
        # normalize whitespace inside category
        cat = ' '.join(cat.split())
        if cat not in unique_categories:
            unique_categories.append(cat)

    return unique_categories

def clean_categories(raw_categories: list, category_map: dict) -> list:
    """Map raw categories to normalized forms and remove duplicates while preserving order.
    
    Args:
        raw_categories: List of raw category strings
        category_map: Dictionary mapping raw -> normalized category strings
        
    Returns:
        List of unique normalized category strings in order of first appearance
    """
    normalized = []
    seen = set()
    
    for cat in raw_categories:
        # Map to normalized form (use raw if not in mapping)
        norm = category_map.get(cat, cat)
        
        # Only add if we haven't seen this normalized category yet
        if norm not in seen:
            normalized.append(norm)
            seen.add(norm)
    
    return normalized

def load_category_mapping(file: str = 'category_mapping.json', path: str = 'data_handling'):
    with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
        category_map = json.load(f)
    
    return category_map
    

def get_catgories(text: str, raw_categories: list = [], category_map: list = []) -> list:
    if category_map == []:
        category_map = load_category_mapping()
    if raw_categories == []:
        raw_categories = check_categories(text)
    
    cleaned_categories = clean_categories(raw_categories, category_map)

    return cleaned_categories
    
def get_links(text: str, path: str = 'GoT_files', no_files: bool = True, no_translations: bool = True) -> list:

    f = open(os.path.join(path, text), 'r', encoding='utf-8')

    if no_files and no_translations:
        links = re.findall(r"\[{2}(?!\w{2,5}(?:-\w{2})?:)(?!File:)(?!Image:).+?\]{2}", f.read())
    elif no_files:
        links = re.findall(r"\[{2}(?!File:)(?!Image:).+?\]{2}", f.read())
    elif no_translations:
        links = re.findall(r"\[{2}(?!\w{2,5}(?:-\w{2})?:).+?\]{2}", f.read())
    
    return links