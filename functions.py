import os
import nltk
import re
import json

def load_text_files(file, path='GoT_files', newLine = False):    
    with open(os.path.join(path, file), 'r', encoding='utf-8', errors='ignore') as f:
        if newLine:
            text = f.read().splitlines()
        else:
            text = f.read()

    return text

def tokenize_text(text):
    tokens = nltk.word_tokenize(text)

    # Filter out punctuation and convert to lowercase in one pass
    tokens = [token.lower() for token in tokens]
    return tokens


def check_categories(text, unique_categories = []):
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

def clean_categories(raw_categories, category_map):
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

def load_category_mapping(file='category_mapping.json', path='data_handling'):
    with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
        category_map = json.load(f)
    
    return category_map
    

def get_catgories(text, raw_categories = [], category_map = []):
    if category_map == []:
        category_map = load_category_mapping()
    if raw_categories == []:
        raw_categories = check_categories(text)
    
    cleaned_categories = clean_categories(raw_categories, category_map)

    return cleaned_categories
    
