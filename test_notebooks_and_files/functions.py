import os
import nltk
import re
import json
import pickle
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def load_text_files(file: str, path: str = 'GoT_files', newLine: bool = False) -> str:
    with open(os.path.join(path, file), 'r', encoding='utf-8', errors='ignore') as f:
        if newLine:
            text = f.read().splitlines()
        else:
            text = f.read()

    return text

def load_graphs(graph: str, path: str = 'graphs'):
    with open(os.path.join(path,graph), 'rb') as f:
        loaded_graph = pickle.load(f)
    
    return loaded_graph

def load_all_files(path: str = 'GoT_files') -> dict:
    pages = os.listdir(path)

    page_titles = [f for f in pages if not f.startswith('fetched_pages_')]
    page_titles = [f for f in page_titles if not f.startswith('redirects_')]
    page_titles = [f for f in page_titles if not f.startswith('failed_pages_')]

    page_texts = {}

    for file in pages:
        # Skip files in Doubles subfolder and skip directories
        if file == 'Doubles' or os.path.isdir(os.path.join(path, file)):
            continue
            
        page_name =  file.replace(".txt", "")
        text = load_text_files(file,path)
        page_texts[page_name] = text
    
    return page_texts

def tokenize_text(text: str, rm_stopwords: bool = False) -> list:
    """
    Tokenize text into lowercase alphabetic words and remove stopwords.
    """
    # tokens = nltk.word_tokenize(text)
    # Filter out punctuation and convert to lowercase in one pass
    # tokens = [token.lower() for token in tokens]
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if rm_stopwords:
        STOPWORDS = set(ENGLISH_STOP_WORDS)
        tokens = [token for token in tokens if token not in STOPWORDS]
    return tokens


def load_labmit_s1(path: str = "data/Data_Set_S1.txt"):
    """
    Load S1 wordlist as {word: happiness_average}.
    Assumes a tab-separated file with columns:
    word, happiness_rank, happiness_average, ...
    """
    df = pd.read_csv(path, sep="\t")  # header is in the file

    # Normalize words to lowercase
    df["word"] = df["word"].astype(str).str.lower()

    # Build dict: word -> happiness_average
    lexicon = dict(zip(df["word"], df["happiness_average"].astype(float)))
    return lexicon

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

def compute_sentiment(tokens, lexicon):
    """
    Compute sentiment score for a list of tokens using a word→score lexicon.
    Here: arithmetic mean of scores for tokens that are in the lexicon.
    Returns: float or None if no tokens are in the lexicon.
    """
    scores = [lexicon[t] for t in tokens if t in lexicon]
    if not scores:
        return None
    return sum(scores) / len(scores)
    
def get_links(text: str, path: str = 'GoT_files', no_files: bool = True, no_translations: bool = True) -> list:

    f = open(os.path.join(path, text), 'r', encoding='utf-8')

    if no_files and no_translations:
        links = re.findall(r"\[{2}(?!\w{2,5}(?:-\w{2})?:)(?!File:)(?!Image:).+?\]{2}", f.read())
    elif no_files:
        links = re.findall(r"\[{2}(?!File:)(?!Image:).+?\]{2}", f.read())
    elif no_translations:
        links = re.findall(r"\[{2}(?!\w{2,5}(?:-\w{2})?:).+?\]{2}", f.read())
    
    return links

def get_links_by_section(file: str, path: str = 'GoT_files', no_files: bool = True, no_translations: bool = True) -> dict:
    """Extract links from a wiki page, organized by section.
    
    Args:
        file: Filename to read (e.g., 'Page Name.txt')
        path: Directory containing the file
        no_files: If True, exclude File: and Image: links
        no_translations: If True, exclude translation links (e.g., [[de:...]])
    
    Returns:
        Dictionary with structure:
        {
            'sections': {
                'Section Name': {
                    'subsections': {
                        'Subsection Name': ['link1', 'link2', ...],
                        ...
                    },
                    'links': ['link1', 'link2', ...]  # Links directly under this section
                },
                ...
            },
            'header': ['link1', 'link2', ...],  # Links before first section
            'categories': ['category1', 'category2', ...]  # Links after <!--Categories-->
        }
    """
    
    with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Build regex pattern for links
    if no_files and no_translations:
        link_pattern = r"\[\[(?!\w{2,5}(?:-\w{2})?:)(?!File:)(?!Image:)(.+?)\]\]"
    elif no_files:
        link_pattern = r"\[\[(?!File:)(?!Image:)(.+?)\]\]"
    elif no_translations:
        link_pattern = r"\[\[(?!\w{2,5}(?:-\w{2})?:)(.+?)\]\]"
    else:
        link_pattern = r"\[\[(.+?)\]\]"
    
    result = {
        'sections': {},
        'header': [],
        'categories': []
    }
    
    # Split text into parts
    lines = text.split('\n')
    
    current_section = None
    current_subsection = None
    in_categories = False
    
    for i, line in enumerate(lines):
        # Check if we've reached categories
        if '<!--Categories-->' in line:
            in_categories = True
            continue
        
        # Check for section headers
        section_match = re.match(r'^==(.*?)==\s*$', line)
        subsection_match = re.match(r'^===(.*?)===\s*$', line)
        
        if section_match and not subsection_match:
            # Main section (== Section ==)
            current_section = section_match.group(1).strip()
            current_subsection = None
            if current_section not in result['sections']:
                result['sections'][current_section] = {
                    'subsections': {},
                    'links': []
                }
        elif subsection_match:
            # Subsection (=== Subsection ===)
            current_subsection = subsection_match.group(1).strip()
            if current_section:
                if current_subsection not in result['sections'][current_section]['subsections']:
                    result['sections'][current_section]['subsections'][current_subsection] = []
        
        # Extract links from this line
        matches = re.findall(link_pattern, line)
        
        for match in matches:
            # Clean the link (remove pipe syntax)
            if '|' in match:
                link = match.split('|')[0]
            else:
                link = match
            
            # Skip category links in normal sections
            if link.startswith('Category:') or link.startswith('category:'):
                if in_categories:
                    result['categories'].append(link.replace('Category:', '').replace('category:', ''))
                continue
            
            # Add to appropriate location
            if in_categories:
                continue  # Skip non-category links after <!--Categories-->
            elif current_subsection and current_section:
                result['sections'][current_section]['subsections'][current_subsection].append(link)
            elif current_section:
                result['sections'][current_section]['links'].append(link)
            else:
                result['header'].append(link)
    
    return result


def plot_character_network(G, weighted=False, top_n=10):
    """
    Plot the largest weakly-connected component of a directed character network.

    - Background: light grey
    - Nodes: size ~ degree, color ~ in-degree
    - Edges: grey; width scaled if weighted=True
    - Labels: top_n nodes by degree, white, fontsize=15
    - Layout: forceatlas2_layout
    """

    # Extract largest weakly connected component
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    G_main = G.subgraph(largest_cc).copy()

    # Compute layout
    pos = nx.forceatlas2_layout(G_main)

    # Degree and in-degree
    degrees = dict(G_main.degree())
    in_degrees = dict(G_main.in_degree())

    # Node sizes
    node_sizes = [degrees[n] * 20 for n in G_main.nodes()]

    # Node colors (in-degree)
    node_color_values = [in_degrees[n] for n in G_main.nodes()]

    # Figure
    fig, ax = plt.subplots(figsize=(15, 12))
    fig.patch.set_facecolor("lightgrey")
    ax.set_facecolor("lightgrey")

    # Edge widths
    if weighted:
        weights = [G_main[u][v].get('weight', 1) for u, v in G_main.edges()]
        max_w = max(weights)
        min_w = min(weights)
        edge_widths = [
            0.5 + 4.5 * (w - min_w) / (max_w - min_w)
            if max_w > min_w else 2
            for w in weights
        ]
    else:
        edge_widths = 1

    # Draw edges
    nx.draw_networkx_edges(
        G_main,
        pos,
        width=edge_widths,
        edge_color="gray",
        alpha=0.5,
        arrows=True if G_main.is_directed() else False,
        arrowsize=10,
        ax=ax
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G_main,
        pos,
        node_size=node_sizes,
        node_color=node_color_values,
        cmap=plt.cm.viridis,
        alpha=0.85,
        ax=ax
    )

    # Labels for top nodes
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels = {node: node for node, _ in top_nodes}
    nx.draw_networkx_labels(
        G_main,
        pos,
        labels=labels,
        font_color='white',
        font_size=15,
        ax=ax
    )

    ax.axis('off')
    plt.tight_layout()
    plt.show()
