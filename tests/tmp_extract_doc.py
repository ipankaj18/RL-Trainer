from bs4 import BeautifulSoup
from pathlib import Path
html = Path('meta_llama_doc.html').read_text()
soup = BeautifulSoup(html, 'lxml')
for section in soup.find_all(['h1','h2','h3','p','li']):
    text = section.get_text(strip=True)
    if 'Special Tokens' in text or 'system message' in text or 'assistant header' in text:
        print(f"{section.name}: {text}\n")
