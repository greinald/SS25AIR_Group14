import json
import time
from pathlib import Path
from Bio import Entrez

# === SETTINGS ===
EMAIL      = "apanci2000@gmail.com"           # your NCBI-registered email
API_KEY    = "a417cd398ae9ba5622989a1f8ef153750f08"  # your NCBI API key
Entrez.email   = EMAIL
Entrez.api_key = API_KEY

RETMX   = 1000     # number of documents to fetch per question

# Input/Output paths
INPUT_FILE  = Path("/Users/greinaldpappa/Desktop/Neural_Reranking/BioASQ-task13bPhaseA-testset4.json")
OUTPUT_FILE = Path("/Users/greinaldpappa/Desktop/Neural_Reranking/Fetching_Articles.json")

# === ESEARCH FUNCTION ===
def esearch_pmids(query: str, retmax: int = RETMX):
    """Fetch PMIDs for a given query string from PubMed."""
    phrase, tokens = query.lower(), query.lower().split()
    parts = [f'"{phrase}"[Title/Abstract]', f'"{phrase}"[MeSH Terms]']
    for t in tokens:
        parts += [f'{t}[Title/Abstract]', f'{t}[MeSH Terms]']
    term = f"({' OR '.join(parts)}) AND hasabstract[text]"

    for attempt in range(3):
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=term,
                retmax=retmax,
                sort="relevance",
                retmode="xml"
            )
            result = Entrez.read(handle)
            time.sleep(0.1)
            return result.get('IdList', [])
        except Exception:
            time.sleep(2 ** attempt)
    return []

# === MAIN PIPELINE ===
def main():
    data = json.loads(INPUT_FILE.read_text())['questions']
    output = []
    cnt = 1
    for item in data:
        print(f"On step {cnt} out of {len(data)} PMIDs")
        qid   = item['id']
        qtype = item.get('type')
        body  = item['body']

        # 1) fetch PMIDs
        pmids = esearch_pmids(body, RETMX)

        # 2) fetch titles and abstracts
        documents = []
        batch_size = 200  # smaller batches for large lists
        for start in range(0, len(pmids), batch_size):
            
            batch = pmids[start:start+batch_size]
            print(f"Fetching batch {start//batch_size+1} with {len(pmids)} PMIDs...")
            handle = Entrez.efetch(
                db="pubmed",
                id=','.join(batch),
                retmode="xml"
            )
            records = Entrez.read(handle)
            time.sleep(0.34)  # ~3 requests/sec per NCBI guidelines

            for rec in records.get('PubmedArticle', []):
                pmid = str(rec['MedlineCitation']['PMID'])
                art  = rec['MedlineCitation']['Article']
                title = art.get('ArticleTitle', '')
                abstract = ''
                if art.get('Abstract'):
                    # join multiple abstract texts if present
                    abstract = ' '.join(art['Abstract']['AbstractText'])

                documents.append({
                    'pmid': pmid,
                    'title': title,
                    'abstract': abstract
                })

        # 3) build output entry without ranking
        output.append({
            'id': qid,
            'type': qtype,
            'body': body,
            'documents': documents
        })
        
        cnt += 1
    # write to file
    OUTPUT_FILE.write_text(json.dumps({'questions': output}, indent=2))
    print(f"Fetched and saved {sum(len(q['documents']) for q in output)} documents in {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
