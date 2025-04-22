import json
import codecs
import math
import numpy as np
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from multiprocessing import Pool, cpu_count
import itertools
import os # Needed for file operations
import pickle # Needed for saving/loading state
from tqdm import tqdm



stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return tokens

def process_line(line):
    processed_line = line.strip()
    if processed_line.endswith(','):
        processed_line = processed_line[:-1]
    if not processed_line:
        return None
    try:
        line_json = json.loads(processed_line)
        pmid = line_json.get("pmid", None)
        abstract = line_json.get("abstractText", "")
        if pmid and abstract:
            tokens = preprocess(abstract)
            return pmid, Counter(tokens), len(tokens)
    except json.JSONDecodeError as e:
        return None
    except Exception as e:
        return None
    return None



def build_index_parallel(corpus_path,
                         checkpoint_path="checkpoint.pkl",
                         save_every_n_docs=100000,
                         chunk_size=1000,
                         num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()

    index = defaultdict(list)
    doc_metadata = {}
    tf = {}
    df = Counter()
    total_docs = 0
    total_dl = 0
    processed_lines_count = 0

    # --- Load from checkpoint if exists ---
    if os.path.exists(checkpoint_path):
        print(f"--- Checkpoint found at {checkpoint_path}. Resuming... ---")
        try:
            with open(checkpoint_path, 'rb') as f_checkpoint:
                saved_state = pickle.load(f_checkpoint)
                index = saved_state['index']
                doc_metadata = saved_state['doc_metadata']
                tf = saved_state['tf']
                df = saved_state['df']
                total_docs = saved_state['total_docs']
                total_dl = saved_state['total_dl']
                processed_lines_count = saved_state['processed_lines_count']
                print(f"--- Resumed state: {total_docs} documents processed from {processed_lines_count} lines. ---")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            index, doc_metadata, tf, df = defaultdict(list), {}, {}, Counter()
            total_docs, total_dl, processed_lines_count = 0, 0, 0
    else:
        print("--- No checkpoint found. Starting from scratch. ---")

    last_save_doc_count = total_docs

    try:
        with codecs.open(corpus_path, 'r', encoding='utf-8', errors='ignore') as corpus, \
             Pool(processes=num_workers) as pool:

            if processed_lines_count > 0:
                print(f"Skipping first {processed_lines_count} lines from input file...")
                for _ in range(processed_lines_count):
                    next(corpus)
                print("Skipping complete.")

            current_line_nr = processed_lines_count
            doc_id_counter = total_docs

            while True:
                lines_chunk = list(itertools.islice(corpus, chunk_size))
                if not lines_chunk:
                    break

                chunk_results = pool.map(process_line, lines_chunk, chunksize=max(1, chunk_size // (num_workers * 2)))

                lines_in_chunk = len(lines_chunk)
                del lines_chunk

                for result in chunk_results:
                    if result is None or not isinstance(result, tuple) or len(result) != 3:
                        continue  # Explicitly skip invalid results

                    pmid, doc_tf_counter, doc_length = result
                    current_doc_id = doc_id_counter

                    doc_metadata[current_doc_id] = {"pmid": pmid, "length": doc_length}
                    tf[current_doc_id] = doc_tf_counter
                    total_dl += doc_length
                    total_docs += 1

                    for token, freq in doc_tf_counter.items():
                        df[token] += 1
                        index[token].append(current_doc_id)

                    doc_id_counter += 1

                current_line_nr += lines_in_chunk
                print(f"Processed up to line ~{current_line_nr}. Total documents: {total_docs}")

                if total_docs >= last_save_doc_count + save_every_n_docs:
                    print(f"\n--- Saving checkpoint at {total_docs} documents (processed ~{current_line_nr} lines)... ---")
                    checkpoint_state = {
                        'index': index,
                        'doc_metadata': doc_metadata,
                        'tf': tf,
                        'df': df,
                        'total_docs': total_docs,
                        'total_dl': total_dl,
                        'processed_lines_count': current_line_nr
                    }
                    temp_checkpoint_path = checkpoint_path + ".tmp"
                    try:
                        with open(temp_checkpoint_path, 'wb') as f_temp_checkpoint:
                            pickle.dump(checkpoint_state, f_temp_checkpoint, protocol=pickle.HIGHEST_PROTOCOL)
                        os.replace(temp_checkpoint_path, checkpoint_path)
                        print(f"--- Checkpoint saved successfully to {checkpoint_path} ---")
                        last_save_doc_count = total_docs
                    except Exception as e:
                        print(f"!!! Error saving checkpoint: {e} !!!")
                        if os.path.exists(temp_checkpoint_path):
                            try:
                                os.remove(temp_checkpoint_path)
                            except OSError:
                                pass
                    del checkpoint_state # Free memory

    except Exception as e:
        print(f"\n!!! An error occurred during processing: {e} !!!")
        print("Please check the error message. Try resuming the script to continue from the last checkpoint.")
        raise

    # --- Final calculations (after loop finishes) ---
    if total_docs == 0:
        print("Warning: No documents processed.")
        return index, {}, {}, {}, 0

    avg_dl = total_dl / total_docs
    # Using +1 smoothing for IDF
    idf = {token: math.log((total_docs + 1) / (freq + 1)) + 1 for token, freq in df.items()}

    print(f"\n--- Finished building index. Total documents: {total_docs} ---")

    # --- Optional: Final save after successful completion ---
    print("--- Saving final state... ---")
    final_state = {
        'index': index, 'doc_metadata': doc_metadata, 'tf': tf, 'df': df,
        'total_docs': total_docs, 'total_dl': total_dl,
        'processed_lines_count': current_line_nr # Save final line count
    }
    temp_checkpoint_path = checkpoint_path + ".tmp"
    try:
        with open(temp_checkpoint_path, 'wb') as f_temp_checkpoint:
            pickle.dump(final_state, f_temp_checkpoint, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_checkpoint_path, checkpoint_path)
        print(f"--- Final state saved successfully to {checkpoint_path} ---")
    except Exception as e:
        print(f"!!! Error saving final state: {e} !!!")

    return index, doc_metadata, tf, idf, avg_dl

# index, doc_ids, tf, idf, avg_dl = build_index_parallel(json_file)

if __name__ == '__main__':
    json_file = '/Users/greinaldpappa/Documents/GitHub/SS25AIR_Group14/data/training13b.json'
    checkpoint_file = 'mesh_index_checkpoint.pkl'
    index, doc_metadata, tf, idf, avg_dl = build_index_parallel(
        json_file,
        checkpoint_path=checkpoint_file,
        save_every_n_docs=100000,
        chunk_size=1000,
        num_workers=None
    )
