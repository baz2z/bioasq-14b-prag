import json
from collections import Counter
import argparse

def jaccard_similarity(text1, text2):
    """Calculate Jaccard similarity between two texts based on words"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if len(union) > 0 else 0

def remove_subset_and_overlapping_snippets(snippets, overlap_threshold=0.9):
    """Remove snippets that are subsets of, or overlap >=90% with, another
    snippet from the same document and section (based on character offsets).

    When two snippets conflict, the shorter one is removed.
    """
    if len(snippets) <= 1:
        return snippets

    remove = set()

    for i in range(len(snippets)):
        if i in remove:
            continue
        si = snippets[i]
        doc_i = si.get("document", "")
        sec_i = (si.get("beginSection", ""), si.get("endSection", ""))
        beg_i, end_i = si["offsetInBeginSection"], si["offsetInEndSection"]
        len_i = end_i - beg_i

        for j in range(i + 1, len(snippets)):
            if j in remove:
                continue
            sj = snippets[j]
            # Must be same document and same section
            if sj.get("document", "") != doc_i:
                continue
            if (sj.get("beginSection", ""), sj.get("endSection", "")) != sec_i:
                continue

            beg_j, end_j = sj["offsetInBeginSection"], sj["offsetInEndSection"]
            len_j = end_j - beg_j

            # 1. Check if one is a complete subset of the other
            if beg_i <= beg_j and end_i >= end_j:
                remove.add(j)
                continue
            if beg_j <= beg_i and end_j >= end_i:
                remove.add(i)
                break  # i is removed, no need to check further

            # 2. Check overlap ratio (relative to the smaller snippet)
            overlap_start = max(beg_i, beg_j)
            overlap_end = min(end_i, end_j)
            overlap = max(0, overlap_end - overlap_start)
            smaller_len = min(len_i, len_j)
            if smaller_len > 0 and overlap / smaller_len >= overlap_threshold:
                # Remove the shorter snippet
                if len_i <= len_j:
                    remove.add(i)
                    break
                else:
                    remove.add(j)

    return [s for idx, s in enumerate(snippets) if idx not in remove]


def remove_similar_snippets(snippets, similarity_threshold=0.7):
    """Remove snippets with high word overlap"""
    if len(snippets) <= 1:
        return snippets

    # Keep track of which snippets to keep
    keep_indices = []

    for i, snippet in enumerate(snippets):
        text_i = snippet["text"]
        is_similar = False

        # Check against already kept snippets
        for j in keep_indices:
            text_j = snippets[j]["text"]
            similarity = jaccard_similarity(text_i, text_j)
            if similarity > similarity_threshold:
                is_similar = True
                break

        if not is_similar:
            keep_indices.append(i)

    return [snippets[i] for i in keep_indices]

def clean_bioasq_data(input_path, output_path, similarity_threshold=0.7):
    """Clean BioASQ data by removing similar snippets"""
    with open(input_path, "r") as fin:
        data = json.load(fin)

    questions = data["questions"]
    total_snippets_before = 0
    total_snippets_after = 0

    for question in questions:
        original_count = len(question["snippets"])
        total_snippets_before += original_count

        # Step 1: Remove subsets / high-overlap snippets by offset
        question["snippets"] = remove_subset_and_overlapping_snippets(
            question["snippets"]
        )
        after_offset = len(question["snippets"])

        # Step 2: Remove similar snippets by Jaccard word overlap
        question["snippets"] = remove_similar_snippets(
            question["snippets"],
            similarity_threshold
        )

        new_count = len(question["snippets"])
        total_snippets_after += new_count

        if original_count != new_count:
            print(f"Question {question['id']}: {original_count} -> {new_count} snippets")

    print(f"\nTotal snippets: {total_snippets_before} -> {total_snippets_after}")
    print(f"Removed {total_snippets_before - total_snippets_after} similar snippets")

    # Save cleaned data
    with open(output_path, "w") as fout:
        json.dump(data, fout, indent=2)

    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/training14b.json",
                        help="Input BioASQ file")
    parser.add_argument("--output", type=str, default="data/training14b_clean.json",
                        help="Output cleaned BioASQ file")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Similarity threshold for removing snippets (0.0-1.0)")

    args = parser.parse_args()
    clean_bioasq_data(args.input, args.output, args.threshold)
