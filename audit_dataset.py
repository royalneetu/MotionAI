"""Audit train/validation/test video splits for byte-identical files and class counts.
Run: python audit_dataset.py --dataset dataset
This is a screening tool, not proof that visually similar clips are independent.
"""
import argparse, hashlib
from collections import defaultdict
from pathlib import Path

EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='dataset')
    args = p.parse_args()
    root = Path(args.dataset)
    if not root.exists():
        print(f'Dataset folder not found: {root.resolve()}'); return 2
    hashes = defaultdict(list)
    print('Dataset split summary')
    for split in ('train', 'validation', 'test'):
        split_dir = root / split
        files = [x for x in split_dir.rglob('*') if x.is_file() and x.suffix.lower() in EXTS] if split_dir.exists() else []
        classes = defaultdict(int)
        for f in files:
            classes[f.parent.name] += 1
            hashes[sha256(f)].append((split, f))
        print(f'  {split}: {len(files)} videos | ' + (', '.join(f'{k}: {v}' for k, v in sorted(classes.items())) or 'no files'))
    duplicated = [(digest, items) for digest, items in hashes.items() if len({s for s, _ in items}) > 1]
    if duplicated:
        print(f'\nWARNING: {len(duplicated)} byte-identical video(s) occur across multiple splits:')
        for _, items in duplicated:
            print('  ' + ' <-> '.join(f'{s}/{f.relative_to(root)}' for s, f in items))
        print('Do not treat evaluation metrics as independent until these duplicates are removed and the model is retrained.')
        return 1
    print('\nNo byte-identical cross-split duplicates found. This does not detect re-encoded or visually similar clips.')
    return 0
if __name__ == '__main__': raise SystemExit(main())
