import os
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# MASK_DIR = "/home/vishnu/Ajitesh/ajitesh/patched_tif_masks/NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO(4)/masks" #- > 0.4%(385/94599) patches are having annotations 
MASK_DIR = "/home/vishnu/Ajitesh/ajitesh/patched_tif_masks/MURDANDA_450879_AWAPALLI_CHINTAKONTA_ORTHO(3)/masks" #- > 1.4%(1583/10697) patches are having annotations 
MIN_ANNO_RATIO = 0.02

def check_patch(f):
    if not f.endswith(".tif"):
        return None
    try:
        mask = tiff.imread(os.path.join(MASK_DIR, f))
        ratio = np.sum(mask > 0) / mask.size
        return ratio >= MIN_ANNO_RATIO  # True = has annotation
    except:
        return None

if __name__ == "__main__":
    files = os.listdir(MASK_DIR)
    
    workers = min(cpu_count(), 16)
    print(f"Using {workers} CPU workers...")

    with Pool(workers) as p:
        results = list(tqdm(p.imap(check_patch, files), 
                           total=len(files), desc="Checking"))

    results = [r for r in results if r is not None]
    total    = len(results)
    has_anno = sum(results)
    empty    = total - has_anno

    print(f"\nTotal patches     : {total}")
    print(f"Has annotation    : {has_anno} ({has_anno/total*100:.1f}%)")
    print(f"Empty (background): {empty} ({empty/total*100:.1f}%)")