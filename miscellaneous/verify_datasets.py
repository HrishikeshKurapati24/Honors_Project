import os
import hashlib

def get_all_files(directory):
    file_list = {}
    for root, _, files in os.walk(directory):
        for f in files:
            if f == ".DS_Store":
                continue
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, directory)
            file_list[rel_path] = full_path
    return file_list

def get_file_hash(filepath):
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None
    return hasher.hexdigest()

def compare_directories(dir1, dir2):
    print(f"--- Comparing Directories ---")
    print(f"Dir 1 (D1): {dir1}")
    print(f"Dir 2 (D2): {dir2}")
    print(f"-"*40)
    
    if not os.path.exists(dir1):
        print(f"Error: Directory {dir1} does not exist.")
        return
    if not os.path.exists(dir2):
        print(f"Error: Directory {dir2} does not exist.")
        return

    files1 = get_all_files(dir1)
    files2 = get_all_files(dir2)
    
    only_in_1 = set(files1.keys()) - set(files2.keys())
    only_in_2 = set(files2.keys()) - set(files1.keys())
    common_files = set(files1.keys()).intersection(set(files2.keys()))
    
    if only_in_1:
        print("\\nFiles only in D1:")
        for f in sorted(only_in_1):
            print(f"  - {f}")
            
    if only_in_2:
        print("\\nFiles only in D2:")
        for f in sorted(only_in_2):
            print(f"  - {f}")
            
    diff_content = []
    same_content_count = 0
    
    print("\\nComparing contents for common files...")
    
    for f in sorted(common_files):
        hash1 = get_file_hash(files1[f])
        hash2 = get_file_hash(files2[f])
        
        if hash1 is None or hash2 is None:
            print(f"Skipping content comparison for {f} due to read error.")
            continue
            
        if hash1 != hash2:
            size1 = os.path.getsize(files1[f])
            size2 = os.path.getsize(files2[f])
            diff_content.append((f, size1, size2))
        else:
            same_content_count += 1
            
    if diff_content:
        print("\\nFiles present in both but with DIFFERENT contents:")
        for f, s1, s2 in diff_content:
            print(f"  - {f}")
            print(f"      Size in D1: {s1} bytes")
            print(f"      Size in D2: {s2} bytes")
    else:
        print("\\nAll common files have identical content.")
    
    print("\\n" + "="*40)
    print("Summary:")
    print(f"  Files only in D1: {len(only_in_1)}")
    print(f"  Files only in D2: {len(only_in_2)}")
    print(f"  Files with differing contents: {len(diff_content)}")
    print(f"  Files with identical contents: {same_content_count}")
    print("="*40)

if __name__ == "__main__":
    dir1 = "/Volumes/Work/Semester - 6/Honors/CDRP models testing/3OmicsBenchmarking/dataset-1"
    dir2 = "/Volumes/Work/Semester - 6/Honors/CDRP models testing/benchmark models/GraphCDR/data"
    compare_directories(dir1, dir2)
