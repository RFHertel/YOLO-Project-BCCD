# scripts/analyze_and_undersample_porosity.py
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import json

def analyze_patches():
    """Analyze which patches contain which defects"""
    source_dir = Path(r'C:\AWrk\SWRD_YOLO_Project\processed_balanced_final\train')
    labels_dir = source_dir / 'labels'
    
    print("\n" + "="*70)
    print("PATCH ANALYSIS - Understanding Defect Distribution")
    print("="*70)
    print("\nClass ID Reference:")
    print("  0 = porosity")
    print("  1 = inclusion") 
    print("  2 = crack")
    print("  3 = undercut")
    print("  4 = lack_of_fusion")
    print("  5 = lack_of_penetration")
    
    print("\nCounting files...")
    label_files = list(labels_dir.glob('*.txt'))
    print(f"Found {len(label_files):,} patches to analyze\n")
    
    patch_info = []
    
    for label_file in tqdm(label_files, desc="Reading patches"):
        defect_classes = set()
        instance_counts = defaultdict(int)
        
        with open(label_file, 'r') as f:
            content = f.read().strip()
            if content:  # Has defects
                for line in content.split('\n'):
                    if line:
                        try:
                            class_id = int(float(line.split()[0]))
                            defect_classes.add(class_id)
                            instance_counts[class_id] += 1
                        except (ValueError, IndexError):
                            continue
                
                patch_info.append({
                    'filename': label_file.stem,
                    'type': 'defect',
                    'classes': list(defect_classes),
                    'num_classes': len(defect_classes),
                    'porosity_count': instance_counts.get(0, 0),
                    'inclusion_count': instance_counts.get(1, 0),
                    'crack_count': instance_counts.get(2, 0),
                    'undercut_count': instance_counts.get(3, 0),
                    'lack_fusion_count': instance_counts.get(4, 0),
                    'lack_penetration_count': instance_counts.get(5, 0),
                    'is_porosity_only': defect_classes == {0}
                })
            else:  # Background
                patch_info.append({
                    'filename': label_file.stem,
                    'type': 'background',
                    'classes': [],
                    'num_classes': 0,
                    'porosity_count': 0,
                    'inclusion_count': 0,
                    'crack_count': 0,
                    'undercut_count': 0,
                    'lack_fusion_count': 0,
                    'lack_penetration_count': 0,
                    'is_porosity_only': False
                })
    
    df = pd.DataFrame(patch_info)
    
    # Save CSV
    df.to_csv('patch_analysis.csv', index=False)
    
    # Print analysis results
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    print(f"\nTotal patches: {len(df):,}")
    print(f"Background patches: {len(df[df['type'] == 'background']):,}")
    print(f"Defect patches: {len(df[df['type'] == 'defect']):,}")
    
    defect_df = df[df['type'] == 'defect']
    print(f"\nPorosity-only patches: {defect_df['is_porosity_only'].sum():,}")
    print(f"Multi-class patches: {(defect_df['num_classes'] > 1).sum():,}")
    
    # Instance counts
    print("\n" + "-"*50)
    print("TOTAL INSTANCES BY CLASS:")
    print("-"*50)
    class_names = ['porosity', 'inclusion', 'crack', 'undercut', 'lack_fusion', 'lack_penetration']
    for i, class_name in enumerate(class_names):
        col_name = f'{class_name.replace(" ", "_")}_count'
        total = df[col_name].sum()
        patches_with = defect_df['classes'].apply(lambda x: i in x).sum()
        print(f"Class {i} ({class_name:20s}): {total:7,} instances in {patches_with:7,} patches")
    
    # CSV explanation
    print("\n" + "="*70)
    print("HOW TO READ THE CSV FILE (patch_analysis.csv):")
    print("="*70)
    print("\nEach row represents one patch. The columns are:")
    print("1. filename:           patch name")
    print("2. type:              'defect' or 'background'")
    print("3. classes:           list of defect types present [0,1,2,3,4,5]")
    print("4. num_classes:       how many different defect types")
    print("5. porosity_count:    number of class 0 bounding boxes")
    print("6. inclusion_count:   number of class 1 bounding boxes")
    print("7. crack_count:       number of class 2 bounding boxes")
    print("8. undercut_count:    number of class 3 bounding boxes")
    print("9. lack_fusion_count: number of class 4 bounding boxes")
    print("10. lack_penetration_count: number of class 5 bounding boxes")
    print("11. is_porosity_only: True if ONLY porosity, False otherwise")
    
    print("\n" + "-"*50)
    print("EXAMPLE CSV LINE:")
    print("-"*50)
    print("A_bam5_390957,defect,\"[0,2,5]\",3,3.0,0.0,15.0,0.0,0.0,1.0,False")
    print("\nThis means patch A_bam5_390957 contains:")
    print("  - 3 porosity defects (class 0)")
    print("  - 15 crack defects (class 2)")
    print("  - 1 lack_of_penetration defect (class 5)")
    print("  - Total: 19 defect instances across 3 different classes")
    print("  - is_porosity_only = False (has other defects besides porosity)")
    print("="*70)
    
    return df

def undersample_porosity(df, target_reduction=0.5):
    """Remove high-porosity patches to balance dataset"""
    
    source_dir = Path(r'C:\AWrk\SWRD_YOLO_Project\processed_balanced_final\train')
    defect_df = df[df['type'] == 'defect'].copy()
    
    # Calculate porosity ratio for each patch
    defect_df['total_defects'] = (defect_df['porosity_count'] + 
                                  defect_df['inclusion_count'] + 
                                  defect_df['crack_count'] + 
                                  defect_df['undercut_count'] + 
                                  defect_df['lack_fusion_count'] + 
                                  defect_df['lack_penetration_count'])
    
    defect_df['porosity_ratio'] = defect_df['porosity_count'] / defect_df['total_defects']
    
    # Find patches dominated by porosity (>80% porosity)
    high_porosity = defect_df[defect_df['porosity_ratio'] > 0.8].copy()
    print(f"\nFound {len(high_porosity):,} patches with >80% porosity")
    
    if len(high_porosity) == 0:
        print("No high-porosity patches to remove!")
        return
    
    # Sort by porosity count
    high_porosity = high_porosity.sort_values('porosity_count', ascending=False)
    
    # Remove top N%
    to_remove = high_porosity.head(int(len(high_porosity) * target_reduction))
    print(f"Removing {len(to_remove):,} high-porosity patches")
    
    # Remove files
    removed_list = []
    for filename in tqdm(to_remove['filename'], desc="Removing files"):
        img_path = source_dir / 'images' / f"{filename}.jpg"
        lbl_path = source_dir / 'labels' / f"{filename}.txt"
        
        if img_path.exists():
            img_path.unlink()
        if lbl_path.exists():
            lbl_path.unlink()
        removed_list.append(filename)
    
    print(f"Removed {len(removed_list):,} patches")

if __name__ == "__main__":
    import time
    start = time.time()
    
    df = analyze_patches()
    print(f"\nAnalysis completed in {time.time() - start:.1f} seconds")
    
    # Ask about undersampling
    response = input("\nRemove high-porosity patches? (y/n): ")
    if response.lower() == 'y':
        undersample_porosity(df, target_reduction=0.5)