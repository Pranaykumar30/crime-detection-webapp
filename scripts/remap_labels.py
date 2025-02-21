import os

def remap_labels(label_dir):
    class_map = {
        46: 0,  # COCO 'gun' -> handguns
        43: 1,  # COCO 'knife' -> knives
        # Approximate sharp-edged-weapons with 'knife' for now; refine with custom training
        43: 2,  # Duplicate mapping (will overwrite to 1 unless filtered separately)
        0: 3,   # COCO 'person' -> masked-intruders (default; refine later with mask context)
        0: 4,   # COCO 'person' -> violence (default; refine later with aggression context)
        0: 5    # COCO 'person' -> normal-behavior (default; refine later with neutral context)
    }
    
    os.makedirs(label_dir, exist_ok=True)
    for label_file in os.listdir(label_dir):
        label_path = f"{label_dir}/{label_file}"
        if not os.path.isfile(label_path):
            continue
        
        with open(label_path, "r") as f:
            lines = f.readlines()
        
        with open(label_path, "w") as f:
            for line in lines:
                parts = line.split()
                if not parts:  # Skip empty lines
                    continue
                old_class = int(parts[0])
                # Custom logic to differentiate overlapping 'person' mappings
                if old_class == 0:  # Person detected
                    # For now, assume filename hints at class (manual curation needed later)
                    if "masked" in label_file.lower():
                        new_class = 3  # masked-intruders
                    elif "violence" in label_file.lower():
                        new_class = 4  # violence
                    else:
                        new_class = 5  # normal-behavior (default)
                elif old_class == 43:  # Knife
                    if "sharp" in label_file.lower():
                        new_class = 2  # sharp-edged-weapons
                    else:
                        new_class = 1  # knives
                else:
                    new_class = class_map.get(old_class, None)
                
                if new_class is not None:
                    f.write(f"{new_class} {' '.join(parts[1:])}\n")

if __name__ == "__main__":
    remap_labels("data/labels")