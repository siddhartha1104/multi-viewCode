import os
import pickle
from tqdm import tqdm

pickle_dir = '/home/sidx/myDrive/shimlaInternship/githubRepos/4MultiViewPaperCode/dataset/ZuCo/task2-NR-2.0/pickle'
task_name = 'task2-NR-2.0'
output_name = f'{task_name}-dataset-spectro.pickle'
output_path = os.path.join(pickle_dir, output_name)

pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.pickle') and not f.startswith(task_name)]
print(f"Found {len(pickle_files)} pickle files to combine.")

combined_dict = {}

for pf in tqdm(pickle_files):
    subject_name = pf.split('-')[0]  # Extract subject from filename like YAC-dataset-spectro.pickle
    with open(os.path.join(pickle_dir, pf), 'rb') as f:
        data_list = pickle.load(f)
    combined_dict[subject_name] = data_list

print(f"Combined {len(combined_dict)} subjects.")

with open(output_path, 'wb') as f:
    pickle.dump(combined_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved combined dataset to: {output_path}")
