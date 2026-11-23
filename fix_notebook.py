import json

path = r'c:\Users\rupam maity\Desktop\Dimensionality Reduction\Dimensionality-Reduction\01_PCA_Analysis.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if '# Load dataset' in src and 'read_csv' in src:
            cell['source'] = [
                "# Load dataset\n",
                "import os, pathlib\n",
                "notebook_dir = pathlib.Path(globals().get('__vsc_ipynb_file__', os.path.abspath('__file__'))).parent\n",
                "df = pd.read_csv(notebook_dir / 'student_grades.csv')\n",
                "\n",
                "# Define the 8 core subject columns (excluding student_id)\n",
                'SUBJECTS = ["math", "science", "cs", "band", "english", "history", "spanish", "physed"]\n',
                "\n",
                'print(f"Dataset Shape: {df.shape[0]} students x {df.shape[1]} columns")\n',
                'print("\\n--- Data Info ---")\n',
                "df.info()\n",
                "\n",
                'print("\\n--- Missing Values per Column ---")\n',
                "print(df.isnull().sum())\n",
                "\n",
                'print(f"\\nDuplicate Rows: {df.duplicated().sum()}")\n',
                "df.head()\n",
            ]
            cell['outputs'] = []
            print('Fixed cell!')
            break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print('Saved.')
