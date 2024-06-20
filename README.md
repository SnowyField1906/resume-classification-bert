## Fundamentals

### Requirements

- `python` and `pip` are required to run the program.
- `miniconda` is recommended to manage the environment.

### Setup the environment

```bash
# Create new Conda environment
conda create -n resume-classification-bert python=3.10.13
conda activate resume-classification-bert

# Install dependencies
pip install -r requirements.txt
```

### Train the model

```bash
python main.py
```

### Alternative approach: Download the model

The training process took around **2 hrs 30 mins** to complete on _MacBook Pro 2021 M1 Pro 16GB_.

To skip this time-consuming process, you can download the trained model we uploaded on _Google Drive (775M)_. [drive.google.com/file/d/1r93iNQdgTqOlgDOm37oNESkX9PoXr_Ln](https://drive.google.com/file/d/1r93iNQdgTqOlgDOm37oNESkX9PoXr_Ln) and put it in the `model/assets` folder.

## Usage

### Start backend server

```bash
python app.py
```

- Default port: 5000
- API endpoint:
  - `POST /train` - `{loss: string, acc: string}`: Train the model.
  - `GET /process?content={string}` - `{[role: string]: string}`: Read content from PDF file and return the classification result.

### Alternative approach: Run directly without activating backend server

```bash
python main.py "path/to/pdf/file"
```

Example:

```bash
python main.py "./model/assets/resume.pdf"
```

```json
{
    "Data Science": 0.01,
    "HR": 0.0,
    "Advocate": 0.0,
    "Arts": 0.0,
    "Web Designing": 0.0,
    "Mechanical Engineer": 0.0,
    "Sales": 0.0,
    "Health and fitness": 0.0,
    "Civil Engineer": 0.0,
    "Java Developer": 0.0,
    "Business Analyst": 0.0,
    "SAP Developer": 0.0,
    "Automation Testing": 0.0,
    "Electrical Engineering": 0.0,
    "Operations Manager": 0.0,
    "Python Developer": 0.0,
    "DevOps Engineer": 0.04,
    "Network Security Engineer": 0.0,
    "PMO": 0.0,
    "Database": 0.0,
    "Hadoop": 0.0,
    "ETL Developer": 0.0,
    "DotNet Developer": 0.0,
    "Blockchain": 99.96,
    "Testing": 0.0
}
```

### Linting and formatting

```bash
python -m black .
```