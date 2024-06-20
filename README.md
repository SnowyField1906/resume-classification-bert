# Usage

## Requirements

- `python` and `pip` are required to run the program.
- `miniconda` is recommended to manage the environment.

## Environment setting up

```bash
# Create new Conda environment
conda create -n resume-classification-bert python=3.10.13
conda activate resume-classification-bert

# Install dependencies
pip install -r requirements.txt
```

## Start backend server

```bash
python app.py
```

- Default port: 5000
- API endpoint:
  - `POST /train` - `{loss: string, acc: string}`: Train the model.
  - `GET /process?content={string}` - `{[role: string]: string}`: Read content from PDF file and return the classification result.

## Run code directly without activating backend server (for debugging purpose)

Uncomment the following code in `main.py` and replace (or change the path of) the destination PDF file.

```python
def load(content=None):
    # from tika import parser
    # content = str(parser.from_file("./model/assets/resume.pdf")["content"])
    
    text_preprocessor = TextPreprocessor()
    data_frame = DataFrame("./model/assets/dataset.csv", "Resume", "Category")
```

```bash
python main.py
```

- default port: 5000
- api endpoint:
  - `POST /train` - `{loss: string, acc: string}`: train the model
  - `GET /process?content={string}` - `{[role: string]: string}`: read content from PDF file and return the classification result

## Linting and formatting

```bash
python -m black .
```