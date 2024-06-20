## Fundamentals

### Requirements

- `python` and `pip` are required to run the program.
- `miniconda` is recommended to manage the environment.

### Environment setting up

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

To skip this time-consuming process, you can download the trained model we uploaded on _Google Drive_. [drive.google.com/file/d/1r93iNQdgTqOlgDOm37oNESkX9PoXr_Ln](https://drive.google.com/file/d/1r93iNQdgTqOlgDOm37oNESkX9PoXr_Ln) and put it in the `model/assets` folder.

## Usage

### Start backend server

```bash
python app.py
```

- Default port: 5000
- API endpoint:
  - `POST /train` - `{loss: string, acc: string}`: Train the model.
  - `GET /process?content={string}` - `{[role: string]: string}`: Read content from PDF file and return the classification result.

### Run code directly without activating backend server

```bash
python main.py "path/to/pdf/file"
```

### Linting and formatting

```bash
python -m black .
```