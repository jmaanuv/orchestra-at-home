
# Orchestra at Home

Orchestra at Home is a desktop application designed to provide users with a unique and interactive music control experience. Utilizing camera sensors to capture body movements, the application allows users to dynamically control music volume, creating an immersive home orchestra experience. The underlying model, trained with self-generated data and meticulously labeled actions, recognizes specific movements, triggering corresponding music playback.

## Technologies used

- Python 3.10
- TKinter
- TensorFlow
Additional dependencies are listed in the `requirements.txt` file.

## Run it yourself

## Installation
Follow these steps to set up and run the application:

### Step 1: Clone the Project

```bash
git clone https://github.com/jmaanuv/orchestra-at-home
cd OrchestraAtHome
```

### Step 2: Install Dependencies
Ensure you have Python 3.10.x installed and then run:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
Execute the following command to launch the application:
```bash
python gui.py
```
this step takes a little time so be patient


## Training the Model (Optional)
If you wish to train the model yourself, follow these steps:

### Step 1: Data Collection
Run the data_collection.py script and make necessary changes to the actions that need to be trained.

### Step 2: Train and Save the Model
Execute the training_and_saving_model.py script to train the model and save it.

Make sure to review and modify the scripts as needed before running them.



