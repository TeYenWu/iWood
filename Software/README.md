# Software of iWood
This page gives the instruction how to run iWood data collection program, in the process of iWood fabrication, and also includes the examples of making smart item using iWood.
Please carefully follow the below steps for the best performance of iWood.


## Installation
1. install [Anaconda](https://docs.anaconda.com/anaconda/install/)
2. install [Processing](https://processing.org)
3. install conda environment using environment.yml (only working on MacOS)
```
conda env create -f server/environment.yml
```

## Data collection
1. Activate conda environment
```
conda activate iWood
```
2. Run main.py in server
```
python server/main.py
```
3. Enter 0 to set server as data collection mode 
```
"Please enter what you are going to do? (0: data), (1:demo) :\n"
0
```
4. Enter the item name 
5. Enter the participant name
6. Open Processing and Run VisualizationForData.pde
```
R: Start Record
S: Stop Record
X: Overwrite last data
Z: Recalibration
```
P.S. You could change the activity name in the main.py and VisualizationForData.pde to collect specific activity


## Training
1. Edit "things" and "activities" to change the targe model
2. Activate conda environment
```
conda activate iWood
```
3. Run train_for_demo.py in server
```
python server/train_for_demo.py
```

## Demo
1. Activate conda environment
```
conda activate iWood
```
2. Run main.py in server
```
python server/main.py
```
3. Enter 10 to set server as demo mode 
```
"Please enter what you are going to do? (0: data), (1:demo) :\n"
1
```
4. Open Processing and Run VisualizationForData.pde

