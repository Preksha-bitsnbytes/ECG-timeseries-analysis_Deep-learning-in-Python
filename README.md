# MFA-matpy

# Matlab-to-Python-time-series-AI

Waveform Segmentation Using Deep Learning - example workflow using Matlab 
https://www.mathworks.com/help/signal/ug/waveform-segmentation-using-deep-learning.html

Use the same data set provided and follow the steps as guide to do the same thing in python

Steps:

### setup env
- Create new virtual env

- Install ipykernel within virtual env
`pip install ipykernel`

- Register your env as a jupytr kernel with name 'myenv'
`python -m ipykernel install --user --name=myenv --display-name="Python (myenv)"`

- Select the Correct Kernel in Jupyter Notebook

## To create requirements file for the repo pip freeze > requirements.txt


## Mat file setup
The data files are in .mat format containing columns with variable names 'Fs', 'ecgSignal' and a nested table for 'signalRegionLables' under 'None' variable denoting blank space. The nested table further needs to be flattened as currently it is in the format - ROI Limits column containing 2 sub columns limit1 and limit2. After flattening the table converts into 3 column matrix array - ROIlimit1, ROIlimit2, Label. 

These mat files could not be extracted as csv files or other readable formats in Python. 
So used MATLAB online to each mat file into corresponding ecgSignal and signalRegionLabels csv files. 

Following code was used in MATLAB:

% Specify the folder containing the .mat files
folder = 'path/to/your/folder';

% Get a list of all .mat files in the folder
matFiles = dir(fullfile(folder, '*.mat'));

% Loop through each .mat file
for i = 1:length(matFiles)
    % Load the .mat file
    fileName = fullfile(folder, matFiles(i).name);
    data = load(fileName);
    
    % Extract ecgSignal
    ecgSignal = data.ecgSignal;
    
    % Save ecgSignal as CSV
    ecgCsvName = [matFiles(i).name(1:end-4) '_ecgSignal.csv'];
    writematrix(ecgSignal, fullfile(folder, ecgCsvName));
    
    % Extract and flatten signalRegionLabels
    if isfield(data, 'signalRegionLabels')
        nestedTable = data.signalRegionLabels;
        flattenedTable = splitvars(nestedTable);
        
        % Save flattened table as CSV
        tableCsvName = [matFiles(i).name(1:end-4) '_signalRegionLabels.csv'];
        writetable(flattenedTable, fullfile(folder, tableCsvName));
    end
end





Install necessary libraries
pip install numpy scipy torch torchvision torchaudio


# Methodology

Model 1: Bidirectional LSTM
OR
Model 2: CNN (Convolutional Neural Network)

These two models are quite different in their architecture and approach to processing sequential data. Let’s compare them:
Model 1: Bidirectional LSTM
This model uses Bidirectional Long Short-Term Memory (LSTM) layers.
Key Features:
	1.	Bidirectional LSTM Layers: Processes sequences in both forward and backward directions, capturing context from both past and future states.
	2.	Sequential Processing: Designed to handle long-term dependencies in sequential data.
	3.	No Explicit Feature Extraction: LSTMs learn to extract relevant features from the raw sequence data.
Suitable for:
	•	Natural Language Processing tasks
	•	Time series analysis where long-term dependencies are crucial
	•	Tasks requiring understanding of context in both directions of a sequence

Model 2: CNN (Convolutional Neural Network)
This model uses Convolutional and Pooling layers, typically associated with image processing but adapted for sequence data.
Key Features:
	1.	Convolutional Layers: Extract local patterns or features from the input sequence.
	2.	MaxPooling Layers: Reduce dimensionality and capture the most important features.
	3.	Dropout Layers: Help prevent overfitting by randomly deactivating neurons during training.
	4.	Flatten Layer: Converts the 2D feature maps to a 1D vector for the dense layers.
	5.	Dense Layers: Make the final classification based on the extracted features.
Suitable for:
	•	Tasks where local patterns in the sequence are important
	•	Scenarios where the relative position of features matters more than long-term dependencies
	•	Often used in text classification, especially for shorter sequences

    Key Differences
	1.	Approach to Feature Extraction:
	•	Model 1 relies on LSTM to learn temporal dependencies.
	•	Model 2 uses convolutions to extract local patterns.
	2.	Handling of Sequence Length:
	•	Model 1 can handle variable-length sequences more naturally.
	•	Model 2 typically requires fixed-length input (defined by `window_s
ize`).
	3.	Complexity and Parameters:
	•	Model 1 likely has fewer parameters but may be more computationally intensive.
	•	Model 2 might be faster to train but could have more parameters depending on the configuration.
	4.	Context Understanding:
	•	Model 1 is better at capturing long-range dependencies.
	•	Model 2 excels at identifying local patterns and their relative positions.
	5.	Regularization:
	•	Model 2 explicitly uses Dropout for regularization.
	•	Model 1 relies on the inherent properties of LSTM for managing overfitting.
In summary, Model 1 is more suited for tasks requiring understanding of long-term dependencies and bidirectional context, while Model 2 is better for tasks where local patterns and their relative positions are more important. The choice between them would depend on the specific characteristics of your data and the nature of the problem you’re trying to solve.

The model is compiled with the following settings:
	•	Optimizer: Adam optimizer is used, which is an adaptive learning rate optimization algorithm.
	•	Loss Function: Categorical crossentropy, suitable for multiclass classification problems.
	•	Metric: Accuracy, which measures the overall correctness of the model’s predictions



# Model outcomes:

LSTM model without z-score normalisation (LSTM1) gives a prediction accuracy of 80% which is improved to 88% after signal normalisation (LSTM2). 

The basic CNN model picks up multiple peak signatures beyond the expected P, QRS & T signals. Will need further refinement. Doesn't seem to be faster or more efficient than LSTM in the current form. 

Check ways to further improve LSTM model. 
