# GestureRecoqnition

To use program, follow this set of instructions:

PRE-REQUISITES: must have python 3, pip and anaconda package manager installed

1. Clone this repo and cd into it

2. Open command line and create a tensorflow anaconda environment:
```$ conda create -n tfges tensorflow```

3. Activate anaconda environment
```$ conda activate tfges```

4. Install required packages ***dont have requirements done yet*** or manually pip install everything
```$ pip install -r requirements.txt ```

    ***Note: This hasn't been tested yet. You may have to manually install all packages***
    If needed, use the Anaconda package manager GUI to install packages. 
    Also, installing tensorflow may require:
    ```$ conda install -c conda-forge tensorflow```
    You may then need to run the following for each of the packages present in the requirements.txt file if you receive messages that a package can't be found when trying to run the program:
    ```$pip install <package>={version number from requirements.txt}```

    Additional configuration might be necessary and documented here. Likely candidates are openCV, which requires a 3rd party conda package (but required setup is not confirmed)
    If opencv is a problem, run
    ```$ conda update conda -c conda-canary``` then
    ```$ pip install opencv-python```
    
5. ```$ python3 <file_to_run>.py```
     a. To train and test the neural network: ```$ python3 trainSqueezenet.py```
     b. To run webcame demo: ```$ python3 demo.py```


When done, deactivate the virtual environment:
```$ conda deactivate```
conda deactivate 

