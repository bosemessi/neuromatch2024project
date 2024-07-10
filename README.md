# neuromatch2024project

### Members - Jian Jin, Bobi, Jingyi, Nasos, Aikendrajit, Soumyajit

### Dataset - AllenSDK

### If you are using conda, here are the steps to create an environment and download and install packages : 
    - clone this repo on your laptop
    - on anaconda powershell in windows, or using mac zsh/bash terminal, change directories to the repo folder
    - conda create -n "allenSDK" python=3.10.0 ipython
    - conda activate allenSDK
    - conda install anaconda::pip 
    - pip install brain_observatory_utilities --upgrade
    - pip install ipykernel jupyter
    - python -m ipykernel install --user --name=allenSDK
    - pip install scikit-learn
    - pip install pyarrow

  This should be enough to start with....we will keep installing stuff as required. Everytime you work on the project, remember to conda activate allenSDK first. When you stop working, you can simply type conda deactivate to get out of the environment, and then quit the terminal. Let's enjoy wokring on the data !!
