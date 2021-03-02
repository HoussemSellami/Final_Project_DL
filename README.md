# Final_Project_DL

# Files:
This project is composed of the following .py files: 

* train.py: This is the main training file. You need to run it to train and evaluate all four models, that were discussed in the report. You can choose in the main function, which models you want to train. This will automatically generate a csv file for each model architecture (see data/CiteSeer/processed), which contains the hyper-parameters description and the training, validation and test accuracy for that model with the defined hyper-parameters. This will also reproduce all the confusion matrices for the 4 different architectures, which include figure 4 from the final report for the JK-Transformer model.

* models.py: This file contains the four models, that will be used for training (GAT, GCN, JK-LSTM, and JK-Transformer)

* experiments.py : This file is used to easily choose the desired hyper-parameters searching space for each model individually. The default values in this file are the hyper-parameters that were chosen for the best validation performance. Don't change them if you want to reproduce exactly the same results from the report. But, feel free to change them, to see how this would affect the performances (just add more elements to the lists, example: 'dropout': [0.0, 0.1, 0.3].)

* utils.py: This file contains some utils used for evaluation of the networks.

* attention.py: Running this file (after running the train.py) would reproduce figure 1, figure 5 and figure 6 from the final report. It is mainly used for analyzing the attention weights.


#Dataset:
The dataset will be automatically downloaded when running train.py. In case of any problem, please download the data from the following link: 
https://drive.google.com/drive/folders/1kqC9E0bsTlKI4Ml8SxK4gR1UXpHq6qom?usp=sharing and place it into the main folder.
Please make sure that the path of the dataset is as following: (your main folder with all .py files)/data/CiteSeer 


#Packages: 
This projects uses many packages that need to be downloaded. 
I will share the version of some of my installed packages: 
troch                     1.7.1
torch-geometric           1.6.3                    
torch-scatter             2.0.5                  
torch-sparse              0.6.8                    
torchvision               0.4.1         
networkx                  2.2                    
You need to have these versions or more recent ones, but not older.
P.S: I tried to make this code platform-independent (works on Windows, Linux, ..). In case of any problem, please contact me on: houssem.sellami@hotmail.fr
