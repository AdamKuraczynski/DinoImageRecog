### Prereq:  

install python 3.11.9  
pip install tensorflow==2.13.1 --user  
pip install keras==2.13.1 --user  
pip install opencv-python, matplotlib, pandas, tabulate, openpyxl, seaborn, joblib  

### Add PATH to environmental variables ERROR
view advanced system settings -> environment variables -> click variable called "Path" -> edit -> add new line with your python scripts, example below:
C:\Users\Adam\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts  

### Testing variables:
- conv layers 1/2/3/5
- pool layers 1/2/3/5
- epochs 5/10/25/50
- different optimizers Adam/SGD/RMSpop/Adadelta
- batch size 16/32/64/128
