### Prereq:  

install python  
pip install numpy   
pip install opencv-python   
pip install tensorflow==2.13.1 --user  
pip install keras==2.13.1 --user  
pip install matplotlib  
pip install pandas  
pip install tabulate  
pip install openpyxl  
pip install seaborn  
pip install joblib

view advanced system settings -> environment variables -> click variable called path -> edit -> add new line ith below location  
C:\Users\Adam\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts  

# Add untracked files
git add $(git ls-files -o --exclude-standard)

# Delete 1 previous commit without deleting local changes
git reset HEAD^

git push origin main  
git pull origin main  
git commit -a -m ""   
git stash       
git status  
   
git remote remove origin  

git init  
git add .  
git remote show origin  
git remote add origin https://github.com/AdamKuraczynski/DinoImageRecog.git  

### Testing variables:
- conv layers 1/2/3/5
- pool layers 1/2/3/5
- epochs 5/10/25/50
- different optimizers Adam/SGD/RMSpop/Adadelta
- batch size 16/32/64/128