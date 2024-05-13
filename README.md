# Prereq:  

install python  
pip install numpy   
pip install opencv-python  
python3 -m pip install tensorflow[and-cuda]  
pip install keras  
pip install pillow  
pip install scipy  
pip install matplotlib  

view advanced system settings -> environment variables -> click variable called path -> edit -> add new line ith below location  
C:\Users\Adam\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts  

### Add untracked files
git add $(git ls-files -o --exclude-standard)