# Repository for Visio Project

## Dependency

* Python >= 3.7.9
* numpy >= 1.19.5
* opencv-python >= 4.5.1.48
* depthai >= 2.0.0.0

## Usage

- Run on recorded video  
`
python main.py (path to video)
`  

- Run and record inference result  
`
python main.py (path to video) -r (path to output video)
`  

## To Do
- Make bluetooth work
- Try to make full/bigger frame tracking work as it may improve tracking performace
- Record tracking data as well and possibly train a lgb model to predict time to close-by?