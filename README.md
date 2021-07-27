# Repository for Visio Project

## Dependency

* Python >= 3.7.9
* numpy >= 1.19.5
* opencv-python >= 4.5.1.48
* depthai >= 2.0.0.0

## Usage

- Run on recorded video  
`
python main.py -v (path to video)
`  
or  
`
python main.py --video (path to video)
`  

- Run and record inference result  
`
python main.py -v (path to video) -r
`  
or  
`
python main.py -v (path to video) --record
`  

- Run on camera  
`
python main.py
`
