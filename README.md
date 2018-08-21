# Incisor Segmentation

Implementation and evaluation of a model-based segmentation approach, capable
of segmenting the upper and lower incisors in panoramic radiographs. 

![example](https://raw.githubusercontent.com/probberechts/IncisorSegmentation/master/Plot/Results/02.png)

### Usage
```shell 
$ python Code/main.py --help
usage: main.py [-h] [-p] [-d] [-f {auto,manual}] [-k K] [-m M] [-o OUT]
               radiographs {1,2,3,4,5,6,7,8}
               {1,2,3,4,5,6,7,8,9,10,11,12,13,14}

A program to segment the upper and lower incisors in panoramic radiographs

positional arguments:
  radiographs           The folder with the radiographs
  {1,2,3,4,5,6,7,8}     The index of the incisor to fit [1-8]
  {1,2,3,4,5,6,7,8,9,10,11,12,13,14}
                        The index of the radiograph to fit on [1-14]

optional arguments:
  -h, --help            show this help message and exit
  -p, --preprocess      Preprocess the radiographs with an adaptive median
                        filter
  -d, --database        Build an appearance model for the autoinit option.
  -f {auto,manual}, --method {auto,manual}
                        The fitting method that should be used
  -k K, --k K           Number of pixels either side of point to represent in
                        grey model
  -m M, --m M           Number of sample points either side of current point
                        for search
  -o OUT, --out OUT     The folder to store the results
```
