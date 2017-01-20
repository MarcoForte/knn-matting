# knn-matting
Python port of source code for KNN Matting, CVPR 2012 / TPAMI 2013 http://dingzeyu.li/projects/knn/

### Requirements
- python 3.5 (Though it should run on 2.7)
- scipy
- numpy
- matplotlib
- sklearn

### Running the demo
- '''python knn_matting.py'''
- mylambda (λ) is a constant controlling the users confidence in the constraints
- image size not larger than 640*480 reccomended for speed and memory reasons.

### Results
<img alt="Original image" src="https://github.com/MarcoForte/knn-matting/blob/master/donkey.png" width="250">
<img alt="Trimap image" src="https://github.com/MarcoForte/knn-matting/blob/master/donkeyTrimap.png" width="250">
<img alt="Result image" src="https://github.com/MarcoForte/knn-matting/blob/master/donkeyAlpha.png" width="250">

### More Information
The sparse matrix solve step may be greatly sped up by using the scikit-umfpack wrapper of UMFPACK sparse direct solver, https://github.com/scikit-umfpack/scikit-umfpack  
For more information see the orginal paper and it's website here http://dingzeyu.li/projects/knn/
The original matlab code is here https://github.com/dingzeyuli/knn-matting

### Disclaimer

The code is free for academic/research purpose. Use at your own risk and we are not responsible for any loss resulting from this code. Feel free to submit pull request for bug fixes.

### Contact 
[Marco Forte](https://marcoforte.github.io/) (fortem@tcd.ie)  

Original authors:  
[Qifeng Chen](http://web.stanford.edu/~cqf/) (cqf@stanford.edu) and [Dingzeyu Li](http://dingzeyu.li/) (dli@cs.columbia.edu)
