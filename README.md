# GSLAM Vocabulary

This project performs comparison between DBoW2, DBoW3, FBoW and Vocabulary of GSLAM.
The load, save, transform and memory usage of them are compared.

# Usage
## 1. Download the source code

```
git clone --recursive https://github.com/zdzhaoyong/GSLAMVocabulary
``` 

## 2. Compile

```
cd GSLAMVocabulary
mkdir build
cd build
cmake ..
make
```

## 3. Run
The program need a prepared file with image path listed. You can generate a sample list file with bash:

```
for img in $(ls ../DBow3/utils/images/*);do echo $img;done > images.txt
```

Show usage with:

```
./compare --help 
```

Run with:

```
./compare -images images.txt -k=10 -level=3 -mem -feature=ORB
```

## 4. Sample result

Here is a sample result:
```

```
