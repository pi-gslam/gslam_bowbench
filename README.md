# GSLAM Vocabulary Benchmark

This project performs comparison between DBoW2, DBoW3, FBoW and Vocabulary of GSLAM.
The load, save, transform and memory usage of them are compared.

# Usage
## 1. Download the source code

```
git clone --recursive https://github.com/pi-lab/gslam_bowbench
``` 

## 2. Compile

```
cd gslam_bowbench
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
gslam_bowbench --help 
```

Run with:

```
gslam_bowbench -images images.txt -k=10 -level=3 -mem -feature=ORB
```

## 4. Sample result

Here is a sample result:
```
build> ./gslam_bowbench -images images.txt 
main.cpp:328 Memory analysis started.
main.cpp:353 ../DBow3/utils/images/image0.png found 843 keypoints.
main.cpp:353 ../DBow3/utils/images/image1.png found 908 keypoints.
main.cpp:353 ../DBow3/utils/images/image2.png found 862 keypoints.
main.cpp:353 ../DBow3/utils/images/image3.png found 896 keypoints.
----------------GSLAM------------------------
main.cpp:26 GSLAM: Creating vocabulary from image features.

main.cpp:34 Vocabulary: k = 10, L = 3, Weighting = tf-idf, Scoring = L1-norm, Number of words = 1000

main.cpp:51 GSLAM used memory 44536 bytes with 5 pieces.
main.cpp:55 Memory leak:0 bytes with 0 pieces.
----------------DBoW2------------------------
main.cpp:89 DBoW2: Creating vocabulary from image features.
main.cpp:97 Created Vocabulary: k = 10, L = 3, Weighting = tf-idf, Scoring = L1-norm, Number of words = 980
main.cpp:116 DBoW2: Saving vocabulary Vocabulary: k = 10, L = 3, Weighting = tf-idf, Scoring = L1-norm, Number of words = 980
main.cpp:130 DBoW2 used memory 245932 bytes with 1204 pieces.
main.cpp:133 Memory leak:0 bytes with 0 pieces.
----------------DBoW3------------------------
main.cpp:204 DBoW3: Creating vocabulary from image features.
main.cpp:209 Created Vocabulary: k = 10, L = 3, Weighting = tf-idf, Scoring = L1-norm, Number of words = 991
main.cpp:228 DBoW3: Saving vocabulary Vocabulary: k = 10, L = 3, Weighting = tf-idf, Scoring = L1-norm, Number of words = 991
main.cpp:242 DBoW3 used memory 248628 bytes with 1217 pieces.
main.cpp:245 Memory leak:0 bytes with 0 pieces.
----------------FBoW------------------------
main.cpp:255 FBoW: Creating vocabulary from image features.
main.cpp:263 Created 111
main.cpp:283 FBoW: Saving vocabulary ...
main.cpp:297 FBoW used memory 45464 bytes with 3 pieces.
main.cpp:300 Memory leak:0 bytes with 0 pieces.

------------------------------------  Timer report ------------------------------------
           FUNCTION                       #CALLS  MIN.T  MEAN.T  MAX.T  TOTAL 
---------------------------------------------------------------------------------------
DBoW2::load                                  1    4.9ms   4.9ms   4.9ms   4.9ms
DBoW2::save                                  1    4.2ms   4.2ms   4.2ms   4.2ms
DBoW2::train                                 1  114.0ms 114.0ms 114.0ms 114.0ms
DBoW2::transDes                           3509  617.0ns 761.0ns   3.3us   2.7ms
DBoW2::transImage                            4  994.1us   1.1ms   1.2ms   4.4ms
DBoW3::load                                  1  870.9us 870.9us 870.9us 870.9us
DBoW3::save                                  1  300.4us 300.4us 300.4us 300.4us
DBoW3::train                                 1  152.1ms 152.1ms 152.1ms 152.1ms
DBoW3::transDes                           3509  329.0ns 463.8ns   6.9us   1.6ms
DBoW3::transImage                            4  743.6us 857.5us 985.2us   3.4ms
FBoW::load                                   1   24.3us  24.3us  24.3us  24.3us
FBoW::save                                   1  217.2us 217.2us 217.2us 217.2us
FBoW::train                                  1  116.8ms 116.8ms 116.8ms 116.8ms
FBoW::transDes                            3509  530.0ns 558.4ns   1.1us   2.0ms
FBoW::transImage                             4  300.3us 352.4us 377.6us   1.4ms
GSLAM::load                                  1   38.0us  38.0us  38.0us  38.0us
GSLAM::save                                  1  179.3us 179.3us 179.3us 179.3us
GSLAM::train                                 1   72.7ms  72.7ms  72.7ms  72.7ms
GSLAM::transDes                           3509  239.0ns 301.0ns 502.0ns   1.1ms
GSLAM::transImage                            4  273.2us 291.1us 302.1us   1.2ms
--------------------------------- End of Timer report ---------------------------------
```
