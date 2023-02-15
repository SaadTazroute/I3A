#!/bin/bash

mkdir semeval_data
cd semeval_data

wget http://ixa2.si.ehu.es/stswiki/images/f/f3/Train_2015_10_22.utf-8.tar.gz
wget http://ixa2.si.ehu.es/stswiki/images/9/99/Train_students_answers_2015_10_27.utf-8.tar.gz
wget http://ixa2.si.ehu.es/stswiki/images/6/68/Test_goldstandard.tar.gz
wget http://ixa2.si.ehu.es/stswiki/images/5/5d/InterpretableSTS2015-en-train.zip
wget http://ixa2.si.ehu.es/stswiki/images/d/d1/InterpretableSTS2015-en-test.zip

tar -xzvf Train_2015_10_22.utf-8.tar.gz
tar -xzvf Train_students_answers_2015_10_27.utf-8.tar.gz
tar -xzvf Test_goldstandard.tar.gz
unzip InterpretableSTS2015-en-train.zip
unzip InterpretableSTS2015-en-test.zip

cd ..

cp semeval_data/test_goldStandard/STSint.testinput.answers-students.wa data/datasets/STSint.testinput.answers-students.wa
cp semeval_data/test_goldStandard/STSint.testinput.images.wa data/datasets/STSint.testinput.images.wa
cp semeval_data/test_goldStandard/STSint.testinput.headlines.wa data/datasets/STSint.testinput.headlines.wa
