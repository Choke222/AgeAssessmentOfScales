#!/bin/bash
file1="uroko20181a4"
ex=".jpg"
file1_1=$file1$ex
echo $file1_1
dir1="../dataset/image/"
echo python3 data_proc.py $dir1$file1_1
python3 data_proc.py $dir1$file1_1
# python3 data_proc.py ../dataset/image/uroko20181a4.jpg

dir2=../dataset/patching/
dir3=../dataset/out_restingzonedetec
python3 predict.py $dir2$file1 $dir3
echo python3 predict.py $dir2$file1 $dir3
# python3 predict.py ../dataset/patching/uroko201822a7 ../dataset/out_restingzonedetec

dir4=../dataset/out_restingzonedetec/
dir5=../dataset/restzone/
python3 image_connect.py $dir4$file1 $dir5$file1
echo python3 image_connect.py $dir4$file1 $dir5$file1
# python3 image_connect.py ../dataset/out_restingzonedetec/uroko201822a7 ../dataset/restzone/uroko201822a7

python3 main.py -r$file1 -d $file1
echo python3 main.py -r $file1 -d $file1
# python3 main.py -r uroko201822a7 -d uroko201822a7