path=/home/gaof/workspace/Depth-VO-Feat/00/image_2
files=$(ls $path)
for filename in $files
do 
    echo $filename >> filename.txt
done
#sed -i 's/$/ 6/g' $train_txt
