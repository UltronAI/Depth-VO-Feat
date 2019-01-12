path=/home/share/kitti_odometry/dataset/sequences/09/image_2
files=$(ls $path)
for filename in $files
do 
    echo $filename >> filename.txt
done
#sed -i 's/$/ 6/g' $train_txt
