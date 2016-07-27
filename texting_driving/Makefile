
data:
    cd code && 
    mkdir ../data/train/images && 
    ffmpeg -i texting_driving.MOV -r 20 ../data/train/images/image_sequence%d.jpeg && 
    python images_to_matrix.py && 
    cd .. 

run_model:
    cd code && 
    python 3d_cnn.py && 
    cd .. 


