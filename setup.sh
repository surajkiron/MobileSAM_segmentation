mkdir -p resources/data  
mkdir resources/experiments
mkdir resources/hdf5     # add hdf5 files
mkdir resources/weights 

wget -P resources/data http://images.cocodataset.org/zips/train2017.zip
unzip resources/data/train2017.zip -d resources/data

wget -P resources/data http://images.cocodataset.org/zips/val2017.zip
unzip resources/data/val2017.zip -d resources/data

wget -P resources/data http://images.cocodataset.org/zips/test2017.zip
unzip resources/data/test2017.zip -d resources/data

wget -P resources/DATA http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip resources/DATA/annotations_trainval2017.zip -d resources/data

