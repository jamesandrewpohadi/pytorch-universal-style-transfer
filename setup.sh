mkdir -p data/content weights ckpt

wget https://transformer-cds.s3-ap-southeast-1.amazonaws.com/vgg_normalised.pth
wget https://transformer-cds.s3-ap-southeast-1.amazonaws.com/decoder.pth

wget http://images.cocodataset.org/zips/val2014.zip
unzip -qq val2014.zip
mv val2014 data/content/

wget https://transformer-cds.s3-ap-southeast-1.amazonaws.com/kaggle.json
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/

kaggle competitions download -c painter-by-numbers -f train_2.zip
unzip -qq train_2.zip
mv train_2 data/style

pip3 install -r requirements.txt