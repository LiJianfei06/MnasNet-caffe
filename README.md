# MnasNet-caffe
A caffe implementation of Mnasnet: MnasNet: Platform-Aware Neural Architecture Search for Mobile.






train:

./train_net.sh






test:

python eval_image.py --proto deploy_MnasNet.prototxt --model ./model_save/MnasNet_model_cat_dog_iter_64000.caffemodel  --image ./cat.jpg






valid:

python verify.py
