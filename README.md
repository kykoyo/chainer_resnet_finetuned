# chainer_tutorial

## Caffeの学習済みモデルを読み込んで画像認識

1. cloneしたのちフォルダに移動  
    ```
    $ git clone https://github.com/ta-oyama/chainer_tutorial.git  
    $ cd chainer_tutorial
    ```

2. データセットをダウンロードし解凍  
    ```
    $ wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz  
    $ tar xzvf 101_ObjectCategories.tar.gz
    ```

3. ILSVRC2012のラベルリストlabels.txtを作成  
    ```
    $ mkdir ilsvrc12; cd ilsvrc12  
    $ wget https://github.com/BVLC/caffe/raw/master/data/ilsvrc12/get_ilsvrc_aux.sh  
    $ chmod +x get_ilsvrc_aux.sh  
    $ ./get_ilsvrc_aux.sh  
    $ cat synset_words.txt |awk '{print $2;}' > ../labels_ilsvrc.txt  
    $ cd ..; rm -r ilsvrc12  
    ```

4. ILSVRC2012データセットでalexnetを用いた学習済みのcaffeのモデルをダウンロード(時間がかかる）
    ```  
    $ python download_model.py alexnet  
    $ mkdir pretrained_models; mv bvlc_alexnet.caffemodel pretrained_models/  
    ```

5. ILSVRC2012データセットの平均画像(ilsvrc_2012_mean.npy)をダウンロード  
`$ python download_mean_file.py`  

6. 画像認識のテストに用いる画像のフォルダを作成し、リサイズした画像をその中に入れる  
`$ python resize.py -i 101_ObjectCategories/airplanes/*.jpg -o resized/airplanes/`  

7. 適当な画像を分類  
`$ python classify.py --gpu 0 --arch alex --initmodel pretrained_models/bvlc_alexnet.caffemodel --img_files resized/airplanes/image_0001.jpg --label_file labels_ilsvrc.txt --mean ilsvrc_2012_mean.npy`  

これ以降他の画像を分類テストしてみたいときは、リサイズした画像がなければ6を行い、リサイズ済みの画像に対しては7のみを実行すればよい

##ファインチューニングによる再学習  

1. データセットをtrain用とvalidation用に分ける(それぞれtrain_val_image/train/*, train_val_image/val/*の下)。さらにtrain, valそれぞれの画像のパスとラベルがペアになったファイルtrain_labels.txt, val_label.txtを作成(--rootの引数にとったディレクトリのサブディレクトリをラベルとすることを想定しており、大文字小文字関係なくアルファベット順に0から順番にラベルを振っている。なのでこれで学習したモデルで分類を行いたい場合は読み込むラベルを大文字小文字関係なくアルファベット順に並べておく必要がある。）  
`$ python separate_train_val.py --root 101_ObjectCategories/ --output_dir train_val_img --val_freq 10`

2. 平均画像の作成、および訓練画像として使用するために、リサイズした訓練画像データフォルダを作成。また、そのデータフォルダにて新たにつけた画像の名前とラベルのリストが書かれたtrainlabel_pairs.txtを作成。  
`python resize.py -i train_label.txt -o resized_train_imgs --rename 1 --label 1`

3. 平均画像(mean.npy)を作成  
`python compute_mean.py trainlabel_pairs.txt --root . --output mean.npy`

4. caffeの学習済みモデルをalexnet.pkl,alexnet.h5として保存しておく。(訓練する際、全く同じネットワークの構成で出力数も同じ場合はhdf5として保存してtrain_caltech101.pyの中でchainer.serializers.load_npz(args.initmodel, model)を使えばよい。それ以外の場合はpickleで保存し、train_caltech101.pyの中でinitmodel = pickle.load(open(args.initmodel)); util.copy_model(initmodel, model)を使って共通部分のパラメータのみコピーする )  
`$ python caffe_to_chainermodel.py`

5. 学習を行う  
`python train_caltech101.py trainlabel_pairs.txt val_label.txt --arch alex --epoch 150 --gpu 0 --initmodel pretrained_models/alexnet.pkl --loaderjob 4 --mean mean.npy --out result --output_model caltech101_alexnet.h5 --output_optimizer optimizer_caltech101_alexnet.h5`

## ファインチューニングで学習したモデルを用いて画像認識  

1. ラベルリストを作成  
`python make_labels.py -i 101_ObjectCategories/ -o labels_caltech101.txt`

2. 画像認識のテストに用いる画像のフォルダを作成し、リサイズした画像をその中に入れる  
`$ python resize.py -i 101_ObjectCategories/airplanes/*.jpg -o resized/airplanes/`

3. 分類  
`$ python classify.py --gpu 0 --arch alex --label_file labels_caltech101.txt --initmodel result/caltech101_alexnet.h5 --img_files resized/airplanes/image_0001.jpg --mean mean.npy`