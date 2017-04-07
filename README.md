## ファインチューニングしたResNet101の実装
###実装の流れ
* データの前処理
* 学習済みモデルの入手
* 学習
* 分類

### データの前処理
（フォルダ構成が以下のようになっていない場合）ラベルリストと画像とラベルの書いたファイルからフォルダ構成を変更する。

```
mkdir assorted
python assort_files_to_label_dir.py -i data_cookpad/clf_train_images_labeled_1/ -o assorted -l data_cookpad/clf_category_master.tsv -p data_cookpad/clf_train_master.tsv
```
フォルダ構成
```
--data_set
 |--category1
 | |-train0.jpg
 |    ...
 |--category2
 | |-train100.jpg
      ...
```


データを学習用と検証用に分割する。
```
python separate_train_val.py --root assorted/ --output_dir train_val_img --val_freq 10
```

整形されたデータで画像を256x256にリサイズし、ラベルリストを作成する。
```
python resize.py -i train_label.txt -o resized_train_imgs --rename 1 --label 1 --out_imglabel ./trainlabel_pairs.txt 
```

```
python resize.py -i val_label.txt -o resized_val_imgs --rename 1 --label 1 --out_imglabel ./vallabel_pairs.txt
```

平均画像を作成する。
```
python compute_mean.py trainlabel_pairs.txt --root . --output mean.npy
```


### 学習済みモデルの入手
以下のコマンドでresnetの学習済みモデルを入手したら、``pretrained_models``配下に入れる。
```
mkdir pretrained_models
wget https://www.dropbox.com/s/yqasroj1poru24u/ResNet101.model
mv ResNet101.model pretrained_models/
```

### 学習

```
python train_caltech101.py trainlabel_pairs.txt vallabel_pairs.txt --arch resnet_c --epoch 150 --gpu 0 --initmodel pretrained_models/ResNet101.model --loaderjob 4 --mean mean.npy --out result --output_model cookpad_resnet.h5 --output_optimizer optimizer_cookpad_resnet.h5
```

### 分類
ラベルのリストを作成する。
```
python make_labels.py -i 101_ObjectCategories/ -o labels_caltech101.txt
```

リサイズしてそれをテスト用のフォルダに入れる。
```
python resize.py -i test_datasets/ -o resized_tests/
```

画像を分類してcsvファイルに出力する。
```
python classify.py --gpu 0 --arch resnet_c --initmodel result/cookpad_resnet.h5 --img_files resized_tests/*.jpg --mean mean.npy
```


### 参考
https://github.com/ta-oyama/chainer_tutorial
https://github.com/yasunorikudo/chainer-ResNet
