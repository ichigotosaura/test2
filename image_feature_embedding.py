"""

このノートブックは、画像から特徴量を抽出して埋め込みベクトルとして利用するためのツールです。
PyTorchとTensorFlowの両方のフレームワークをサポートしており、複数の事前学習済みモデルから選択できます。
特に画像分類タスクに適した特徴量抽出を行います。

- 画像ファイルのアップロード
- PyTorchまたはTensorFlowによる特徴量抽出
- 複数の事前学習済みモデルのサポート
- 特徴量の可視化（PCA）
- 特徴量の保存と読み込み
- 分類タスク向けの機能
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from google.colab import files
from IPython.display import display

def upload_images(by_class=False):
    """
    Google Colabでファイルをアップロードする関数
    
    Args:
        by_class (bool): クラスごとにファイルをアップロードするかどうか
    
    Returns:
        uploaded_files (dict): アップロードされたファイル名と画像オブジェクトの辞書
        labels (dict, optional): 画像ファイル名とラベルの辞書（by_class=Trueの場合のみ）
    """
    uploaded_files = {}
    labels = {}
    
    if by_class:
        print("クラスごとに画像をアップロードします。")
        continue_upload = True
        
        while continue_upload:
            class_name = input("クラス名を入力してください: ")
            print(f"クラス '{class_name}' の画像ファイルをアップロードしてください。")
            uploaded = files.upload()
            
            for filename in uploaded.keys():
                try:
                    img = Image.open(filename)
                    uploaded_files[filename] = img
                    labels[filename] = class_name
                    print(f"ファイル '{filename}' を読み込みました。サイズ: {img.size}, クラス: {class_name}")
                    display(img)
                except Exception as e:
                    print(f"ファイル '{filename}' の読み込みに失敗しました: {e}")
            
            continue_input = input("別のクラスの画像をアップロードしますか？ (y/n) [デフォルト: y]: ").lower()
            continue_upload = continue_input != 'n'
        
        print(f"合計 {len(uploaded_files)} 個の画像を {len(set(labels.values()))} クラスにアップロードしました。")
        return uploaded_files, labels
    else:
        print("画像ファイルをアップロードしてください。")
        uploaded = files.upload()
        
        for filename in uploaded.keys():
            try:
                img = Image.open(filename)
                uploaded_files[filename] = img
                print(f"ファイル '{filename}' を読み込みました。サイズ: {img.size}")
                display(img)
            except Exception as e:
                print(f"ファイル '{filename}' の読み込みに失敗しました: {e}")
        
        return uploaded_files

def extract_features_pytorch(images, model_name='resnet50', layer='avgpool'):
    """
    PyTorchの事前学習済みモデルを使用して画像特徴量を抽出する関数
    
    Args:
        images (dict): 画像ファイル名と画像オブジェクトの辞書
        model_name (str): 使用するモデル名（'resnet18', 'resnet50', 'vgg16', 'densenet121'など）
        layer (str): 特徴量を抽出するレイヤー名
    
    Returns:
        features (dict): 画像ファイル名と特徴量の辞書
    """
    available_models = {
        'resnet18': (torchvision.models.resnet18, torchvision.models.ResNet18_Weights.IMAGENET1K_V1),
        'resnet50': (torchvision.models.resnet50, torchvision.models.ResNet50_Weights.IMAGENET1K_V2),
        'vgg16': (torchvision.models.vgg16, torchvision.models.VGG16_Weights.IMAGENET1K_V1),
        'densenet121': (torchvision.models.densenet121, torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
    }
    
    if model_name not in available_models:
        raise ValueError(f"モデル '{model_name}' は利用できません。利用可能なモデル: {list(available_models.keys())}")
    
    model_func, weights = available_models[model_name]
    model = model_func(weights=weights)
    model.eval()
    
    if model_name.startswith('resnet'):
        if layer == 'avgpool':
            new_model = torch.nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"ResNetでは '{layer}' レイヤーはサポートされていません")
    elif model_name.startswith('vgg'):
        if layer == 'features':
            new_model = model.features
        else:
            raise ValueError(f"VGGでは '{layer}' レイヤーはサポートされていません")
    elif model_name.startswith('densenet'):
        if layer == 'features':
            new_model = model.features
        else:
            raise ValueError(f"DenseNetでは '{layer}' レイヤーはサポートされていません")
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    preprocess_grayscale = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),  # グレースケールをRGB形式に変換
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    features = {}
    for filename, img in images.items():
        if img.mode == 'L' or img.mode == '1':
            input_tensor = preprocess_grayscale(img)
        else:
            input_tensor = preprocess(img)
            
        input_batch = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            feature = new_model(input_batch)
            if len(feature.shape) == 4:  # [B, C, H, W] -> [B, C]
                feature = torch.mean(feature, dim=[2, 3])
            feature = feature.squeeze().numpy()
            features[filename] = feature
    
    return features

def extract_features_tensorflow(images, model_name='resnet50', include_top=False):
    """
    TensorFlowの事前学習済みモデルを使用して画像特徴量を抽出する関数
    
    Args:
        images (dict): 画像ファイル名と画像オブジェクトの辞書
        model_name (str): 使用するモデル名（'resnet50', 'vgg16', 'mobilenet'など）
        include_top (bool): 分類層を含めるかどうか
    
    Returns:
        features (dict): 画像ファイル名と特徴量の辞書
    """
    available_models = {
        'resnet50': tf.keras.applications.ResNet50,
        'vgg16': tf.keras.applications.VGG16,
        'mobilenet': tf.keras.applications.MobileNet,
        'inception_v3': tf.keras.applications.InceptionV3
    }
    
    if model_name not in available_models:
        raise ValueError(f"モデル '{model_name}' は利用できません。利用可能なモデル: {list(available_models.keys())}")
    
    base_model = available_models[model_name](weights='imagenet', include_top=include_top)
    model = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
    
    if model_name == 'resnet50':
        preprocess_input = tf.keras.applications.resnet50.preprocess_input
        target_size = (224, 224)
    elif model_name == 'vgg16':
        preprocess_input = tf.keras.applications.vgg16.preprocess_input
        target_size = (224, 224)
    elif model_name == 'mobilenet':
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input
        target_size = (224, 224)
    elif model_name == 'inception_v3':
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        target_size = (299, 299)
    
    features = {}
    for filename, img in images.items():
        if img.mode == 'L' or img.mode == '1':
            img = img.convert('RGB')
            
        img_array = img.resize(target_size)
        img_array = np.array(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        feature = model.predict(img_array)
        feature = feature.flatten()
        features[filename] = feature
    
    return features

def visualize_features(features, method='pca', n_components=2):
    """
    抽出された特徴量を可視化する関数
    
    Args:
        features (dict): 画像ファイル名と特徴量の辞書
        method (str): 次元削減手法（'pca'または'tsne'）
        n_components (int): 次元削減後の次元数
    """
    if len(features) < 2:
        print("可視化には少なくとも2つの画像が必要です。")
        return
    
    feature_matrix = np.array(list(features.values()))
    filenames = list(features.keys())
    
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        reduced_features = reducer.fit_transform(feature_matrix)
        explained_variance = reducer.explained_variance_ratio_
        print(f"説明された分散: {explained_variance}")
        variance_label = f"説明された分散: {sum(explained_variance):.2%}"
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
        reduced_features = reducer.fit_transform(feature_matrix)
        variance_label = "t-SNEによる次元削減"
    else:
        raise ValueError(f"次元削減手法 '{method}' はサポートされていません。")
    
    if n_components == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1])
        
        for i, filename in enumerate(filenames):
            plt.annotate(os.path.basename(filename), (reduced_features[i, 0], reduced_features[i, 1]))
        
        plt.title(f'画像特徴量の2次元表現 ({method.upper()})')
        plt.xlabel(f'次元1')
        plt.ylabel(f'次元2')
        plt.grid(True)
        plt.figtext(0.5, 0.01, variance_label, ha='center')
        plt.show()
    
    elif n_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2])
        
        for i, filename in enumerate(filenames):
            ax.text(reduced_features[i, 0], reduced_features[i, 1], reduced_features[i, 2], os.path.basename(filename))
        
        ax.set_title(f'画像特徴量の3次元表現 ({method.upper()})')
        ax.set_xlabel('次元1')
        ax.set_ylabel('次元2')
        ax.set_zlabel('次元3')
        plt.figtext(0.5, 0.01, variance_label, ha='center')
        plt.show()
    
    return reduced_features

def save_features(features, filename='image_features.npy'):
    """
    抽出した特徴量をファイルに保存する関数
    
    Args:
        features (dict): 画像ファイル名と特徴量の辞書
        filename (str): 保存するファイル名
    """
    np.save(filename, features)
    print(f"特徴量を '{filename}' に保存しました。")
    
    files.download(filename)

def load_features(filename='image_features.npy'):
    """
    保存された特徴量をファイルから読み込む関数
    
    Args:
        filename (str): 読み込むファイル名
    
    Returns:
        features (dict): 画像ファイル名と特徴量の辞書
    """
    uploaded = files.upload()
    
    if filename in uploaded:
        features = np.load(filename, allow_pickle=True).item()
        print(f"特徴量を '{filename}' から読み込みました。")
        return features
    else:
        print(f"ファイル '{filename}' をアップロードしてください。")
        return None

def prepare_data_for_classification(features, labels=None):
    """
    特徴量を分類タスク用に準備する関数
    
    Args:
        features (dict): 画像ファイル名と特徴量の辞書
        labels (dict, optional): 画像ファイル名とラベルの辞書。Noneの場合はファイル名からラベルを抽出
    
    Returns:
        X (numpy.ndarray): 特徴量の配列
        y (numpy.ndarray): ラベルの配列
        label_map (dict): ラベルとインデックスのマッピング
    """
    X = np.array(list(features.values()))
    filenames = list(features.keys())
    
    if labels is None:
        extracted_labels = [os.path.basename(filename).split('_')[0] for filename in filenames]
        unique_labels = sorted(set(extracted_labels))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        y = np.array([label_map[label] for label in extracted_labels])
    else:
        y = np.array([labels[filename] for filename in filenames])
        unique_labels = sorted(set(labels.values()))
        label_map = {label: i for i, label in enumerate(unique_labels)}
    
    print(f"データセット: {X.shape[0]}サンプル, {X.shape[1]}次元")
    print(f"クラス: {len(label_map)}個 - {list(label_map.keys())}")
    
    return X, y, label_map

def train_classifier(X, y, classifier_type='svm', test_size=0.2, random_state=42):
    """
    特徴量を使用して分類器を訓練する関数
    
    Args:
        X (numpy.ndarray): 特徴量の配列
        y (numpy.ndarray): ラベルの配列
        classifier_type (str): 分類器の種類（'svm'など）
        test_size (float): テストデータの割合
        random_state (int): 乱数シード
    
    Returns:
        classifier: 訓練された分類器
        X_train, X_test, y_train, y_test: 訓練データとテストデータ
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_samples_per_class = np.min(class_counts)
    
    if min_samples_per_class >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print("層別サンプリングを使用してデータを分割しました。")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=None
        )
        print("警告: 一部のクラスのサンプル数が少ないため、層別サンプリングを使用せずにデータを分割しました。")
    
    print(f"訓練データ: {X_train.shape[0]}サンプル, テストデータ: {X_test.shape[0]}サンプル")
    
    if classifier_type == 'svm':
        classifier = SVC(kernel='rbf', probability=True, random_state=random_state)
        classifier.fit(X_train, y_train)
    else:
        raise ValueError(f"分類器タイプ '{classifier_type}' はサポートされていません。")
    
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"分類精度: {accuracy:.4f}")
    
    report = classification_report(y_test, y_pred)
    print("分類レポート:")
    print(report)
    
    return classifier, X_train, X_test, y_train, y_test, scaler

def classify_new_images(classifier, scaler, images, extract_func, model_name='resnet50', label_map=None):
    """
    新しい画像を分類する関数
    
    Args:
        classifier: 訓練された分類器
        scaler: 特徴量の標準化に使用したスケーラー
        images (dict): 画像ファイル名と画像オブジェクトの辞書
        extract_func: 特徴量抽出関数
        model_name (str): 使用するモデル名
        label_map (dict): ラベルとインデックスのマッピング
    
    Returns:
        predictions (dict): 画像ファイル名と予測結果の辞書
    """
    features = extract_func(images, model_name=model_name)
    
    predictions = {}
    for filename, feature in features.items():
        feature_scaled = scaler.transform(feature.reshape(1, -1))
        
        pred_class = classifier.predict(feature_scaled)[0]
        pred_proba = classifier.predict_proba(feature_scaled)[0]
        
        if label_map is not None:
            inv_label_map = {v: k for k, v in label_map.items()}
            pred_class_name = inv_label_map[pred_class]
            class_probas = {inv_label_map[i]: prob for i, prob in enumerate(pred_proba)}
        else:
            pred_class_name = str(pred_class)
            class_probas = {str(i): prob for i, prob in enumerate(pred_proba)}
        
        predictions[filename] = {
            'class': pred_class_name,
            'probabilities': class_probas
        }
        
        print(f"画像 '{filename}' の予測クラス: {pred_class_name}")
        print(f"クラス確率: {sorted(class_probas.items(), key=lambda x: x[1], reverse=True)}")
    
    return predictions

def main():
    """
    画像特徴量抽出と分類の実行フロー
    """
    print("画像の特徴量抽出・分類ツール")
    print("=" * 50)
    
    print("\n## 画像のアップロード")
    upload_by_class = input("クラスごとに画像をアップロードしますか？ (y/n) [デフォルト: y]: ").lower() != 'n'
    
    if upload_by_class:
        images, labels = upload_images(by_class=True)
    else:
        images = upload_images(by_class=False)
        labels = None
    
    if not images:
        print("画像がアップロードされていません。処理を終了します。")
        return
    
    framework = input("\n## 使用するフレームワークを選択してください (pytorch/tensorflow) [デフォルト: pytorch]: ").lower()
    if not framework:
        framework = 'pytorch'
    
    print(f"\n## {framework}による特徴量抽出")
    if framework == 'pytorch':
        model_name = input("使用するモデルを選択してください (resnet18/resnet50/vgg16/densenet121) [デフォルト: resnet50]: ")
        if not model_name:
            model_name = 'resnet50'
        
        features = extract_features_pytorch(images, model_name=model_name)
        extract_func = extract_features_pytorch
    
    elif framework == 'tensorflow':
        model_name = input("使用するモデルを選択してください (resnet50/vgg16/mobilenet/inception_v3) [デフォルト: resnet50]: ")
        if not model_name:
            model_name = 'resnet50'
        
        features = extract_features_tensorflow(images, model_name=model_name)
        extract_func = extract_features_tensorflow
    
    else:
        print(f"フレームワーク '{framework}' は利用できません。pytorchまたはtensorflowを選択してください。")
        return
    
    print("\n## 抽出された特徴量")
    for filename, feature in features.items():
        print(f"ファイル '{filename}' の特徴量: 形状 {feature.shape}, ノルム {np.linalg.norm(feature):.4f}")
    
    if len(images) >= 2:
        print("\n## 特徴量の可視化")
        visualize = input("特徴量を可視化しますか？ (y/n) [デフォルト: y]: ").lower()
        if visualize != 'n':
            method = input("次元削減手法を選択してください (pca/tsne) [デフォルト: pca]: ").lower()
            if not method or method not in ['pca', 'tsne']:
                method = 'pca'
            
            n_components = int(input("可視化する次元数を選択してください (2/3) [デフォルト: 2]: ") or 2)
            visualize_features(features, method=method, n_components=n_components)
    
    print("\n## 分類タスク")
    classify = input("特徴量を使用して分類器を訓練しますか？ (y/n) [デフォルト: y]: ").lower()
    if classify != 'n':
        if upload_by_class and labels:
            X, y, label_map = prepare_data_for_classification(features, labels)
        else:
            X, y, label_map = prepare_data_for_classification(features)
        
        classifier, X_train, X_test, y_train, y_test, scaler = train_classifier(X, y)
        
        classify_new = input("\n新しい画像をアップロードして分類しますか？ (y/n) [デフォルト: n]: ").lower()
        if classify_new == 'y':
            print("\n## 新しい画像のアップロード")
            new_images = upload_images(by_class=False)
            
            if new_images:
                print("\n## 新しい画像の分類")
                predictions = classify_new_images(
                    classifier, scaler, new_images, extract_func, 
                    model_name=model_name, label_map=label_map
                )
    
    print("\n## 特徴量の保存")
    save = input("特徴量を保存しますか？ (y/n) [デフォルト: y]: ").lower()
    if save != 'n':
        filename = input("保存するファイル名を入力してください [デフォルト: image_features.npy]: ")
        if not filename:
            filename = 'image_features.npy'
        save_features(features, filename=filename)

if __name__ == "__main__":
    main()
