# YOLOテスト実行ガイド

このディレクトリには、YOLOフレームワークの包括的なテストが含まれています。テストを実行するには、必要なモデルファイルをダウンロードして配置する必要があります。

## テスト前の準備

### 1. テストリソースディレクトリを確認

以下のディレクトリが存在することを確認してください：

```
Tests/YOLOTests/Resources/
```

存在しない場合は作成してください：

```bash
mkdir -p Tests/YOLOTests/Resources/
```

### 2. 必要なモデルファイルを取得

テストに必要な以下のCoreMLモデルファイルを準備してください：

- `yolo11n.mlpackage` - 検出モデル
- `yolo11n-seg.mlpackage` - セグメンテーションモデル
- `yolo11n-cls.mlpackage` - 分類モデル
- `yolo11n-pose.mlpackage` - ポーズ推定モデル
- `yolo11n-obb.mlpackage` - 向き付き境界ボックスモデル

### 3. モデルファイルの取得方法

#### 方法1: 公式サイトからダウンロード

1. [Ultralytics GitHub](https://github.com/ultralytics/ultralytics) からモデルをダウンロード
2. Python環境で以下のコードを実行して、CoreML形式に変換：

```python
from ultralytics import YOLO

# 検出モデル
model = YOLO("yolo11n.pt")
model.export(format="coreml")

# セグメンテーションモデル
model = YOLO("yolo11n-seg.pt")
model.export(format="coreml")

# 分類モデル
model = YOLO("yolo11n-cls.pt")
model.export(format="coreml")

# ポーズ推定モデル
model = YOLO("yolo11n-pose.pt")
model.export(format="coreml")

# 向き付き境界ボックスモデル
model = YOLO("yolo11n-obb.pt")
model.export(format="coreml")
```

#### 方法2: Ultralyticsのサンプルモデルを使用

Ultralyticsの[モデルハブ](https://docs.ultralytics.com/models/)からモデルをダウンロードし、変換することもできます。

### 4. モデルファイルの配置

変換した`.mlpackage`ファイルを`Tests/YOLOTests/Resources/`ディレクトリに配置してください。

## テストの実行

テストの準備ができたら、SwiftPMを使用してテストを実行できます：

```bash
swift test
```

または、Xcodeでプロジェクトを開いてテストを実行します：

1. Package.swiftを開く
2. Product > Test (⌘U)を選択

## トラブルシューティング

### モデルファイルが見つからない場合

テストエラーで「Test model file not found」というメッセージが表示される場合：

1. モデルファイルが正しいパスに配置されているか確認
2. モデルファイル名と拡張子が正確か確認（例: `yolo11n.mlpackage`）
3. Package.swiftのリソース設定が正しいか確認

### 他の問題

テスト実行中に問題が発生した場合は、以下を確認してください：

1. Swift Package Managerのバージョンが互換性があるか
2. 必要なiOSバージョン（iOS 16.0以上）をサポートしているか
3. CoreMLおよびVisionフレームワークが利用可能か