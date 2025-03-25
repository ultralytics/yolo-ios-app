# テストリソースディレクトリ

このディレクトリには、YOLOのテストに必要なモデルファイルを配置します。

## 必要なモデルファイル

テストを実行するには、以下のCoreMLモデルファイルをこのディレクトリに配置してください：

- `yolo11n.mlpackage` - 検出モデル
- `yolo11n-seg.mlpackage` - セグメンテーションモデル
- `yolo11n-cls.mlpackage` - 分類モデル
- `yolo11n-pose.mlpackage` - ポーズ推定モデル
- `yolo11n-obb.mlpackage` - 向き付き境界ボックスモデル

## モデルファイルの取得方法

詳細な取得方法については、`Tests/YOLOTests/README.md`を参照してください。