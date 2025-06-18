# Ultralytics YOLO iOS App UI移行戦略

## 概要
現在のUIKit/Storyboardベースの実装から、新しいデザイン仕様（Ultralytics_YOLO_iOS_New_UI_Spec.md）への段階的移行戦略を定義します。

## 現状分析

### 現在のアーキテクチャ
- **フレームワーク**: UIKit + Storyboard
- **主要ファイル**: ViewController.swift, Main.storyboard
- **カメラ実装**: YOLOView（フレームワークから提供）
- **デザイン**: 黒背景に白テキストのシンプルなUI

### 新デザインの主要変更点
1. **レイアウト構造**: 4層構造（Status Bar、Camera Preview、Task Tabs、Shutter Bar）
2. **カラーシステム**: ライムグリーン（#CFFF1A）をアクセントカラーに採用
3. **新機能**: 
   - パラメータ調整UI（信頼度、IoU、線の太さ）
   - デジタルズーム切り替え
   - カメラシャッターバー
   - モデルセレクタードロップダウン

## 移行戦略

### フェーズ1: 基盤整備（1-2週間）

#### 1.1 デザインシステムの構築
```swift
// Colors.swift - 新しいカラーパレット
extension UIColor {
    static let ultralyticsLime = UIColor(hex: "#CFFF1A")
    static let ultralyticsBrown = UIColor(hex: "#6A5545")
    static let ultralyticsTextSubtle = UIColor(hex: "#7D7D7D")
}

// Typography.swift - フォントシステム
struct Typography {
    static let statusBar = UIFont(name: "SFProRounded-Bold", size: 10)
    static let tabLabel = UIFont(name: "SFProRounded-Semibold", size: 11)
}
```

#### 1.2 新UIコンポーネントの作成
- `StatusMetricBar`: 上部のステータスバー
- `TaskTabStrip`: 下部のタスクタブ
- `ShutterBar`: カメラコントロールバー
- `ParameterEditView`: パラメータ調整UI

### フェーズ2: UIKit実装（2-3週間）

#### 2.1 ViewControllerのリファクタリング
```swift
class NewViewController: UIViewController {
    // 新しいUI構造
    private let statusBar = StatusMetricBar()
    private let cameraPreview = CameraPreviewContainer()
    private let taskTabStrip = TaskTabStrip()
    private let shutterBar = ShutterBar()
    private let parameterEditor = ParameterEditView()
}
```

#### 2.2 段階的な置き換え
1. **ステップ1**: 新しいViewControllerを作成（既存と並行）
2. **ステップ2**: 機能を段階的に移植
3. **ステップ3**: A/Bテストの実装（設定で切り替え可能）
4. **ステップ4**: 完全移行

### フェーズ3: 新機能の実装（1-2週間）

#### 3.1 パラメータ調整UI
- トースト通知の実装
- リアルタイムスライダー
- UserDefaultsへの保存

#### 3.2 カメラ機能の拡張
- デジタルズーム（1.0x/1.8x切り替え）
- 写真/動画撮影機能
- サムネイル表示

#### 3.3 アニメーション
- 250msのアクションシート
- 150msのトースト表示
- 80msのフラッシュエフェクト

### フェーズ4: 最適化と仕上げ（1週間）

#### 4.1 パフォーマンス最適化
- 60fps描画の確保
- メモリ使用量の最適化
- バッテリー消費の改善

#### 4.2 デバイス対応
- iPhone SE〜iPhone 15 Pro Maxの画面サイズ対応
- iPadのレイアウト調整
- ダークモード対応の確認

## 実装の優先順位

### 高優先度
1. **カラーシステムの導入**: 全体の見た目に大きく影響
2. **レイアウト構造の変更**: 4層構造への移行
3. **モデルセレクター**: ユーザビリティの向上

### 中優先度
1. **パラメータ調整UI**: 新機能だが既存機能には影響しない
2. **カメラシャッターバー**: 録画機能の改善
3. **アニメーション**: UXの向上

### 低優先度
1. **隠しInfoページ**: 補助的な機能
2. **細かいアニメーション調整**: 最終的な磨き込み

## リスクと軽減策

### リスク1: 既存ユーザーの混乱
- **軽減策**: 
  - 段階的なロールアウト
  - 設定での新/旧UI切り替えオプション
  - アプリ内チュートリアル

### リスク2: パフォーマンスの低下
- **軽減策**:
  - プロファイリングツールでの継続的な監視
  - 重い処理の非同期化
  - 描画の最適化

### リスク3: YOLOViewとの互換性問題
- **軽減策**:
  - YOLOViewのラッパークラス作成
  - 必要に応じてフレームワーク側の修正
  - 十分なテスト期間の確保

## テスト計画

### ユニットテスト
- 新UIコンポーネントの個別テスト
- カラー/タイポグラフィシステムのテスト
- パラメータ保存/読み込みテスト

### UIテスト
- 各画面遷移のテスト
- ジェスチャー操作のテスト
- 画面回転時のレイアウトテスト

### 統合テスト
- カメラ→推論→表示の一連の流れ
- モデル切り替え時の動作
- パラメータ変更の即時反映

### デバイステスト
- 各iPhoneモデルでの表示確認
- iPadでの動作確認
- iOS 16以降の各バージョンでのテスト

## タイムライン

```
週1-2:  フェーズ1 - 基盤整備
週3-5:  フェーズ2 - UIKit実装
週6-7:  フェーズ3 - 新機能実装
週8:    フェーズ4 - 最適化と仕上げ
週9-10: テストとバグ修正
週11:   リリース準備
```

## 将来的な検討事項

### SwiftUIへの移行
- **メリット**: 
  - より宣言的なUI構築
  - アニメーションの簡潔な実装
  - 将来性

- **デメリット**:
  - 学習コスト
  - 既存コードの大幅な書き換え
  - iOS 16以降のみサポート

**推奨**: 現時点ではUIKitで実装し、将来的にSwiftUIへの段階的な移行を検討

## まとめ

この移行戦略により、既存の機能を維持しながら、新しいデザインを段階的に実装できます。ユーザーへの影響を最小限に抑えつつ、モダンで使いやすいUIを提供することが可能です。

重要なのは、各フェーズで動作するアプリを維持し、継続的にテストとフィードバックを行うことです。