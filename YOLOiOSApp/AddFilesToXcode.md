# Xcode プロジェクトへのファイル追加手順

## 追加が必要なファイル（新UI実装）

以下の新しいSwiftファイルをXcodeプロジェクトに追加してください：

### デザインシステム
1. `Colors.swift` - カラーパレット定義
2. `Typography.swift` - フォントシステム定義

### UIコンポーネント
3. `StatusMetricBar.swift` - 上部のステータスバー
4. `TaskTabStrip.swift` - タスク選択タブ
5. `ShutterBar.swift` - カメラコントロールバー
6. `RightSideToolBar.swift` - パラメータ調整ボタン
7. `ParameterEditView.swift` - パラメータ編集UI
8. `HiddenInfoViewController.swift` - 隠し情報画面

### ヘルパー
9. `ModelSizeHelper.swift` - モデルサイズ判定ヘルパー

## 追加手順

### 方法 1: ドラッグ&ドロップ（推奨）

1. **Xcode で YOLOiOSApp.xcodeproj を開く**

2. **Finder でファイルを選択**
   - Finder で `/Users/majimadaisuke/Downloads/release/yolo-ios-app/YOLOiOSApp/YOLOiOSApp` フォルダを開く
   - 上記 6 つのファイルを選択（Cmd を押しながらクリック）

3. **Xcode にドラッグ**
   - 選択したファイルを Xcode のプロジェクトナビゲーター（左側のファイルリスト）にドラッグ
   - YOLOiOSApp グループ（フォルダアイコン）の上にドロップ

4. **オプションを設定**
   ダイアログが表示されたら：
   - ✅ "Copy items if needed" はチェックしない（すでにプロジェクトフォルダ内にあるため）
   - ✅ "Create groups" を選択
   - ✅ "Add to targets: YOLOiOSApp" にチェック

### 方法 2: Xcode メニューから追加

1. **Xcode でプロジェクトナビゲーターの YOLOiOSApp グループを右クリック**

2. **"Add Files to YOLOiOSApp..." を選択**

3. **ファイルを選択**
   - ファイル選択ダイアログで上記 6 つのファイルを選択
   - 複数選択：Cmd を押しながらクリック

4. **オプションを設定して "Add" をクリック**

## ビルドエラーの確認

ファイル追加後：

1. **クリーンビルド**
   - メニュー: Product > Clean Build Folder（Shift+Cmd+K）

2. **ビルド**
   - メニュー: Product > Build（Cmd+B）

## よくあるエラーと解決方法

### "No such module 'YOLO'" エラー
- Swift Package の依存関係を確認
- File > Packages > Resolve Package Versions

### "Use of undeclared type" エラー
- ファイルがターゲットに追加されているか確認
- ファイルを選択 > 右側の File Inspector > Target Membership を確認

### Info.plist エラー
- Info.plist の構文エラーがないか確認
- Product > Clean Build Folder 後に再ビルド

## ビルド成功の確認

1. エラーがないことを確認（Issue Navigator で確認）
2. シミュレーターで実行: Cmd+R
3. コンソールログを確認

## トラブルシューティング

問題が解決しない場合：

1. **プロジェクトのバックアップを取る**
2. **DerivedData を削除**
   - ~/Library/Developer/Xcode/DerivedData を削除
3. **Xcode を再起動**
4. **プロジェクトを閉じて再度開く**