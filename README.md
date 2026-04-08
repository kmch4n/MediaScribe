# MediaScribe

MediaScribe は、`faster-whisper` を使って音声ファイルや動画ファイルを
単一ファイル単位で文字起こしするための Python CLI ツールです。

もともとは個人用の自動化スクリプトでしたが、GitHub で公開しやすいように
安全性と再利用性を意識した構成へ整理しています。

## 主な機能

- 音声ファイル・動画ファイルを 1 ファイルずつ文字起こし
- 出力形式は `txt` / `srt` / `vtt` / `json` に対応
- 既存ファイルの上書き有無を切り替え可能
- 標準出力またはログファイルへの記録に対応

## 動作要件

- Python 3.10+

## 重要

CPU 実行は推奨しません。文字起こしにかなり時間がかかるため、実用上は
GPU 環境での利用を前提にしてください。

MediaScribe は CUDA ランタイムが見つからない場合に CPU へフォールバック
しますが、これはあくまで非常用です。普段使いでは GPU を使う構成を推奨します。

## インストール

```bash
python -m pip install -r requirements.txt
```

## GPU 利用について

`faster-whisper` は内部で `ctranslate2` を使います。GPU を使うには、Python
パッケージを入れるだけでは不十分で、OS 側に CUDA / cuBLAS / cuDNN の実行時
ライブラリが必要です。

Windows で GPU を使う場合は、少なくとも `cublas64_12.dll` などの CUDA 12 系
ランタイムが読み込める状態にしてください。

CUDA ランタイムが不足していると、次のようなエラーが出ることがあります。

```text
Library cublas64_12.dll is not found or cannot be loaded
```

この場合、MediaScribe は CPU へフォールバックしますが、速度面では非推奨です。

`mediascribe` コマンドとして使いたい場合は、追加でパッケージとしてインストールします。

```bash
python -m pip install -e .
```

## 使い方

### 基本例

```bash
mediascribe transcribe ./media/lecture.mp3 --format srt --model large-v3
```

```bash
python transcribe.py ./media/lecture.mp3 --language ja
```

```bash
python -m mediascribe transcribe ./media/lecture.mp4 --format json
```

## 設定ファイル

`mediascribe.toml` を用意しておくと、モデル名や言語などの既定値を
ファイルで管理できます。このリポジトリには実際の設定ファイルとして
`mediascribe.toml` を同梱しています。

```toml
[transcribe]
model = "large-v3"
language = "ja"
device = "auto"
compute_type = "default"
output_format = "txt"

[logging]
verbose = false
log_file = "logs/mediascribe.log"
```

CLI はカレントディレクトリの `mediascribe.toml` を自動で読み込みます。
別の設定ファイルを使いたい場合は `--config` を指定します。

```bash
mediascribe --config ./configs/dev.toml transcribe ./media/lecture.mp3
```

CLI 引数を指定した場合は、設定ファイルの値より CLI 引数が優先されます。

## 文字起こし例

入力ファイルと同じ場所に文字起こし結果を書き出す:

```bash
mediascribe transcribe ./lectures/week01.mp3
```

出力先ファイルを明示して保存する:

```bash
mediascribe transcribe ./lectures/week01.mp3 --output ./outputs/week01.txt
```

タイムスタンプ付きの字幕ファイルを作る:

```bash
mediascribe transcribe ./lectures/week01.mp4 --format vtt
```

## 開発

品質確認コマンド:

```bash
black --check .
ruff check .
mypy src
python -m unittest discover -s tests
```
