# AlwaysWhisper

[English README is here (README.md)](README.md)

**AlwaysWhisper** は、動画や音声ファイルを入れると、字幕(キャプション)を焼き込んだ動画と、字幕のテキストファイルを作ってくれる道具です。文字起こし(音声を聞き取って文字にすること)には Whisper(OpenAIが作った音声認識の仕組み)を使いますが、すべてあなた自身のパソコンの中だけで動きます。音声データをインターネット越しにどこかへ送る必要はありません。

具体的には、聞き取った内容を日本語の文法に合わせて字幕向けの短い行に分割し、タイプライターのように1文字ずつ表示されていく演出をつけて動画に焼き込みます。焼き込む前には、できあがった字幕をもう一度音声と照らし合わせて確認する自動チェック(QA)が走るので、字幕がズレたまま完成してしまうことを防げます。単語ごとの正確な発話タイミング(「word-level timestamps」= どの単語が何秒何コマ目に話されたかという情報)は、文字起こしから字幕の分割、(使う場合は)タイミングの再調整まで、パイプライン(一連の処理の流れ)を通してずっと保持されます。

## なにをする道具か（かんたんな説明）

- **聞き取って文字にする** — 動画や音声の中で話されている内容を、コンピューターが聞き取ってテキストにします(これを「文字起こし」と呼びます)。
- **字幕向けの短い行に切り分ける** — 長い文章のままだと画面に収まらないので、読みやすい長さごとに区切ります。日本語なら「です」「ます」のような文の終わり方を手がかりに、自然な位置で区切ります。
- **動画に文字を描き込む(焼き込む)** — 区切った字幕を、動画の映像そのものに書き込みます。あとから字幕をオン・オフすることはできませんが、どんな再生環境でも必ず表示されます。
- **もう一度聞いて答え合わせする(自動QA)** — 焼き込む前に、できた字幕の一部をランダムに選び、その部分の音声だけをもう一度聞き取り直して、字幕の文字と合っているか確認します。合っていない字幕が見つかったら、そこで処理を止めて教えてくれます。
- **文字が出てくる演出をつける** — 字幕は一瞬で表示されるのではなく、タイプライターのように1文字ずつ表示されていく演出つきで焼き込まれます。
- **すべて自分のパソコンの中で完結** — 音声データを外部のサーバーに送る必要はありません(オプションでネット経由のWhisperを使うことも選べます)。

## この説明で出てくる言葉

先に、これから何度も出てくる言葉をまとめておきます。ここだけ読んでおけば、以降の説明で迷わなくなります。

| 言葉 | かんたんな意味 |
|---|---|
| モデル(model) | 音声を聞いてテキストに変換する、学習済みの「頭脳」にあたるファイルです。大きいモデルほど賢く(正確に)なりますが、そのぶん動きが遅く、たくさんのメモリを必要とします。 |
| Whisper | OpenAIが作った、音声を聞き取って文字にする技術(音声認識モデル)です。AlwaysWhisperはこのWhisperを使って文字起こしをします。 |
| faster-whisper | Whisperを、より少ないメモリで・より速く動くように作り直した別実装です。AlwaysWhisperが実際に使っているのはこちらです。 |
| GPU / VRAM | GPUはパソコンの中にある画像処理専用のチップで、この手の計算をCPUよりずっと速くこなせます。VRAMはそのGPUだけが使う専用メモリのことです。 |
| CPU | パソコンの中心となる、一般的な計算を担当するチップです(GPUがない環境では、こちらがすべての計算を引き受けます)。 |
| RAM | パソコン全体が使う一時的な作業メモリです(GPU専用のVRAMとは別物です)。 |
| 精度(int8 / float16 など) | モデルの数値をどれくらい細かく保存するかという設定です。`int8` は数値を圧縮して保存する方式で、メモリを節約でき速く動きますが、精度はほんの少しだけ落ちます。`float16`・`float32` はより細かく保存する方式で、精度は高いぶんメモリを多く使います。 |
| SRT | 字幕ファイルの形式のひとつです。中身はただのテキストファイルで、「この時刻からこの時刻まで、この字幕を表示する」という情報が並んでいます。 |
| word-level timestamps(単語レベルタイムスタンプ) | 文章の中の一語一語について、「何秒何コマ目に話されたか」を記録した情報です。これがあると、字幕を単語単位でぴったり動画に合わせられます。 |
| ffmpeg | 動画・音声ファイルを読み書きするための、無料のコマンドラインツールです。AlwaysWhisperは内部でこれを使います。 |
| 仮想環境(virtual environment) | このプロジェクト専用に用意する、Pythonのパッケージ(部品)を入れておく専用フォルダです。パソコンの他のプロジェクトと部品がごちゃ混ぜにならないようにするためのものです。 |
| 環境変数(environment variable) | OS(シェル)に名前を付けて登録しておく設定値です。プログラムは起動時にこれを読み取れるので、フォルダの場所やトークンのような秘密の値を、コマンドに直接書かずに渡せます。 |
| グロッサリー / バイアスプロンプト | 「この動画にはこういう固有名詞や専門用語が出てきます」とあらかじめ教えておく仕組みです。Whisperが聞き取りに迷ったときの手がかりになり、誤変換を減らせます。 |
| QA | Quality Assurance(品質チェック)の略です。ここでは、できあがった字幕が本当に音声と合っているかを自動で確認する処理を指します。 |

## 特徴

- **完全に自分のパソコンだけで動く** — 音声を聞き取る仕組みには [faster-whisper](https://github.com/SYSTRAN/faster-whisper) を使います。APIキー(外部サービスを使うための鍵)は不要で、モデルは初めて使うときに自動でダウンロードされます。ネット越しのWhisper(OpenAIのAPI)を使いたい場合は、オプションの拡張機能として選べます。
- **日本語の文法をわかったうえで字幕を区切る** — ただ文字数を数えて機械的に切るのではなく、「です」「ます」のような文の終わり方や、「そして」のような接続表現を見つけて、自然な区切りで字幕を分けます。英語やドイツ語のように単語をスペースで区切る言語は、日本語向けの文法ルールの代わりに、よりシンプルなスペース区切りの分け方で処理します。
- **Whisperの「聞き間違いの定番」を機械的に取り除く** — Whisperは音声が無音の区間(特に動画の最後の無音部分)で、実際には話されていない「ご視聴ありがとうございました」のような決まり文句を、あたかも聞こえたかのように作り出してしまうことがあります。AlwaysWhisperはこれを機械的に検出して取り除くので、字幕にも文字起こし結果にも一切残りません(日本語向けの初期設定があり、設定で変更もできます)。
- **焼き込む前に自動で答え合わせ(自動QA)** — 動画の音声からランダムに選んだ字幕区間だけを、あらかじめの手がかり(バイアスプロンプト)なしでもう一度独立して聞き取り直し、字幕の文字と突き合わせます。一致率がしきい値を下回る区間が一つでもあれば、そこで処理を止めます。
- **文字が出てくる演出つきで焼き込む** — 動画に字幕を書き込む方法(レンダリング経路)を2つ用意しています。何度も試行錯誤したいときに向く高速な「libass/ffmpeg」経路と、ふつうのffmpegさえあればどんな環境でも動く「MoviePy/PIL」経路です。
- **字幕のタイミングだけをあとから調整できる(任意)** — 手作業などで字幕のタイミングがズレてしまった場合、各字幕の開始時刻を、もとの文字起こしで記録した「単語ごとに一番近いタイミング」に合わせ直せます。

## 必要なもの

- **Python 3.10以上** — AlwaysWhisper自体がPythonで書かれたプログラムなので、動かすためのPython本体が必要です。
- **ffmpeg** — 動画・音声ファイルを読み書きする無料のツールです([この説明で出てくる言葉](#この説明で出てくる言葉)を参照)。今のところ、動画や音声を扱うAlwaysWhisperのすべてのコマンドで必要になります。`transcribe`/`auto` は文字起こしの前にffmpegで音声だけをWAVファイルとして取り出しますし、`caption`/`qa` はQA(答え合わせ)用の音声クリップを切り出したり、標準モード(後述の`--fast`を使わない方)では焼き込み後の動画に元の音声トラックを戻したりするのにffmpegを使います。(faster-whisper自体はPyAV(Pythonから動画・音声データを直接読み書きするためのライブラリ)経由でffmpegなしに音声を直接読み込めますが、AlwaysWhisper側の音声抽出処理は今のところその機能を使っていないため、どちらのバックエンドを選んでもffmpegのインストールが前提になります。)
- **高速焼き込みモード(`--fast`)を使う場合は、libass対応のffmpeg** — 字幕を高速に焼き込む `--fast` モードは、`ass` という字幕フィルタを使うため、通常のffmpegでは動きません。「libass」という字幕描画の部品を組み込んだffmpegが別途必要です。
  - **macOS**: Homebrew(macOS用のパッケージ管理ツール)の通常の `ffmpeg` はlibassが無効になっているため、代わりに `brew install ffmpeg-full` でインストールしてください。`/opt/homebrew/opt/ffmpeg-full/bin`(Apple Silicon = M1/M2などのMac)と `/usr/local/opt/ffmpeg-full/bin`(Intel Mac)は自動的に検出されます。環境変数 `FFMPEG_LIBASS_BIN`(および `FFPROBE_LIBASS_BIN`)で、libass対応バイナリの場所を直接指定することもできます。
  - **Debian/Ubuntu**: `apt install ffmpeg` で入るふつうのffmpegに、通常すでにlibassが含まれています。
  - **Windows**: [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) などで配布されているフルビルド(機能をフルに含んだ版)を使ってください。
  - 標準(`--fast` を付けない)の焼き込みにはlibassは不要です。`PATH`(コマンドを探す場所として登録されたフォルダの一覧)上にあるffmpegであれば、どれでも動作します。

## インストール

AlwaysWhisperはまだPyPI(Pythonのパッケージを配布している公式の置き場所)には公開していません。GitHubのリポジトリから直接インストールできます。次のコマンドを実行すると、GitHub上のコードを取ってきてそのままインストールします。

```bash
pip install "git+https://github.com/kzkhykw/AlwaysWhisper.git"
```

インターネット経由のWhisper(OpenAI Whisper API)をバックエンドとして使いたい場合は、`[api]` という追加部品(エクストラ)を付けてインストールしてください。

```bash
pip install "alwayswhisper[api] @ git+https://github.com/kzkhykw/AlwaysWhisper.git"
```

このバックエンドを使うには、環境変数(OSにあらかじめ設定しておく値)か、作業ディレクトリ(作業用フォルダ)に置いた `.env` ファイルに `OPENAI_API_KEY`(OpenAIのAPIを使うための鍵)を設定しておく必要があります。

コードを手元にダウンロードしてから使いたい場合は、次のように `clone`(リポジトリをまるごと手元にコピーすること)してからインストールもできます。

```bash
git clone https://github.com/kzkhykw/AlwaysWhisper.git
cd AlwaysWhisper
pip install .
# または: pip install ".[api]"
```

同梱の `setup.sh` スクリプトを使うと、上のclone方式に加えて仮想環境(このプロジェクト専用のPythonパッケージ置き場。[この説明で出てくる言葉](#この説明で出てくる言葉)を参照)を自動で作ってくれます。

```bash
./setup.sh          # .venv を作成し、alwayswhisperをeditableモードでインストール
./setup.sh small     # さらに "small" モデルを事前ダウンロード
```

`.venv` は仮想環境が作られるフォルダの名前です。「editableモード」とは、コードを書き換えたらすぐその変更が反映される、開発者向けのインストール方法のことです。

## モデル

AlwaysWhisperは、faster-whisper用に変換済みのWhisperモデルを使います。この変換形式は「CTranslate2」と呼ばれ、モデルを高速に・少ないメモリで動かせるようにしたものです。モデルは、その名前を最初に使った瞬間に自動的にダウンロードされます。事前に個別のインストール作業は不要です。ダウンロードしたモデルは、Hugging Face(AIのモデルを配布している代表的なサイト)のHubキャッシュという場所に保存され、次回からはそこから読み込まれます。デフォルト(何も指定しなかったときに使われるモデル)は `large-v3` です(設定ファイル `src/alwayswhisper/data/default_config.yaml` の `transcribe.model` というキーで決まっています)。

### あなたのハードウェアに合わせたモデル選び

モデルにはいくつかのサイズがあります。大きいモデルほど賢く(聞き取りが正確に)なりますが、ダウンロードにも実行にも時間とメモリが余分にかかります。

| モデル | パラメータ数 | ダウンロードサイズ | 対応言語 | 備考 |
|---|---|---|---|---|
| `tiny` | 39M | 約76MB | 多言語 | パイプラインの動作確認用途のみ |
| `base` | 74M | 約145MB | 多言語 | まだ精度は低め。動作確認用途のみ |
| `small` | 244M | 約484MB | 多言語 | 手元での素早い試行錯誤に good |
| `medium` | 769M | 約1.53GB | 多言語 | 中堅クラスの安定した精度 |
| `large-v3` | 1550M | 約3.09GB | 多言語 | **精度は最良 — デフォルトであり日本語にも推奨** |
| `large-v3-turbo`（エイリアス `turbo`） | 809M | 約1.62GB | 多言語 | `large-v3` の最適化版。openai/whisperのREADMEいわく「精度の低下を最小限に抑えつつ」より高速に文字起こしできる — 実際の素材で確認すること |
| `*.en` 系サイズ、`distil-*` 系 | 様々 | 様々 | **英語専用** | `tiny.en`〜`medium.en` とDistil-Whisperモデル群（`distil-large-v3` 等）は英語専用 — 日本語には絶対に使わない |

表の読み方: 「パラメータ数」はモデルの中にある調整可能な数値の個数のことで、多いほど表現力が高い(賢い)目安になります。「M」は100万(メガ)、次の表に出てくる「GB」はギガバイト(データ量の単位)です。

パラメータ数と `turbo` の説明は
[openai/whisperのモデル一覧表](https://github.com/openai/whisper#available-models-and-languages)
から。ダウンロードサイズはHugging Face上の `Systran/faster-whisper-*`
リポジトリ(turboは `mobiuslabsgmbh/faster-whisper-large-v3-turbo`)にある
`model.bin` ファイルのサイズです。

**参考数値。** openai/whisperのREADMEには、変換前のオリジナル実装(PyTorch版)のVRAM(GPU専用メモリ。[この説明で出てくる言葉](#この説明で出てくる言葉)を参照)使用量と速度が載っています。A100という高性能GPUで英語音声を文字起こしして計測した値で、`large` モデルを基準(1倍)とした相対値です(実際の速度は言語・話す速さ・使うハードウェアによって大きく変わります)。

| サイズ | パラメータ数 | 必要VRAM | 相対速度 |
|---|---|---|---|
| tiny | 39M | 約1GB | 約10倍 |
| base | 74M | 約1GB | 約7倍 |
| small | 244M | 約2GB | 約4倍 |
| medium | 769M | 約5GB | 約2倍 |
| large | 1550M | 約10GB | 1倍 |
| turbo | 809M | 約6GB | 約8倍 |

faster-whisper(実際にAlwaysWhisperが動かしている、変換後のCTranslate2版)は、これよりかなり少ないメモリで動きます。faster-whisper自身のREADMEに載っているベンチマーク(13分の音声を、精度重視の設定「ビームサイズ5」で文字起こし)はこちらです。

| モデル | デバイス | 精度 | 時間 | メモリ |
|---|---|---|---|---|
| large-v2 | RTX 3070 Ti（GPU） | fp16 | 1分03秒 | VRAM 4525MB |
| large-v2 | RTX 3070 Ti（GPU） | int8 | 59秒 | VRAM 2926MB |
| small | i7-12700K・8スレッド（CPU） | fp32 | 2分37秒 | RAM 2257MB |
| small | i7-12700K・8スレッド（CPU） | int8 | 1分42秒 | RAM 1477MB |

表の中の「fp16」「fp32」「int8」は、数値をどれくらい細かく保存するかという精度設定です([この説明で出てくる言葉](#この説明で出てくる言葉)を参照)。上の表のlarge-v2をfp16でGPU実行した場合(4525MB)は、openai/whisperがオリジナルの `large` モデルについて挙げている約10GBの半分にも満たない値です。さらにint8に圧縮すればほぼ半分になります。`large-v2` と同じサイズの `large-v3` にも同じ傾向が当てはまると考えられます。

**ハードウェア別の選び方:**

| あなたのハードウェア | 試すべきモデル | 理由 |
|---|---|---|
| NVIDIA GPU、VRAM 8GB以上 | `large-v3`、`compute_type: float16`（または `auto` のまま。GPUが対応する最速の型が選ばれる） | GPU速度のまま最良精度 |
| NVIDIA GPU、VRAM 4〜6GB | `large-v3` + `compute_type: int8_float16`（上のlarge-v2の `int8` 行から約3GB — large-v2/v3はfloat16で保存されているため、CTranslate2への `int8` リクエストは実際には `int8_float16` として実行される）、または `large-v3-turbo` | 限られたVRAM予算に収まる |
| Apple SiliconのMac、またはRAM 16GB以上のCPUのみのPC | CPU上で `large-v3` + `compute_type: int8`、速度重視なら `large-v3-turbo` | 動作はするが、上の `small` のCPUベンチマークより何倍も遅くなると見込む — **これは推定値**: これらのドキュメントに `large-v3` のCPUベンチマークは存在しない。`small` の約6倍のパラメータ数（1550M対244M）から見積もった目安 |
| RAM約8GBのマシン | `small`（入るなら `medium`） | `large-v3` は短いクリップ限定にとどめる |
| 英語のみの素材 | `.en` または `distil-large-v3` | より小さく速い。日本語には絶対使わない（上表の対応言語列を参照） |

**`device`**(コマンドに付ける `--device`、または設定ファイルの `transcribe.device`。デフォルト `auto`): `cpu`(CPUだけを使う)、`cuda`(NVIDIA GPUを使う)、`auto`(自動判定)のいずれかを指定します。CTranslate2(モデルを動かす仕組み)のGPUアクセラレーション(GPUによる高速化)はNVIDIA製GPU専用です — 「Compute Capability」というNVIDIAのGPU世代を表す指標で3.5以上が必要で、現行リリースはCUDA 12 + cuDNN 9(どちらもNVIDIAが提供するGPU計算用のソフトウェア)を要求します(詳しくは後述の「GPUセットアップ」を参照)。Apple Silicon・AMD・IntelのGPUは、CTranslate2では一切高速化の対象になっていないため、これらのマシンでは `auto`/`cpu`/`cuda` のどれを選んでも結局CPUで処理されます。

**`compute_type`**(コマンドに付ける `--compute-type`、または `transcribe.compute_type`。デフォルト `auto`): モデルの数値をどれくらいの精度で計算するかという設定です。

- `auto` — 「このシステム・デバイスでサポートされている最速の計算タイプを使う」(CTranslate2のドキュメントより)
- `default` — モデルを変換したときの型のまま使う。Systran公式の変換版(`large-v3` のようなサイズ名でAlwaysWhisperがダウンロードするもの)はfloat16で保存されています — モデルカード(モデルの説明ページ)に「モデルの重みはFP16で保存されている」と明記されています
- **CPU**では、CTranslate2がネイティブに(そのままの形では)実行できない型を、黙って別の型に置き換えます: `float16` は `float32` に、`int8_float16`/`int8_bfloat16`/`int16` 系は `int8_float32` になります(CTranslate2の暗黙的型変換テーブルより)。そのためCPUでの実質的な選択肢は `int8`(最速・省メモリ)か `float32`(フル精度)です — このMac(Apple Silicon、ctranslate2 4.8.1)で `ctranslate2.get_supported_compute_types("cpu")` を実行して確認済みで、`{'int8_float32', 'float32', 'int8'}` と表示されます。`float16` や単独の `int8_float16` はこの中に含まれません。
- **GPU**での実質的な選択肢は `float16` か `int8_float16` です(CTranslate2の文書によれば、GPUでのint8はCompute Capability 7.0以上または6.1が必要。それより古いカードは `float16` のままにしてください)。

**CPUスレッド数。** faster-whisperの `cpu_threads`(CPUで同時に計算に使うスレッド = 処理の並列単位の数)はデフォルトで0ですが、これはドキュメント上「デフォルトで4、0以外の値を指定するとOMP_NUM_THREADS環境変数を上書きする」という意味です。ただしAlwaysWhisperの `FasterWhisperBackend` は `cpu_threads` をオプションとして公開していません(`WhisperModel` へは `model`・`device`・`compute_type` しか渡していません)。4コア(CPUの処理単位)より多く使いたい場合は、実行前に `OMP_NUM_THREADS` を設定してください。

```bash
OMP_NUM_THREADS=8 alwayswhisper transcribe clip.mp4 --model large-v3 --device cpu --compute-type int8
```

(スレッド数ではなく、物理コア数を指定すること)

**自分の環境でベンチマークする。** 上の数値はすべて他人のハードウェアでの計測値です。目安に過ぎないので、手元の1〜2分程度のクリップ(動画の断片)で実際に時間を計ってみるのが確実です。`time` は、コマンドの実行にかかった時間を表示するツールです。

```bash
time alwayswhisper transcribe clip.mp4 --model small
time alwayswhisper transcribe clip.mp4 --model large-v3
```

`caption`/`qa` の自動AV QA(音声と字幕を突き合わせる自動チェック)で行う再文字起こしは、`caption`/`qa` 自体に個別の `--model` を渡さない限り `transcribe.model`(最初の文字起こしに使ったのと同じモデル)を再利用します。

コマンドごとにモデル・デバイス・計算タイプを指定するには:

```bash
alwayswhisper transcribe clip.mp4 --model large-v3 --device cpu --compute-type int8
```

または、設定ファイル(何度も使う設定をまとめて書いておけるテキストファイル。詳しくは後述の[設定](#設定)を参照)にまとめて書いておく方法もあります(`--config config.yaml` で読み込みます。`auto` コマンドでも読み込まれます):

```yaml
transcribe:
  model: large-v3
  device: cpu
  compute_type: int8
```

### モデルのインストールと事前ダウンロード

事前にインストールしておく必要はありません — 最初に `transcribe`/`auto`/`caption`/`qa` のどれかを実行したときに、指定されたモデルが自動的にダウンロードされます。文字起こしはせずモデルだけ先に取得しておきたい場合(オフラインになる前にキャッシュを準備しておきたい、CI(自動テストの仕組み)実行の前に済ませておきたい、など)は、次のコマンドを使います。

```bash
alwayswhisper prefetch --model large-v3
```

または、cloneした直後に `setup.sh` にやらせる方法もあります:

```bash
./setup.sh large-v3
```

**モデルの保存先。** Hugging FaceのHubキャッシュに保存され、デフォルトの場所は `~/.cache/huggingface/hub` です。AlwaysWhisperを実行する前に環境変数 `HF_HOME`(Hugging Faceキャッシュのルート全体を変更する)または `HF_HUB_CACHE`(Hubキャッシュだけを変更する)を設定すれば、保存先を変更できます。

**オフライン利用。** モデルが一度キャッシュされたら、`HF_HUB_OFFLINE=1` という環境変数を設定すると、Hugging FaceのHubへの確認そのものが発生しなくなります(ネットワーク接続が不要になります)。ネットに繋がっていない別のマシンに持ち込むには、キャッシュディレクトリ全体(またはその中の目的のモデルだけが入った `models--<org>--<name>` というサブフォルダ)をコピーしてください。

**`--model` に指定できる値**(`transcribe`、`auto`、`caption`、`qa`、`prefetch` 共通):
- 上表のサイズ名(`tiny`、`small`、`large-v3`、`large-v3-turbo` など)
- Hugging Face Hub上の、CTranslate2変換済みWhisperリポジトリのID(例: `Systran/faster-whisper-large-v3`)
- ローカルの変換済みモデルディレクトリへのパス(フォルダの場所) — `transcribe`/`auto`/`caption`/`qa` では使えますが、`prefetch` では使えません(すでにディスク上にあるモデルを、あらためてダウンロードする意味がないためです)

### カスタムモデル・ファインチューニング済みモデルの利用

自分で追加学習(ファインチューニング)したWhisperのチェックポイント(学習済みの重みファイル)や、faster-whisperがまだ変換版を提供していないモデルを使いたい場合は、先にCTranslate2形式へ変換します(faster-whisperのREADME「Model conversion」より)。

```bash
pip install "transformers[torch]>=4.23"

ct2-transformers-converter --model openai/whisper-large-v3 --output_dir whisper-large-v3-ct2 \
    --copy_files tokenizer.json preprocessor_config.json --quantization float16
```

`openai/whisper-large-v3` の部分を自分のチェックポイント(Hugging Face Hub上のリポジトリID、またはローカルディレクトリ)に置き換えてください。変換後は、出力ディレクトリをAlwaysWhisperに指定します。

```bash
alwayswhisper transcribe talk.mp4 --model ./whisper-large-v3-ct2
```

または、[変換したディレクトリをHugging Face Hubにアップロード](https://huggingface.co/docs/transformers/model_sharing#upload-with-the-web-interface)して、`large-v3` と同じ書き方で名前指定することもできます。

```bash
alwayswhisper transcribe talk.mp4 --model username/whisper-large-v3-ct2
```

### GPUセットアップ（NVIDIA）

`--device cuda` は、NVIDIA製のGPU(Compute Capability 3.5以上)でだけ高速化されます。Apple Silicon・AMD・IntelのGPUは、このフラグを指定しても常にCPU経路にフォールバック(切り替わる)します。現行のCTranslate2リリースは**CUDA 12 + cuDNN 9**(どちらもNVIDIAが配布するGPU計算用ソフトウェア)を要求します。

Linuxでは、システム全体にCUDAをインストールしなくても、pip(Pythonのパッケージインストーラー)でNVIDIA製のライブラリだけを入れられます。ただしPythonを起動する前に、ライブラリの場所を教える環境変数 `LD_LIBRARY_PATH` を設定しておく必要があります。

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*

export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
```

Windowsでは、Purfviewの
[whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win)
が配布しているライブラリ一式を使うか、CUDA Toolkit + cuDNNを自分でインストールしてください。

古いドライバのままの場合は、`ctranslate2` 自体をバージョンダウン(ダウングレード)してください: CUDA 11 + cuDNN 8の組み合わせなら `pip install --force-reinstall ctranslate2==3.24.0`、CUDA 12 + cuDNN 8の組み合わせなら `ctranslate2==4.4.0` を使います。

## クイックスタート

まず、いちばん手っ取り早い、ワンショット(ひとつのコマンドで最初から最後まで一気に片づける)方法です。

```bash
alwayswhisper auto talk.mp4 -o final.mp4 --max-chars 21 --fast
```

このコマンド1つで、文字起こしから字幕の焼き込みまで全部終わり、`final.mp4` という字幕入り動画ができあがります。

ただし、おすすめの手順は、字幕を焼き込む**前**に文字起こしのテキストを見直して間違いを直すことです(テキストの中身を書き換えるだけなら、エントリ = 字幕の行そのものを追加・削除さえしなければ、Whisperが記録した単語ごとのタイミング情報はそのまま有効に使えます)。

```bash
# 1. 単語タイムスタンプ + 生SRTへ文字起こし
alwayswhisper transcribe talk.mp4 -o talk_alwayswhisper --language ja --max-chars 21

# 2. talk_alwayswhisper/transcript_raw.srt を手で編集する:
#    人名・同音異義語・誤認識を修正し、タイムスタンプはそのままにする。

# 3. 修正済みキャプションを焼き込む
alwayswhisper caption talk.mp4 talk_alwayswhisper/transcript_raw.srt -o final.mp4 --fast
```

それぞれのステップで何ができるかというと、**1**は聞き取り結果を単語単位のタイミング情報つきで書き出し(`transcript_words.json`)、まだ手直し前の字幕ファイル(`transcript_raw.srt`)を作ります。**2**はそのファイルをテキストエディタで開いて、間違いを直す作業です(タイムスタンプの数字自体は触りません)。**3**は直し終えた字幕を動画に焼き込み、最終的な `final.mp4` を作ります。

**日本語向けのレシピ(おすすめの組み合わせ)**: `--language ja --max-chars 21` を指定しておくと、よくあるフォントサイズであればほとんどの字幕が1行に収まります(文字サイズと1行の文字数の関係は、後述の[キャプションスタイル](#キャプションスタイル)で詳しく説明します)。人名や専門用語など、Whisperが聞き間違えやすい固有名詞を正しく聞き取ってほしいときは、`--glossary terms.txt`(正しい表記を1行ずつ並べたテキストファイル)を使ってください。内部で1行にまとめた「ヒント文」(バイアスプロンプト)に変換され、Whisperがヒントとして読める文字数の上限に収まるよう自動的に短く切り詰められます。

## CLIリファレンス

どのサブコマンド(`transcribe` や `caption` のような、個別の機能を実行するコマンド)も、共通のグローバルオプション `--config FILE` を受け付けます。これは、パッケージに同梱されているデフォルト設定に重ね合わせる(ディープマージする)YAMLファイルです(詳しくは後述の[設定](#設定)を参照)。`alwayswhisper --version` は、インストール済みのバージョン番号を表示して終了します。以下で「パッケージデフォルト」と表記しているフラグ(コマンドに付けるオプション)の既定値は、特に個別のデフォルトが明記されていない限り `data/default_config.yaml` という設定ファイルに由来します。

### `alwayswhisper transcribe INPUT`

動画・音声ファイルを、単語タイムスタンプ(各単語の発話タイミング)付きのデータと、SRT字幕ファイルに変換します。

| フラグ | 説明 |
|---|---|
| `-o, --output DIR` | 出力ディレクトリ（デフォルト: INPUTと同じ場所の `<INPUT stem>_alwayswhisper`） |
| `--backend BACKEND` | 文字起こしバックエンド: `faster-whisper` または `openai-api` |
| `--model MODEL` | モデル名（faster-whisperのみ有効。openai-apiは常に `whisper-1`） |
| `--language LANGUAGE` | 言語コード（例: `ja`、`en`） |
| `--device DEVICE` | faster-whisperのデバイス: `cpu`、`cuda`、`auto` |
| `--compute-type COMPUTE_TYPE` | faster-whisperの計算タイプ |
| `--prompt TEXT` | Whisperのバイアスプロンプト（`--glossary` より優先） |
| `--glossary FILE` | 文字起こしをバイアスさせる語彙を書いたテキストファイル |
| `--max-chars MAX_CHARS` | キャプション1エントリあたりの最大文字数 |
| `--min-chars MIN_CHARS` | キャプション1エントリあたりの最小文字数 |
| `--vad-filter` | faster-whisperの音声区間検出（VAD）フィルタを有効化 |

出力: 出力ディレクトリに `transcript_words.json`(単語レベルタイムスタンプ)と `transcript_raw.srt`。

### `alwayswhisper segment WORDS_JSON -o OUT_SRT`

すでにある `transcript_words.json` ファイルを、あらためてSRT字幕に分割し直すコマンドです。単語のテキストを手で修正した後にもう一度分割し直したいときや、`--max-chars`(1行の最大文字数)を変更してやり直したいときに便利です。

| フラグ | 説明 |
|---|---|
| `-o, --output OUT_SRT` | 出力SRTパス（必須） |
| `--language LANGUAGE` | 言語コード — セグメンターの選択に使われる（ja/zh/yue/th/lo/myは文字ベース、それ以外はスペース区切り） |
| `--max-chars MAX_CHARS` | キャプション1エントリあたりの最大文字数 |
| `--min-chars MIN_CHARS` | キャプション1エントリあたりの最小文字数 |

### `alwayswhisper caption VIDEO SRT -o OUT_MP4`

すでにある字幕ファイル(SRT)を、動画に焼き込むコマンドです。

| フラグ | 説明 |
|---|---|
| `-o, --output OUT_MP4` | 出力動画パス（必須） |
| `--words FILE` | `transcript_words.json`。`--realign` を使う場合に必須 |
| `--edit-plan FILE` | `edit_plan.json`。`--realign` と併用 |
| `--style NAME_OR_PATH` | キャプションスタイル: `default`、`en`、またはYAMLファイルへのパス |
| `--fast` | MoviePy/PILの代わりに高速なlibass/ffmpeg焼き込みを使用 |
| `--no-qa` | AV QAスポットチェックをスキップ |
| `--qa-samples QA_SAMPLES` | AV QAのサンプル数 |
| `--qa-min-ratio QA_MIN_RATIO` | AV QAの最小マッチ率 |
| `--realign` | 焼き込み前にSRT開始時刻を単語タイムスタンプにスナップ |
| `--backend BACKEND` | AV QAの再文字起こしに使う文字起こしバックエンド |
| `--model MODEL` | AV QAバックエンドのモデル名 |
| `--language LANGUAGE` | AV QAバックエンドの言語コード |

出力: 焼き込み済み動画。(`--no-qa` でない限り)その隣に `qa_report.json`。`--realign` を使った場合はさらに `<output stem>.realigned.srt`。

### `alwayswhisper qa VIDEO SRT`

動画への焼き込みは行わず、SRT字幕が動画の音声とちゃんと合っているかだけを単独でチェックするコマンドです。外部のツールで編集したSRTを、本番の焼き込みの前に検証したいときに便利です。

| フラグ | 説明 |
|---|---|
| `--samples SAMPLES` | チェックするサンプル数 |
| `--min-ratio MIN_RATIO` | 合格とみなす最小マッチ率 |
| `--backend BACKEND` | 再文字起こしに使う文字起こしバックエンド |
| `--model MODEL` | QAバックエンドのモデル名 |
| `--language LANGUAGE` | QAバックエンドの言語コード |

標準出力(ターミナルの画面)にサンプルごとのレポートを表示し、SRTの隣に `qa_report.json` を書き出します。QAが失敗した場合は終了コード1(「失敗した」という意味の合図)で終了します。

### `alwayswhisper auto INPUT -o OUT_MP4`

文字起こしと焼き込みを1つのコマンドで最初から最後まで行う、エンドツーエンド(端から端まで)コマンドです。

| フラグ | 説明 |
|---|---|
| `-o, --output OUT_MP4` | 出力動画パス（必須） |
| `--workdir DIR` | 文字起こしの中間生成物を置く作業ディレクトリ（デフォルト: OUTPUTと同じ場所の `<OUTPUT stem>_work`） |
| `--style NAME_OR_PATH` | キャプションスタイル: `default`、`en`、またはYAMLファイルへのパス |
| `--fast` | MoviePy/PILの代わりに高速なlibass/ffmpeg焼き込みを使用 |
| `--no-qa` | AV QAスポットチェックをスキップ |
| `--realign` | 焼き込み前にSRT開始時刻を単語タイムスタンプにスナップ |
| `--backend BACKEND` | 文字起こしバックエンド: `faster-whisper` または `openai-api` |
| `--model MODEL` | モデル名（faster-whisperのみ） |
| `--language LANGUAGE` | 言語コード（例: `ja`、`en`） |
| `--device DEVICE` | faster-whisperのデバイス: `cpu`、`cuda`、`auto` |
| `--compute-type COMPUTE_TYPE` | faster-whisperの計算タイプ |
| `--glossary FILE` | 文字起こしをバイアスさせる語彙を書いたテキストファイル |
| `--max-chars MAX_CHARS` | キャプション1エントリあたりの最大文字数 |
| `--min-chars MIN_CHARS` | キャプション1エントリあたりの最小文字数 |
| `--vad-filter` | faster-whisperの音声区間検出（VAD）フィルタを有効化 |

出力: キャプション付き動画、その隣に実際に焼き込んだSRTと同内容の `.srt`、(`--no-qa` でない限り)`qa_report.json`。

### `alwayswhisper prefetch`

文字起こしは行わず、faster-whisperのモデルだけを事前にダウンロードしておくコマンドです。

| フラグ | 説明 |
|---|---|
| `--model MODEL` | ダウンロードするモデル名（デフォルト: `large-v3`） |

正確で最新のフラグ一覧は、各サブコマンドに `--help` を付けて実行して確認してください。

## 設定

`--config config.yaml` はAlwaysWhisperのパッケージ同梱デフォルトに対して**ディープマージ**されます。一部のパイプラインとは異なり、設定ファイルは部分的な記述で構いません — 設定しなかったキーは、セクションごと・キーごとにパッケージデフォルトの値のまま残ります。CLIフラグはその上にさらに重ね合わされます(実際に指定したフラグのみが反映され、指定しなかったフラグが `--config` の値を上書きすることはありません)。

`src/alwayswhisper/data/default_config.yaml` に対応する全キー一覧です。「デフォルト」は何も設定しなかったときに使われる値、「意味」はそのキーが何を制御するかです。

| キー | デフォルト | 意味 |
|---|---|---|
| `transcribe.backend` | `faster-whisper` | `faster-whisper`（ローカル）または `openai-api`（ホスト型、`[api]` エクストラが必要） |
| `transcribe.model` | `large-v3` | faster-whisperのモデル名。openai-apiバックエンドでは無視される（常に `whisper-1`） |
| `transcribe.device` | `auto` | faster-whisperのデバイス: `cpu`、`cuda`、`auto` |
| `transcribe.compute_type` | `auto` | faster-whisperの計算タイプ（例: `int8`、`float16`） |
| `transcribe.language` | `ja` | バックエンドに渡す言語コード。キャプションのセグメンター選択にも使われる（[言語サポート](#言語サポート)を参照） |
| `transcribe.prompt` | `null` | 任意のWhisperバイアスプロンプトのテキスト。`null` は「なし」（または `--glossary FILE` を使う） |
| `transcribe.prompt_max_tokens` | `224` | バイアスプロンプトは、Whisper自身のプロンプト予算に合わせてこの推定トークン数まで（先頭側から）切り詰められる |
| `transcribe.strip_phrases` | `null` | `null` は「`language` が `ja` のときは組み込みの日本語幻聴句リストを使う（それ以外の言語ではなし）」という意味。`[]` は除去を無効化。空でないリストを指定するとデフォルトを丸ごと置き換える |
| `srt.max_chars` | `null` | キャプション1エントリあたりの最大文字数。`null` は35（ja/zh/yue/th/lo/my）または42（その他の言語）に解決される |
| `srt.min_chars` | `4` | キャプション1エントリあたりの最小文字数 |
| `caption.style` | `null` | `null` はパッケージ同梱のデフォルトスタイル。`"default"`、`"en"`、またはスタイルYAMLへのパスも指定可能 |
| `caption.fast_mode` | `false` | `true` にするとlibass/ffmpeg焼き込みを使用（libass対応ffmpegが必要。[必要なもの](#必要なもの)を参照） |
| `caption.realign` | `false` | 焼き込み前にSRT開始時刻を単語レベルタイムスタンプにスナップする（[単語レベル再アライン](#単語レベル再アライン)を参照） |
| `qa.enabled` | `true` | 焼き込み前に自動AV QAスポットチェックを実行する |
| `qa.samples` | `5` | スポットチェックするキャプションエントリの数 |
| `qa.min_ratio` | `0.5` | サンプルが合格とみなされる最小ファジーマッチ率 |
| `qa.pad_ms` | `300` | サンプリングした各エントリの音声を再抽出する際に前後に加えるパディング（ミリ秒） |
| `qa.min_entry_ms` | `500` | これより短いエントリはサンプリング対象から完全に除外される（[QA](#qa)を参照） |

パッケージ同梱YAMLには含まれていないものの認識されるキーが一つあります: `transcribe.vad_filter`(`--vad-filter` または自前の `config.yaml` で設定可能)は、未指定時はコード側のデフォルトで `false` になります。

## キャプションスタイル

`--style default|en|/path/to/style.yaml` で、字幕の見た目のスタイルを選びます。AlwaysWhisperには2つのスタイルが同梱されています(`data/styles/default.yaml` は日本語向けにCJK(中国語・日本語・韓国語)フォントの並びと `font_size: 64` を設定したもの、`data/styles/en.yaml` はラテン文字(アルファベット)向けフォントの並びで `font_size: 44`)。同じ構造のYAMLファイル(設定を書くためのテキストファイル形式)のパスを渡せば、見た目を完全にカスタマイズできます。注釈つきの構造は次の通りです。

```yaml
position:
  align: "center_bottom"     # 現状これ以外の値はサポートされていない
  margin_bottom: 60          # 画面下端からのpx

background:
  color: [0, 0, 0, 200]      # [R, G, B, A]
  corner_radius: 12          # px。standardモードのみ有効（下表を参照）
  padding: 16                # テキスト周囲の余白（px）

text:
  color: "#FFFFFF"
  font_family:                # 順に試される。ファイルパスとインストール済み
    - "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"   # ファミリー名のどちらも使える
    - "Hiragino Sans"
  font_size: 64
  font_weight: "bold"        # fastモードのみ有効（下表を参照）

shadow:
  enabled: true
  offset: [2, 2]              # [x, y] px
  color: "#333333"

animation:
  type: "typewriter"          # 現状これ以外の値はサポートされていない
  completion_ms: 400          # 1エントリを全て表示し終えるまでの時間
```

**fastモードとstandardモードのレンダリング差異** — どちらの経路(焼き込みの2つの方式。[特徴](#特徴)を参照)も同じスタイルファイルを読みますが、libass(fast)とPIL/MoviePy(standard)とでは、表現できる見た目が完全には一致しません。

| スタイルキー | Standard（MoviePy/PIL） | Fast（libass/ffmpeg） |
|---|---|---|
| `background.corner_radius` | 反映される — 角丸の背景 | **無視される** — libassの背景ボックスは常に矩形 |
| `shadow.color` | テキストの背後に描画される本来のドロップシャドウの色 | 代わりに背景ボックスの**枠線（outline）**の色として再利用される |
| `shadow.offset` | X・Yどちらも反映される | Xのみ反映される（ASSのシャドウ距離はスカラー値1つのみ） — Yは**無視される** |
| `text.font_weight` | **無視される** — 太字かどうかは `font_family` が解決するフォントファイル自体で決まる | 反映される — libassが `Bold` フラグから太字を合成・選択する |

**フォントサイズの目安**: 720p(画面の横幅が1280ピクセルの画質)では、1行あたりの最大文字数はだいたい `font_size × 0.20` で計算できます(96px ≈ 19文字、64px ≈ 28文字、48px ≈ 38文字)。いちばん長い字幕が1行に収まる `font_size` を選んでください — これは見た目以上に重要なポイントです。standardモードには**折り返し機能が一切なく**、幅が収まらない字幕はそのまま画面の両端をはみ出し、フレーム(1コマの画像)の境界でちぎれたように表示されます。fastモードのlibass `WrapStyle: 0` による折り返しも、日本語・CJKテキストには区切り用のスペースがないため当てになりません。実際には、幅が収まらない字幕はどちらのモードでも折り返されずに欠けると考えてください。また日本語セグメンター(字幕を区切る部品)の `--max-chars` はハードな上限(絶対に超えない値)ではなく、ソフトな目安でもあります: 文の途中で切らないよう、きりの良い文末表現に着地する場合は、上限を数文字超えることがあります。したがって `font_size` は `max_chars` ぴったりに合わせるのではなく、少し余裕を持たせて選んでください。

## 単語レベル再アライン

ときどき、字幕の**文字**は合っているのに、表示される**タイミング**だけが少しずつズレていくことがあります。たとえば、手作業やAIによる文章修正を何度も重ねると、1つ1つはわずかでも、動画の後半に行くほどズレが積み重なっていきます。これを「ドリフト」と呼びます。`--realign` は、このズレを直すためのオプトイン(自分で明示的に指定したときだけ動く)機能です。

`--realign`(`--words transcript_words.json` と一緒に使います)を付けると、各字幕エントリの開始時刻が、もとの文字起こしで記録しておいた単語ごとのタイミングのうち、いちばん近いものに「スナップ」(ぴったり吸い寄せられるように補正)されます。各エントリの表示時間の長さそのものは変わりません。補正した結果、前後のエントリが時間的に重なってしまう場合は、前のエントリの終了時刻を次のエントリの開始時刻に合わせて短縮し、重なりを解消します(`prev.end <= next.start` というルールです)。

再アライン対象のSRTが、**カット済みの音声**から作られたもの(もとの文字起こしとこのSRTの間で、外部の編集ツールなどによってフィラー語や間(ま)が取り除かれている場合)は、`--edit-plan edit_plan.json` もあわせて渡してください。これは `filler_removals`/`pause_removals` というリスト(除去した箇所の一覧)を持つJSONファイルで、各エントリはSRTのタイムスタンプ形式の `start`/`end` のペアと、任意で明示的な `removed_ms`(実際に除去した時間の長さ・ミリ秒単位)を持ちます。これを渡すことで、再アラインの計算は実際にカットされた箇所も考慮に入れられます。再アライン済みのコピーは、出力の隣に `<output stem>.realigned.srt` というファイル名で保存されます。

## QA

QAとは、「本当に字幕と音声が合っているか」を、焼き込む前に自動で答え合わせする仕組みです(QAの意味は[この説明で出てくる言葉](#この説明で出てくる言葉)を参照)。字幕を動画に焼き込む前に(`qa.enabled: false` や `--no-qa` を指定していない限り)、AlwaysWhisperは次のことを行います。

1. `qa.min_entry_ms` 以上の長さがある字幕エントリの中から、`qa.samples` 個をランダムに選びます。
2. 選んだ字幕それぞれについて、対応する部分の音声を(前後に `qa.pad_ms` ぶんの余白を付けて)切り出します。
3. その音声を、**ヒント(バイアスプロンプト)なしで**もう一度独立に聞き取り直します。ヒントを与えないのがポイントです。もしヒントを与えてしまうと、すでにズレている字幕の文字をヒントとして使ってしまい、間違いを「確認できました」と誤判定しかねないからです。
4. 聞き取り直した結果からWhisperの定型幻聴句を取り除き、両方のテキストを同じ形に正規化(全角半角の統一・小文字化・句読点の除去)したうえで、`difflib.SequenceMatcher`(Pythonの標準ライブラリにある、2つの文字列がどれくらい似ているかを計算する仕組み)でファジーマッチング(完全一致でなくても、どれくらい似ているかで判定すること)します。

一致率が `qa.min_ratio` を下回るサンプルが一つでもあれば、チェック全体が「失敗(FAIL)」となります。`caption`/`auto` コマンドはこの場合エラーを出して停止し、字幕の焼き込みを一切行いません。合格・不合格のどちらの場合も `qa_report.json` というファイルが書き出されるので、何が原因で失敗したのか、あとから確認できます。

チューニング(調整のしかた):

- `qa.samples` — チェックするエントリの数です(多いほど時間はかかりますが、広い範囲を確認できます)。
- `qa.min_ratio` — ファジーマッチをどれくらい甘く判定するかです(値が低いほど甘くなり、合格しやすくなります)。
- `qa.min_entry_ms` — **短いエントリは「偽陽性」(本当は正しいのに間違いだと判定されてしまうこと)で失敗しやすい**という注意点があります。0.5秒程度の短い字幕では、聞き取り直したときの単語の区切り位置がほんの少しズレただけでも、内容自体は合っているのに一致率が低く出てしまうことがあります。原因のわからない失敗が続く場合は、`qa.min_entry_ms` を(たとえば `2000` に)上げて、短いエントリを最初からサンプリング対象から外してください。
- `--no-qa`(または `qa.enabled: false`)は、このチェックそのものを丸ごとスキップします。この場合、聞き取り直す処理自体が行われません。

## Python API

コマンドラインからではなく、自分のPythonプログラムの中からAlwaysWhisperの機能を直接呼び出したい場合(たとえば、他の処理と組み合わせて自動化したいときなど)は、次のようにPython API(プログラムからプログラムを呼び出すための窓口)を使えます。

```python
from alwayswhisper import load_config, transcribe_file, caption_video, auto_run

cfg = load_config(overrides={
    "transcribe": {"language": "ja"},
    "srt": {"max_chars": 21},
    "caption": {"fast_mode": True},
})

# ワンショット: 文字起こし + キャプション焼き込み
report = auto_run("talk.mp4", "final.mp4", cfg)
print(report["srt_path"], report["caption"]["qa_report_path"])

# あるいは2ステップに分けて実行（間でSRTを手動編集する場合など）
transcribe_report = transcribe_file("talk.mp4", "talk_work", cfg)
# ... ここで transcribe_report["srt_path"] をディスク上で編集してもよい ...
caption_report = caption_video(
    "talk.mp4",
    transcribe_report["srt_path"],
    "final.mp4",
    cfg,
    words_json=transcribe_report["words_path"],
)
```

引数なしで呼び出した `load_config()` は、パッケージ同梱のデフォルト設定をそのまま返します。YAMLファイルを指定するには `config_path=` を、辞書(Pythonのキーと値の組み合わせのデータ)を指定するには `overrides=` を渡してください。どちらもコマンドラインの `--config` と同じ方式で、デフォルト設定に重ね合わせ(ディープマージ)されます。

## 言語サポート

AlwaysWhisperは、まず**日本語**にいちばん合うように作られています。字幕をどこで区切るか決める部品のことを「セグメンター」と呼びますが、どちらのセグメンターを使うかの振り分けは、faster-whisperのトークナイザー(文章を単語や記号の単位に分解する下ごしらえの部品)が内部で持っている「単語の間にスペースを使わない言語」の一覧をそのまま踏襲しています。

- `ja`(日本語)、`zh`(中国語)、`yue`(広東語)、`th`(タイ語)、`lo`(ラオス語)、`my`(ミャンマー語) → 文字ベースのセグメンターが使われます。この文末検出のルール(「です」「ます」のような語尾、接続表現の検出など)は日本語向けに特化していますが、この6言語のうち日本語以外の言語に対しては、単純に文字数の上限で区切るだけの、より簡素な動き方に自然に縮退します。それでも、単語と単語の間にスペースを入れない書記体系(文字の書き方の体系)向けのセグメンターとしては、正しい"種類"の挙動になっています。
- それ以外の言語は、**設定されていない場合や、認識できない言語コードの場合も含めて**すべて、英語のようにスペースで単語を区切る言語向けの、よりシンプルなセグメンターで処理されます。

Whisperの定型幻聴句を取り除く機能(`transcribe.strip_phrases`。[特徴](#特徴)を参照)は日本語向けの機能で、初期設定では `transcribe.language` が `ja` のときにだけ自動的に有効になります。日本語以外の言語でこの機能を使いたい場合は、自分で除去したい言葉のリストを設定してください。使わない場合は `[]`(空のリスト)のままにしておきます。

## オプション: 結果を Notion に保存する（コーディングエージェント向けの仕様メモ）

**先にはっきりさせておきたいこと**: この機能はAlwaysWhisper自体には実装されていません。ここから先に書いてあるのは、Claude CodeやCodexのような「コーディングエージェント」(人に代わってコードを書いて実行してくれるAIツール)にそのまま渡せば、あなたの手元でこの機能を実装できるように書いた仕様メモです。AlwaysWhisperがすでに作ってくれているファイル(文字起こし結果のSRTファイルや、字幕入りの動画など)の上に追加で作る想定で、AlwaysWhisper本体のコードを直接変更する必要はありません。

### 人間が先にやっておく準備

コードを書く前に、人間が手作業でやっておくことが5つあります。

1. **Notionで「内部連携(インテグレーション)」を作る** — Notionの管理画面(https://www.notion.so/profile/integrations)で、新しい内部コネクションを作ります。これは、あなたのプログラムがNotionのデータを読み書きしてよいと許可するための「通行証」のようなものです。
2. **トークンをコピーする** — 作った連携の「Configuration」タブに表示される、インストール用のアクセストークン(さきほどの「通行証」の実体となる文字列)をコピーしておきます。
3. **保存先のデータベースを、その連携と共有する** — Notion側で、字幕データを保存したいデータベースを開き、ページメニューから「Connections」(または「Connect to」)を選んで、さきほど作った連携と共有します。これをしないと、トークンを持っていてもプログラムからそのデータベースが見えません。
4. **データベースIDをURLからコピーする** — Notionでそのデータベースをブラウザで開いたときのURLの中に含まれる、英数字の並び(データベースID)を控えておきます。
5. **環境変数を2つ設定する** — `NOTION_TOKEN`(手順2でコピーしたトークン)と `NOTION_DATABASE_ID`(手順4でコピーしたID)を、環境変数(OSに設定しておく値。[この説明で出てくる言葉](#この説明で出てくる言葉)を参照)として設定します。

### 作るもの

たとえば `alwayswhisper notion-push` のような新しいサブコマンドか、同じことをする小さな独立したPythonスクリプトを作ります。これは、AlwaysWhisperの作業フォルダ、またはSRTファイルを受け取り、動画1本につきNotionのページを1つ作る、というものです。読み込む相手は、すでにAlwaysWhisperが書き出しているファイルです。

- `transcribe` サブコマンドは、作業ディレクトリに `<workdir>/transcript_raw.srt`(字幕ファイル)と `<workdir>/transcript_words.json`(単語レベルタイムスタンプ)を書き出します。
- `caption`/`auto` サブコマンドは、`-o` で指定した場所に字幕入りの動画を書き出し、あわせてQAレポート(答え合わせの結果)も書き出します。
- Python API([Python API](#python-api)を参照)を使う場合、`transcribe_file(...)` は `srt_path` を含む辞書(データのまとまり)を返すので、そこからSRTの場所を取得できます。

ページのプロパティ(データベースの列にあたる項目)の案は次の通りです。

| プロパティ名の案 | 型 | 内容 |
|---|---|---|
| タイトル | title | 動画のファイル名 |
| 収録日 | date | 動画を収録した(または処理した)日付 |
| 長さ(秒) | number | 動画の長さを、数値として秒単位で |
| 言語 | select | 文字起こしに使った言語コード(例: `ja`) |
| 動画URL | url | (任意)動画を置いた場所のURL |

ページの本文には、文字起こしのテキストを段落ブロック(Notionページの中の、ひとかたまりの文章の単位)として書き込みます。

### API呼び出しの順番

Notion側とやり取りする処理(API呼び出し)は、次の順番で行います。「API」とは、プログラム同士が決まった形式でデータをやり取りするための窓口のことです。ここに書く内容は、developers.notion.com(Notionの公式開発者向けサイト)で2026-09-04に確認した情報にもとづいています。

1. **データソースIDを取得する** — Notionのデータベースは、内部に1つ以上の「データソース」を持っています(2025-09-03以降の仕様)。ページは、データベース自体にではなく、このデータソースの下に作ります。まず `GET /v1/databases/{database_id}` を呼び出すと、返ってくるデータの中に `data_sources: [{id, name}, ...]` という一覧が入っているので、この `id` を控えます。
2. **ページを作る** — `POST /v1/pages` を、次のようなボディ(送信するデータの中身)で呼び出します。

    ```json
    {
      "parent": {"type": "data_source_id", "data_source_id": "<手順1で控えたid>"},
      "properties": { "...": "上の表のプロパティを、下記の形式で" }
    }
    ```

    プロパティの値は、型ごとに次のような形式で書きます(プロパティ名は実際のデータベースの列名に合わせてください)。

    ```json
    {
      "Name": {"title": [{"text": {"content": "動画ファイル名"}}]},
      "Summary": {"rich_text": [{"text": {"content": "..."}}]},
      "Recorded": {"date": {"start": "2026-09-04"}},
      "Duration (s)": {"number": 613.2},
      "Video": {"url": "https://..."},
      "Language": {"select": {"name": "ja"}}
    }
    ```

3. **本文のブロックを分割して追加する** — ページの本文(文字起こしテキスト)は、`PATCH /v1/blocks/{page_id}/children` を、次のようなボディで呼び出して追加します。

    ```json
    {
      "children": [
        {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": "..."}}]}}
      ]
    }
    ```

4. **429エラーが返ってきたら、`Retry-After` ヘッダーの秒数だけ待ってから再試行する** — Notion側の混雑などで、一時的にリクエストが受け付けられなかったときのサイン(HTTPステータスコード429)です。

すべてのリクエストには、次の2つのヘッダー(リクエストに添える付加情報)を必ず付けます。

- `Notion-Version: 2026-03-11`(現行バージョン。古い `2022-06-28` も動きますが、上で説明した「データソース」が導入される前の仕組みのままです)
- `Content-Type: application/json`

送信先のベースURL(すべてのAPIの共通の起点)は `https://api.notion.com/v1` です。認証には、手順2でコピーしたトークンを `Authorization: Bearer <token>` という形でヘッダーに含めます。

**守るべき上限がいくつかあります**(超えるとエラーになります):

- 本文を追加するリクエスト1回につき、最大100ブロックまで
- テキスト1要素(`text.content`)は最大2000文字まで
- 配列(ブロックの並びやテキストの並びなど)は最大100要素まで
- リクエスト全体では、最大1000ブロック要素、かつ最大500KBまで
- URLは最大2000文字まで
- 送信できる速さは、1つの連携につき平均で毎秒約3リクエストまで(瞬間的な集中は多少許容されます)

文字起こしのテキストが長い動画では、2000文字ごと・100ブロックごとに区切って、複数回のリクエストに分けて送る必要がある、ということです。

(補足: バージョン2026-03-11では、本文追加のリクエストで以前は `after` というパラメータを使っていたものが `position` というオブジェクトに置き換わりましたが、今回のようにただ末尾に追記するだけなら指定不要です。また同バージョンで、リクエスト・レスポンス中の `archived` という項目名が `in_trash` に変わりましたが、これはページを削除・復元する場合にだけ関係します。)

**Pythonでの実装** — `requests` や `httpx` のような、ふつうのHTTP通信ライブラリで、上のヘッダーを付けて呼び出すだけで十分です。SDK(あらかじめ用意された部品集)を使いたい場合は、コミュニティが保守している `notion-client`(PyPI配布、最新版3.1.0。Notion公式のSDKはJavaScript版のみ)があります。これを使う場合は、初期設定のNotionバージョンに任せず、`2026-03-11` を明示的に指定してください。

### テスト方法

Notionへの実際の書き込みを毎回発生させずに動作確認できるよう、次の2つを用意することをおすすめします。

- **HTTP通信そのものをモック(偽物に差し替えること)する** — テスト実行時は、実際にNotionのサーバーへ接続する代わりに、あらかじめ用意した「こういうリクエストが来たら、こういう返事を返す」という偽のサーバー(またはライブラリの偽装機能)に差し替えます。
- **`--dry-run` フラグを付ける** — 実際にはNotionへ送信せず、「送信するはずだった内容」を画面に表示するだけのモードを用意します。設定ミスの確認や、内容のプレビューに使えます。

### 参考リンク

- Notion公式の開発者向けドキュメント一覧(機械可読): https://developers.notion.com/llms.txt — Notionの全ドキュメントページの一覧が載っており、それぞれのページは `.md` 形式でも読めます。実装するコーディングエージェントは、着手前にここから該当ページを直接読みに行くことをおすすめします。
- 内部連携の作成: https://www.notion.so/profile/integrations

### コーディングエージェントにそのまま貼れる依頼文

以下は、この節の内容をそのままコーディングエージェント(Claude CodeやCodexなど)に貼り付けて実装を頼むための文面です。

```text
AlwaysWhisper の `transcribe` サブコマンドが出力する transcript_raw.srt / transcript_words.json
（または `caption`/`auto` が出力する字幕入り動画と QA レポート）を読み込み、
Notion に1動画につき1ページを作成する notion-push サブコマンド（または独立スクリプト）を実装してください。

制約:
- 認証は環境変数 NOTION_TOKEN（Bearerトークン）と NOTION_DATABASE_ID を使う。
- すべてのリクエストに Notion-Version: 2026-03-11 と Content-Type: application/json を付ける。
- ページは GET /v1/databases/{database_id} で取得した data_sources[0].id を使い、
  POST /v1/pages の parent に {"type": "data_source_id", "data_source_id": "<id>"} を指定して作成する。
- プロパティ案: タイトル=動画ファイル名、収録日=date、長さ(秒)=number、言語=select、動画URL=url(任意)。
- 本文（文字起こし全文）は PATCH /v1/blocks/{page_id}/children で paragraph ブロックとして追加する。
  1リクエストあたり最大100ブロック、rich_text の text.content は最大2000文字、
  リクエスト全体は最大1000ブロック要素かつ500KBまでという上限を守り、超える場合は分割して複数回送信する。
- 429 が返ってきたら Retry-After ヘッダーの秒数だけ待ってリトライする（送信速度の目安は1連携あたり平均毎秒3リクエスト）。
- HTTPをモックしたテストと、実際には送信しない --dry-run フラグを用意する。
- 実装前に https://developers.notion.com/llms.txt から該当ページを確認し、
  ここに書かれていない詳細（エラーハンドリングの形式など）はそちらを一次情報として参照する。
```

## ライセンス

AlwaysWhisperは [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) の下で提供されています — 詳細は [LICENSE](LICENSE) を参照してください。

非商用目的（個人利用・教育・研究・非営利組織での利用）では自由に使用・改変・共有できます。商用利用は許可されていません。商用ライセンスについては support@pmdao.org までご連絡ください。

Required Notice: Copyright (c) 2026 kzkhykw (support@pmdao.org)

かんたんに言うと、趣味で自分の動画に字幕を付けたり、学校の課題や研究に使ったりする分には無料で自由に使えますが、会社の仕事として使ったり、この道具を使ったサービスでお金を稼いだりする場合は、商用ライセンスの契約が別途必要になる、ということです。
