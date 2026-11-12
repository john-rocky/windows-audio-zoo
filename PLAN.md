# PLAN — 合否の線と測り方(測る前に書いた: 2026-11-09 09:58 JST、この lane の数はまだ 1 つも無い)

(2026-11-10、公開の前に: 他の作業の名、判断の番号、private の path と、この repo に無い file への指しを、中立の語かこの repo の file に置き換えた。線・数・手順は測る前のまま。)

## 1. 合否の線(cell ごと、測る前に固定)
- **ASR(S150)**: 言語ごとに WER(en)/ CER(zh / ja)≤ 同じ clip・同じ prompt の参照 + 0.5 pt、かつ空の答え 0(この PC の先の Qwen3-ASR の作業と同じ線の形)。参照 = Whisper と Parakeet はこの lane が PC の CPU で回す transformers fp32、Qwen3-ASR は先の Qwen3-ASR の作業の official(`raw/refs/qwen3asr-fp32/`)、Granite は先の Granite Speech の作業の R(`raw/refs/granite-fp32/`)。Parakeet は en だけ。
  - Qwen3-ASR の線の数 = official: en 4.848 / zh 7.358 / ja 6.004(先の Qwen3-ASR の作業と同じ)。Whisper と Parakeet の数は参照の run の後に、表の前にこの節へ書き足す(参照の run は cell の run より前)。
  - Whisper large-v3-turbo の線の数 = en 5.630 / zh 9.610 / ja 4.330(参照 = この PC の CPU の transformers 5.22.2・torch 2.14.0 fp32、greedy、言語は clip ごとに指定、誤り 59 / 1150 語・178 / 1954 字・103 / 2689 字、空 0、`raw/refs/whisper/ref.json`、11-09 13:38 に済み)。
  - Distil-Whisper distil-large-v3 の線の数 = en 7.022(英語の model = S150 の en_us 50 clip だけ。参照 = この PC の CPU の transformers 5.22.2・torch 2.14.0 fp32、greedy、language en、誤り 75 / 1150 語 = 6.522 %、空 0、`raw/refs/distil-large-v3/ref.json`、11-12 18:45 に済み。線の形は参照の run(11-12 18:40 開始)の前に固定)。
  - Parakeet TDT 0.6B v3 の線の数 = en 5.630(参照 = この PC の CPU の transformers 5.22.2 `ParakeetForTDT` fp32、greedy、誤り 59 / 1150 語、空 0、`raw/refs/parakeet/ref.json`、11-09 13:44 に済み。誤りの合計は Whisper の en と同じ 59 だが clip ごとの数は 50 中 24 clip で違う = 偶然の一致)。
- **ASR(S220、Granite)**: 先の Granite Speech の作業の線 = R + 0.5 pt(en-FL 5.963 / ja 5.632 / fr+de+es+pt 12.603)、空・loop・EOS なし 0。en-LS は計器。
- **TTS(T50、Kokoro)**: 書き戻しの WER ≤ 参照(PyTorch の kokoro の音声を同じ ASR で書き戻した WER)+ 1.0 pt、かつ音声の無い答え(長さ 0 か 0.2 s 未満か全部が無音)0。+1.0 にした理由 = 書き戻しは ASR 自身の揺れを足すので ASR の線(0.5)より 1 段広く、T50 の約 1,000 語で 10 語。書き戻しの ASR = OpenVINO GenAI の ASRPipeline、whisper-large-v3-turbo-fp16-ov、CPU、language `<|en|>` 固定(全 TTS cell で同じ、計時しない)。
- 計器(線ではない): 参照と token が同じ clip の数(先の Qwen3-ASR の作業の token-exact、取れる runtime だけ)、first pass と計時の pass の文が同じか、zh の答えの繁体字の clip 数(Whisper)、split-half の RTF の差(奇数 clip と偶数 clip)。
- 落ちた cell(load・compile・生成で落ちる)は error の全文、runtime と EP の版、command を JSON に(`status` = `error`)。probe の cell が通れば、その cell は「測る」に上がり、線で読む。

## 2. workload(runtime をまたいで同じにする物)
- 音声 = WAV を PCM16 → float32 [-1, 1] で読み、16 kHz mono のまま渡す(読みは計時の外)。runtime が file の path しか取らない時(LiteRT-LM の `AudioFile`)は path を渡し、読みも計時に入る = cell の JSON に書く。
- decode = greedy(sampling なし、beam 1)。最大の新 token: Whisper 440(decoder の位置 448 − prompt 4 − 余白)、Qwen3-ASR 512(先の Qwen3-ASR の作業と同じ)、Granite 256(先の Granite Speech の作業と同じ)。
- prompt: Whisper = `<|startoftranscript|><|xx|><|transcribe|><|notimestamps|>`(xx = clip の言語、timestamps なし)。Qwen3-ASR = official(language を与えない = 答えの頭の `language X<asr_text>` を runtime が切る、切らない runtime は先の Qwen3-ASR の作業の `split_raw` と同じ規則で切る)。Granite = 先の Granite Speech の作業の prompt。Parakeet = prompt なし。
- Kokoro: 入力 = T50 の文(`raw_transcription` そのまま)、voice `af_heart`、言語 en-us、speed 1.0。G2P は各 runtime の既定の道(OpenVINO GenAI = 内蔵の辞書 + fallback、kokoro-onnx = 自分の phonemizer)= JSON に書く。runtime をまたいで phoneme を揃えない(使う人が得る text → 音声 → text の通しを測る)。
- 1 cell = 1 process: model を load → 暖め 1 本(ASR = `examples/en.wav`、TTS = 固定の 1 文、採点しない)→ 組の全部を順に 1 回。run の数え方 = 先の Qwen3-ASR の作業と同じ 1 pass(その数を同じ表に並べるため)。

## 3. 指標(cell の JSON の `table`)
- **RTF** = Σ(clip ごとの処理の wall)/ Σ(clip の秒)。処理の wall = 入力(float の配列か path)を渡してから最後の文を得るまで。model の load と暖めは入れない。TTS は Σ(合成の wall)/ Σ(出た音声の秒)。
- **first token**(ASR、自己回帰の model だけ): clip ごとの「入力を渡してから最初の text の token まで」の平均。OpenVINO GenAI = `perf_metrics.get_ttft()`、ORT GenAI = `set_inputs` + 最初の `generate_next_token`、LiteRT-LM = streaming の最初の chunk(取れなければ n/a と書く)。Parakeet(TDT)= n/a。
- **peak memory** = engine の process の PeakWorkingSet(chain が 0.5 s ごとに読む = windows-llm-zoo の形)、GPU の memory = `\GPU Process Memory(pid_*)` の Dedicated / Shared の最大(adapter の LUID ごと)。
- **load s** = model の load(と compile)の wall、`cache_dir` の有無(OpenVINO の GPU / NPU の compile cache は cell ごとに新しい folder = 冷たい compile を計る、2 回目以降は計器)。
- WER / CER は計時の pass の文で採点(first pass の文と同じかは計器)。

## 4. sitting(計時)と静かさ
- 計時の sitting は `%USERPROFILE%\pcup\busy.txt` を CreateNew で取る(label `Z3`)。取った後の門 = windows-llm-zoo の Pass-LaneGate(5 s の CPU 3 回続けて 15 % 未満、他の process の RTX memory 50 MiB 未満、build の道具 200 ms 未満、上限 10 分)。run ごとの門 = windows-llm-zoo の Wait-RunQuiet(CPU 25 % 未満 3 読み、NVIDIA 800 MiB 未満、igpu の cell は GPU engine の和 10 % 未満、空き RAM ≥ cell の必要量、RAM が 10 分続けて足りなければ sitting を止めて上位 3 の持ち主を log に)。
- 鍵の duty(PC の列の規則): 他 lane が待つ間(`active\` か `yield\` の *.txt、年齢で切らない)、run の前に「鍵を持って 30 分経った」か「その run の終わりが鍵の 30 分を越える」なら 16.5 分外す(取り直しの直後の 1 run は長さに依らず始める)。外す間は `pcup\window.txt` に「Z3 open until <ISO>」、窓の時刻が終わったら消す。他 lane が返した後は 60 s 空いてから取り直し、60 s の settle と門。鍵の文に今の run の見込みの終わり(「run until about HH:mm」)。時刻の決まった他 lane の計時には window.txt を書かず、run の間で止めて時刻に外す(windows-llm-zoo の `slots.ps1` の形)。
- **対照の行**: sitting の頭と終わりに、暖めの 1 run(数えない)の後、Whisper int8 の OpenVINO GenAI の CPU で S150 の各言語の最初の 10 clip(30 clip)の RTF。既知値 = この lane の最初の sitting の頭の値(その sitting では頭と終わりが 5 % 以内)、2 回目以降の sitting は頭が既知値(同じ load の組 = 新しい process の cache なし)から 5 % を越えたら他の行を回さない。
- 計時の間は自分の untimed の工程を走らせず、suspend で RAM を持たせない(windows-llm-zoo の lock.ps1 take の拒否を写す)。
- 静かさの要らない工程(download、venv、export、参照の run、first pass)は `pcup\active\Z3-<列>.txt` に「何を・いつから」、`pc\steps.ps1` か chain の first mode。他 lane の鍵が立ったら kill して、鍵が 60 s 続けて空いてから頭からやり直す(11-09 13:00 に suspend をやめた: suspend した process は RAM を返さず他 lane の計時の門を止めた)。8 GB を超える step は PC の列の枠で Z3 の「(not a timing)」の鍵の中で走らせる。鍵なしで並べる非計時の仕事は同時の RAM の合計 16 GB 以下(PC の列の規則)。

- **11-09 20:4x の書き足し(s4 の数の前、元の文は上のまま)**: (1) 20:12 に PC の MSI Center が Extreme Performance に固定された(registry `User Scenario` `Mode` 2 → 1)= s1 の頭(既知値 0.26535)は前の mode。夜の s4 は既知値を s1 から取らず、自分の頭 3 本(広がり 5 % 以内)から決め、終わりの行をその値と比べる(5 % の線は同じ)。頭が線で止めたら記録を別名に移して 1 回だけ頭から取り直す。s1 の 5 cell も s4 で測り直す = 表の計時の行は全部 Extreme の下の s4 から、s1 は記録(その tail は Extreme の後に 0.25861 = 頭から −2.5 %)。(2) duty の「年齢で切らない」を変えた: `active\` の札は 120 分以内に書かれた物だけを待つ lane と数える(列の規則 = 書き直されない札は 2 時間で消える、`yield\` は在る間)。(3) control-tail の前に duty の窓を開けない、end_by の前の取り直しは空き 1 分・settle・門・tail が end_by に入る時だけ。
- **11-09 22:4x の書き足し(s4 の数の前)**: 23:10 に他 lane が MSI の Mode を 2(Balanced)に戻す試し(Extreme の下では他 lane の門の CPU % が process の負荷なしに上がる)= s4 の mode は試しの結果次第。どちらでも s4 は自分の頭 3 本で既知の値を決め(戻った後の状態が s1 と同じとは確かめていない)、s1 の 5 cell も s4 で測る。run ごとの meta に `power`(MSI の Mode、Windows の電源 overlay の AC / DC、電池の状態)と `desktopLocked`。

## 5. 置き場
- PC: `C:\src\audiozoo\`(venv、model、runs、logs)= この lane の物だけ。他 lane の folder は読むだけ。
- Mac: `raw/<sitting>/`(PC から 1 接続の tar)、`results/<model>/<cell>.json`(`tools/build_results.py`。先の作業から集めた cell は local の表だけ)、表 = `tools/table.py` → `RESULTS_AUDIO_WINDOWS.md`。record は commit の前に account 名を USER に(`tools/scrub.py`)。

## 6. 公開
- runtime をまたぐ表は git に入れない(`tools/table.py` が手元で書く。edge-llm-bench、windows-llm-zoo と同じ)。TensorRT-RTX と TensorRT-Edge-LLM は測らない・書かない、`NvTensorRtRtx` EP は使わない。
