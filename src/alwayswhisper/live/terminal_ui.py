"""Japanese/English text for terminal setup and the live session guide."""
import shlex
from pathlib import Path


# Each entry is (English, Japanese). Recognition language is configured separately.
MESSAGES = {
    "folder_title": ("\nAlwaysWhisper — transcript folder setup", "\nAlwaysWhisper — 保存先の設定"),
    "folder_intro": ("Your words are saved automatically as daily Markdown and session JSONL files.", "文字起こしは日別Markdownとセッション別JSONLに自動保存されます。"),
    "existing": ("Existing transcripts found. Keep this folder to continue using them.", "既存の文字起こしが見つかりました。同じフォルダを使って続けられます。"),
    "suggested": ("\nSuggested folder: {path}", "\n保存先の候補: {path}"),
    "folder_enter": ("Press Enter to use it, or enter another folder path (spaces and ~ are OK).", "Enterでこの場所を使うか、別のフォルダのパスを入力してください（空白・~も使用可）。"),
    "remember_folder": ("This choice is remembered. Existing files will not be moved.", "選んだ場所を記憶します。既存のファイルは移動しません。"),
    "sync_hint": ("If you sync transcripts to Notion, also update the sync job when changing folders.", "Notion同期を使っている場合、保存先を変えたら同期ジョブの参照先も合わせてください。"),
    "change_folder": ("  Change folder: {command} --choose-transcript-dir", "  保存先を変更: {command} --choose-transcript-dir"),
    "folder_input": ("Save folder [Enter = suggested]: ", "保存先 [Enter = 候補の場所]: "),
    "folder_cancelled": ("Folder setup cancelled. No recording was started.", "保存先の設定を中止しました。録音は開始していません。"),
    "folder_invalid": ("Cannot use that folder: {error}\nPlease enter a writable folder.", "このフォルダは使えません: {error}\n書き込み可能なフォルダを入力してください。"),
    "folder_saved": ("\nSaved folder: {path}\n", "\n保存先を記憶しました: {path}\n"),
    "folder_terminal": ("Folder selection requires a terminal. Use --transcript-dir PATH for this run.", "保存先の選択には対話ターミナルが必要です。今回の保存先は --transcript-dir PATH で指定できます。"),
    "folder_error": ("Cannot configure transcript folder: {error}. Use --choose-transcript-dir to select a folder, or --transcript-dir PATH for this run.", "保存先を設定できません: {error}。--choose-transcript-dir で選び直すか、--transcript-dir PATH で今回の保存先を指定してください。"),
    "language_saved": ("Display language: English (remembered).", "表示言語: 日本語（記憶しました）。"),
    "language_change": ("  Switch to Japanese: {command} --ui-language ja (remembered)", "  英語表示に変更: {command} --ui-language en（記憶します）"),
    "title": ("\nAlwaysWhisper — live transcription", "\nAlwaysWhisper — ライブ文字起こし"),
    "microphone": ("  Microphone: {name}", "  マイク: {name}"),
    "recognition_language": ("  Recognition language: {language} (independent of display language)", "  認識言語: {language}（表示言語とは別の設定です）"),
    "save_folder": ("  Save folder: {path}", "  保存先: {path}"),
    "read_later": ("  Read later: YYYY-MM-DD.md (daily notes); live_*.jsonl (session data)", "  読み返す: YYYY-MM-DD.md（日別ノート） / live_*.jsonl（セッションデータ）"),
    "open_folder": ("  Open folder: open {path}", "  Finderで開く: open {path}"),
    "prompt_title": ("\nRecognition vocabulary / custom prompt", "\n認識用カスタムプロンプト（固有名詞・用語のヒント）"),
    "prompt_default": ("  Current: bundled glossary (kept unless you override it).", "  現在: デフォルトのグロッサリーを使用（変更しなければそのまま継続）"),
    "prompt_custom": ("  Current: custom prompt.", "  現在: 指定されたカスタムプロンプトを使用"),
    "prompt_none": ("  Current default: no custom prompt.", "  現在: カスタムプロンプトなし"),
    "prompt_file": ("  Using: {path}", "  ファイル: {path}"),
    "prompt_edit": ("  Edit this UTF-8 file to customize names and terms for your next run.", "  このUTF-8ファイルに人名・製品名を追記すると、次回起動から反映されます。"),
    "prompt_preview": ("  Preview: {preview} ({count} characters)", "  内容: {preview} ({count}文字)"),
    "prompt_customize": ('  Customize: {command} {option} "/path/to/terms.txt"', '  別ファイルを使う: {command} {option} "/path/to/terms.txt"'),
    "prompt_inline": ('  Inline prompt: {command} {option} "Notion, Claude"', '  直接指定する: {command} {option} "Notion, Claude"'),
    "prompt_hint": ("  Add names and specialist terms you use; this is a recognition hint, not a chat instruction.", "  普段使う人名・専門用語を登録してください。これは認識用のヒントで、チャットへの指示文とは異なります。"),
    "prompt_override": ("  An explicit prompt replaces the bundled default.", "  指定したプロンプトはデフォルトに代わって使われます。"),
    "model": ("\nModel: {model}", "\nモデル: {model}"),
    "microphone_start": ("Starting microphone. Allow microphone access if macOS asks.", "マイクを起動します。アクセスを求められたら許可してください。"),
    "model_load": ("The model loads on your first speech; if uncached, it downloads automatically.", "最初の発話でモデルを読み込みます。未取得の場合は自動ダウンロードで時間がかかります。"),
    "speak": ("Speak normally, then pause briefly. Text appears below and saves automatically.", "話して少し間を置くと、認識結果が表示されて自動保存されます。"),
    "stop": ("Stop: Ctrl-C (wait for pending transcription to finish).\n", "停止: Ctrl-C（残りの文字起こしが終わるまでお待ちください）。\n"),
    "listening": ("Microphone: {name} — receiving audio.", "マイク: {name} — 音声を受け付けています。"),
    "finishing": ("\nFinishing pending transcription…", "\n残りの文字起こしを保存しています…"),
    "notes": ("Daily notes (Markdown): {path}", "日別ノート（Markdown）: {path}"),
    "session": ("Session data (JSONL): {path}", "セッションデータ（JSONL）: {path}"),
    "dropped": ("Audio chunks dropped while processing fell behind: {count}", "処理が追いつかず破棄した音声チャンク: {count}"),
}


def message(ui_language, key, **values):
    return MESSAGES[key][1 if ui_language == "ja" else 0].format(**values)


def print_session_guide(language, *, directory, command, recognition_language,
                        prompt=None, prompt_file=None, bundled_prompt=False,
                        glossary_option="--glossary", inline_option=None,
                        microphone=None, model=None):
    def say(key, **values):
        print(message(language, key, **values))
    say("title")
    if microphone is not None:
        say("microphone", name=microphone)
    say("recognition_language", language=recognition_language)
    say("save_folder", path=directory)
    say("read_later")
    say("open_folder", path=shlex.quote(str(directory)))
    say("change_folder", command=command)
    say("language_change", command=command)
    say("prompt_title")
    say("prompt_default" if bundled_prompt else "prompt_custom" if prompt else "prompt_none")
    if prompt_file:
        say("prompt_file", path=Path(prompt_file).expanduser().resolve())
        say("prompt_edit")
    if prompt:
        preview = prompt.replace("\n", " ")
        say("prompt_preview", preview=preview[:80] + ("…" if len(preview) > 80 else ""), count=len(prompt))
    say("prompt_customize", command=command, option=glossary_option)
    if inline_option:
        say("prompt_inline", command=command, option=inline_option)
        say("prompt_override")
    say("prompt_hint")
    if model:
        say("model", model=model)
    say("microphone_start")
    say("model_load")
    say("speak")
    say("stop")
    # Finish the guide before the worker can print recognition output.
    print(end="", flush=True)
