"""Tests for alwayswhisper.cli: argument parsing, override routing into cfg,
and clean (no-traceback) error handling for user-facing failures.

Pipeline functions are monkeypatched on alwayswhisper.cli's namespace for most
tests, so no real ffmpeg/Whisper/MoviePy call happens; these tests exercise
argument parsing and override wiring, not pipeline behavior itself (that's
covered by test_pipeline_*.py). The "unknown backend" test is a deliberate
exception -- it lets the real transcribe_file/create_backend run, since
create_backend's ValueError fires before any real work happens.
"""

import sys
from types import SimpleNamespace

import pytest

import alwayswhisper.cli as cli


def _input_file(tmp_path, name="input.mp4"):
    path = tmp_path / name
    path.write_bytes(b"fake")
    return path


def _srt_file(tmp_path, name="captions.srt"):
    path = tmp_path / name
    path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
    return path


class TestTranscribeOverrides:
    def test_flags_route_into_cfg(self, tmp_path, monkeypatch):
        input_file = _input_file(tmp_path)
        captured = {}

        def fake_transcribe_file(input_path, workdir, cfg, glossary_path=None):
            captured["cfg"] = cfg
            return {"words_path": "w.json", "srt_path": "s.srt", "word_count": 1, "entry_count": 1}

        monkeypatch.setattr(cli, "transcribe_file", fake_transcribe_file)

        cli.main(["transcribe", str(input_file), "--model", "small", "--language", "en"])

        assert captured["cfg"]["transcribe"]["model"] == "small"
        assert captured["cfg"]["transcribe"]["language"] == "en"
        # Untouched keys still come from the packaged defaults.
        assert captured["cfg"]["transcribe"]["backend"] == "faster-whisper"

    def test_unset_flags_do_not_override_packaged_defaults(self, tmp_path, monkeypatch):
        input_file = _input_file(tmp_path)
        captured = {}

        def fake_transcribe_file(input_path, workdir, cfg, glossary_path=None):
            captured["cfg"] = cfg
            return {"words_path": "w.json", "srt_path": "s.srt", "word_count": 0, "entry_count": 0}

        monkeypatch.setattr(cli, "transcribe_file", fake_transcribe_file)

        cli.main(["transcribe", str(input_file)])

        assert captured["cfg"]["transcribe"]["language"] == "ja"  # packaged default

    def test_default_output_dir_is_input_stem_alwayswhisper(self, tmp_path, monkeypatch):
        input_file = _input_file(tmp_path, name="myvideo.mp4")
        captured = {}

        def fake_transcribe_file(input_path, workdir, cfg, glossary_path=None):
            captured["workdir"] = workdir
            return {"words_path": "w.json", "srt_path": "s.srt", "word_count": 0, "entry_count": 0}

        monkeypatch.setattr(cli, "transcribe_file", fake_transcribe_file)

        cli.main(["transcribe", str(input_file)])

        assert captured["workdir"] == tmp_path / "myvideo_alwayswhisper"

    def test_explicit_output_dir_is_used(self, tmp_path, monkeypatch):
        input_file = _input_file(tmp_path)
        captured = {}

        def fake_transcribe_file(input_path, workdir, cfg, glossary_path=None):
            captured["workdir"] = workdir
            return {"words_path": "w.json", "srt_path": "s.srt", "word_count": 0, "entry_count": 0}

        monkeypatch.setattr(cli, "transcribe_file", fake_transcribe_file)

        custom_dir = tmp_path / "custom_out"
        cli.main(["transcribe", str(input_file), "-o", str(custom_dir)])

        assert captured["workdir"] == custom_dir

    def test_glossary_flag_forwarded_as_path_not_cfg(self, tmp_path, monkeypatch):
        input_file = _input_file(tmp_path)
        glossary = tmp_path / "glossary.txt"
        glossary.write_text("term1\nterm2", encoding="utf-8")
        captured = {}

        def fake_transcribe_file(input_path, workdir, cfg, glossary_path=None):
            captured["glossary_path"] = glossary_path
            return {"words_path": "w.json", "srt_path": "s.srt", "word_count": 0, "entry_count": 0}

        monkeypatch.setattr(cli, "transcribe_file", fake_transcribe_file)

        cli.main(["transcribe", str(input_file), "--glossary", str(glossary)])

        assert captured["glossary_path"] == str(glossary)

    def test_max_chars_and_min_chars_route_to_srt_cfg(self, tmp_path, monkeypatch):
        input_file = _input_file(tmp_path)
        captured = {}

        def fake_transcribe_file(input_path, workdir, cfg, glossary_path=None):
            captured["cfg"] = cfg
            return {"words_path": "w.json", "srt_path": "s.srt", "word_count": 0, "entry_count": 0}

        monkeypatch.setattr(cli, "transcribe_file", fake_transcribe_file)

        cli.main(["transcribe", str(input_file), "--max-chars", "21", "--min-chars", "3"])

        assert captured["cfg"]["srt"]["max_chars"] == 21
        assert captured["cfg"]["srt"]["min_chars"] == 3

    def test_missing_input_file_exits_1_with_clean_message(self, tmp_path, capsys):
        missing = tmp_path / "does_not_exist.mp4"

        with pytest.raises(SystemExit) as exc_info:
            cli.main(["transcribe", str(missing)])

        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "Error:" in err
        assert "Traceback" not in err

    def test_unknown_backend_exits_1_with_clean_message(self, tmp_path, capsys):
        input_file = _input_file(tmp_path)

        # transcribe_file is deliberately NOT monkeypatched: create_backend
        # raises ValueError for an unknown name before any real work
        # (audio extraction, model load) happens, so this exercises the
        # real error path end-to-end without needing ffmpeg/Whisper.
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["transcribe", str(input_file), "--backend", "nonsense"])

        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "Error:" in err
        assert "nonsense" in err
        assert "Traceback" not in err


class TestSegmentCommand:
    def test_flags_route_into_cfg(self, tmp_path, monkeypatch):
        words_json = tmp_path / "transcript_words.json"
        words_json.write_text("[]", encoding="utf-8")
        captured = {}

        def fake_segment_words(words_json_arg, out_srt, cfg):
            captured.update(words_json=words_json_arg, out_srt=out_srt, cfg=cfg)
            return out_srt

        monkeypatch.setattr(cli, "segment_words", fake_segment_words)

        out_srt = tmp_path / "out.srt"
        cli.main([
            "segment", str(words_json), "-o", str(out_srt),
            "--language", "en", "--max-chars", "40",
        ])

        assert captured["cfg"]["transcribe"]["language"] == "en"
        assert captured["cfg"]["srt"]["max_chars"] == 40
        assert str(captured["out_srt"]) == str(out_srt)

    def test_missing_words_json_exits_1(self, tmp_path, capsys):
        missing = tmp_path / "nope.json"

        with pytest.raises(SystemExit) as exc_info:
            cli.main(["segment", str(missing), "-o", str(tmp_path / "out.srt")])

        assert exc_info.value.code == 1
        assert "Error:" in capsys.readouterr().err


class TestCaptionOverrides:
    def test_no_qa_sets_qa_enabled_false(self, tmp_path, monkeypatch):
        video = _input_file(tmp_path, "video.mp4")
        srt = _srt_file(tmp_path)
        captured = {}

        def fake_caption_video(video_arg, srt_arg, out_path, cfg, words_json=None, edit_plan_json=None):
            captured["cfg"] = cfg
            return {"out_path": out_path, "entry_count": 1}

        monkeypatch.setattr(cli, "caption_video", fake_caption_video)

        cli.main(["caption", str(video), str(srt), "-o", str(tmp_path / "out.mp4"), "--no-qa"])

        assert captured["cfg"]["qa"]["enabled"] is False

    def test_qa_enabled_left_at_default_when_no_qa_not_passed(self, tmp_path, monkeypatch):
        video = _input_file(tmp_path, "video.mp4")
        srt = _srt_file(tmp_path)
        captured = {}

        def fake_caption_video(video_arg, srt_arg, out_path, cfg, words_json=None, edit_plan_json=None):
            captured["cfg"] = cfg
            return {"out_path": out_path, "entry_count": 1}

        monkeypatch.setattr(cli, "caption_video", fake_caption_video)

        cli.main(["caption", str(video), str(srt), "-o", str(tmp_path / "out.mp4")])

        assert captured["cfg"]["qa"]["enabled"] is True  # packaged default, untouched

    def test_fast_and_realign_flags_and_words_path_forwarded(self, tmp_path, monkeypatch):
        video = _input_file(tmp_path, "video.mp4")
        srt = _srt_file(tmp_path)
        words = tmp_path / "transcript_words.json"
        words.write_text("[]", encoding="utf-8")
        captured = {}

        def fake_caption_video(video_arg, srt_arg, out_path, cfg, words_json=None, edit_plan_json=None):
            captured.update(cfg=cfg, words_json=words_json)
            return {"out_path": out_path, "entry_count": 0}

        monkeypatch.setattr(cli, "caption_video", fake_caption_video)

        cli.main([
            "caption", str(video), str(srt), "-o", str(tmp_path / "out.mp4"),
            "--fast", "--realign", "--words", str(words),
        ])

        assert captured["cfg"]["caption"]["fast_mode"] is True
        assert captured["cfg"]["caption"]["realign"] is True
        assert str(captured["words_json"]) == str(words)

    def test_missing_video_exits_1(self, tmp_path, capsys):
        missing_video = tmp_path / "nope.mp4"
        srt = _srt_file(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            cli.main(["caption", str(missing_video), str(srt), "-o", str(tmp_path / "out.mp4")])

        assert exc_info.value.code == 1
        assert "Error:" in capsys.readouterr().err

    def test_missing_srt_exits_1(self, tmp_path, capsys):
        video = _input_file(tmp_path, "video.mp4")
        missing_srt = tmp_path / "nope.srt"

        with pytest.raises(SystemExit) as exc_info:
            cli.main(["caption", str(video), str(missing_srt), "-o", str(tmp_path / "out.mp4")])

        assert exc_info.value.code == 1
        assert "Error:" in capsys.readouterr().err


class TestQaCommand:
    def test_exit_code_1_on_failed_report(self, tmp_path, monkeypatch, capsys):
        video = _input_file(tmp_path, "video.mp4")
        srt = _srt_file(tmp_path)

        fail_report = {
            "passed": False, "avg_ratio": 0.1, "min_ratio": 0.5,
            "sampled": 1, "eligible": 1,
            "samples": [{
                "index": 1, "start": "00:00:00,000", "end": "00:00:01,000",
                "srt_text": "hi", "asr_text": "bye", "ratio": 0.1, "ok": False,
            }],
        }

        def fake_qa_check(video_arg, srt_arg, cfg):
            return {"report": fail_report, "report_path": tmp_path / "qa_report.json"}

        monkeypatch.setattr(cli, "qa_check", fake_qa_check)

        with pytest.raises(SystemExit) as exc_info:
            cli.main(["qa", str(video), str(srt)])

        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "QA FAILED" in out
        assert "#1" in out

    def test_exit_code_0_on_passed_report(self, tmp_path, monkeypatch, capsys):
        video = _input_file(tmp_path, "video.mp4")
        srt = _srt_file(tmp_path)

        pass_report = {
            "passed": True, "avg_ratio": 1.0, "min_ratio": 0.5,
            "sampled": 0, "eligible": 0, "samples": [],
        }

        def fake_qa_check(video_arg, srt_arg, cfg):
            return {"report": pass_report, "report_path": tmp_path / "qa_report.json"}

        monkeypatch.setattr(cli, "qa_check", fake_qa_check)

        cli.main(["qa", str(video), str(srt)])  # must not raise SystemExit

        assert "QA PASSED" in capsys.readouterr().out

    def test_samples_and_min_ratio_routed_into_cfg(self, tmp_path, monkeypatch):
        video = _input_file(tmp_path, "video.mp4")
        srt = _srt_file(tmp_path)
        captured = {}

        def fake_qa_check(video_arg, srt_arg, cfg):
            captured["cfg"] = cfg
            return {
                "report": {
                    "passed": True, "avg_ratio": 1.0, "min_ratio": 0.9,
                    "sampled": 0, "eligible": 0, "samples": [],
                },
                "report_path": tmp_path / "qa_report.json",
            }

        monkeypatch.setattr(cli, "qa_check", fake_qa_check)

        cli.main(["qa", str(video), str(srt), "--samples", "2", "--min-ratio", "0.9"])

        assert captured["cfg"]["qa"]["samples"] == 2
        assert captured["cfg"]["qa"]["min_ratio"] == 0.9


class TestAutoCommand:
    def test_flags_route_into_cfg_and_paths_forwarded(self, tmp_path, monkeypatch):
        input_file = _input_file(tmp_path, "input.mp4")
        captured = {}

        def fake_auto_run(input_path, out_path, cfg, workdir=None, glossary_path=None):
            captured.update(cfg=cfg, workdir=workdir, glossary_path=glossary_path)
            return {
                "caption": {"out_path": out_path, "entry_count": 3},
                "srt_path": tmp_path / "out.srt",
            }

        monkeypatch.setattr(cli, "auto_run", fake_auto_run)

        cli.main([
            "auto", str(input_file), "-o", str(tmp_path / "out.mp4"),
            "--model", "small", "--no-qa", "--workdir", str(tmp_path / "work"),
        ])

        assert captured["cfg"]["transcribe"]["model"] == "small"
        assert captured["cfg"]["qa"]["enabled"] is False
        assert captured["workdir"] == str(tmp_path / "work")

    def test_device_and_compute_type_route_into_cfg(self, tmp_path, monkeypatch):
        input_file = _input_file(tmp_path, "input.mp4")
        captured = {}

        def fake_auto_run(input_path, out_path, cfg, workdir=None, glossary_path=None):
            captured["cfg"] = cfg
            return {
                "caption": {"out_path": out_path, "entry_count": 3},
                "srt_path": tmp_path / "out.srt",
            }

        monkeypatch.setattr(cli, "auto_run", fake_auto_run)

        cli.main([
            "auto", str(input_file), "-o", str(tmp_path / "out.mp4"),
            "--device", "cpu", "--compute-type", "int8",
        ])

        assert captured["cfg"]["transcribe"]["device"] == "cpu"
        assert captured["cfg"]["transcribe"]["compute_type"] == "int8"

    def test_missing_input_exits_1(self, tmp_path, capsys):
        missing = tmp_path / "nope.mp4"

        with pytest.raises(SystemExit) as exc_info:
            cli.main(["auto", str(missing), "-o", str(tmp_path / "out.mp4")])

        assert exc_info.value.code == 1
        assert "Error:" in capsys.readouterr().err


class TestPrefetchCommand:
    def test_help_does_not_download(self):
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["prefetch", "--help"])

        assert exc_info.value.code == 0

    def test_prefetch_calls_download_model_with_requested_model(self, monkeypatch, capsys):
        calls = []
        monkeypatch.setitem(
            sys.modules, "faster_whisper",
            SimpleNamespace(
                download_model=lambda model: calls.append(model) or "/fake/cache/hub/tiny"
            ),
        )

        cli.main(["prefetch", "--model", "tiny"])

        assert calls == ["tiny"]
        out = capsys.readouterr().out
        assert "tiny" in out
        # download_model()'s return value is the real destination (honors
        # HF_HOME/HF_HUB_CACHE) -- it must be what's printed, not a
        # hardcoded ~/.cache/huggingface/hub guess.
        assert "/fake/cache/hub/tiny" in out

    def test_prefetch_default_model_is_large_v3(self, monkeypatch):
        calls = []
        monkeypatch.setitem(
            sys.modules, "faster_whisper",
            SimpleNamespace(
                download_model=lambda model: calls.append(model) or "/fake/cache/hub/large-v3"
            ),
        )

        cli.main(["prefetch"])

        assert calls == ["large-v3"]


class TestVersionFlag:
    def test_version_flag_exits_0_and_prints_version(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["--version"])

        assert exc_info.value.code == 0
        assert "alwayswhisper" in capsys.readouterr().out
