import json
import math

import pytest

import document_kv_cache.benchmark_runner as public_benchmark_runner
import restaurant_kv_serving.benchmark_runner as legacy_benchmark_runner
from document_kv_cache.benchmark_runner import (
    BENCHMARK_RUN_RECORD_TYPE,
    BenchmarkEngineRequest,
    BenchmarkGeneration,
    OpenAICompatibleBenchmarkConfig,
    benchmark_run_result_to_record,
    default_benchmark_arms,
    load_benchmark_jsonl,
    load_v1_jsonl_suite,
    run_benchmark_suite,
    run_openai_compatible_v1_benchmark,
)
from document_kv_cache.benchmarks import (
    BASELINE_PREFILL_ARM,
    CACHE_REUSE_ARM,
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_HANDOFF_RECORD_PARAM,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
    BenchmarkArm,
    BenchmarkExample,
    BenchmarkSuite,
)
from document_kv_cache.engine import EngineReadyRequest
from document_kv_cache.engine_adapters import (
    build_engine_adapter_request,
    engine_adapter_request_to_record,
    vllm_adapter_spec,
)
from document_kv_cache.engine_protocol import KVCacheHandle, KVLayout, KVSegment
from document_kv_cache.workflow import SourceDocument


class RecordingEngine:
    def __init__(self, *, output: str = "Ada Lovelace", fail: bool = False) -> None:
        self.output = output
        self.fail = fail
        self.requests: list[BenchmarkEngineRequest] = []

    def generate(self, request: BenchmarkEngineRequest) -> BenchmarkGeneration:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("engine unavailable")
        return BenchmarkGeneration(
            output_text=self.output,
            prompt_tokens=len(request.prompt_text.split()),
            completion_tokens=len(self.output.split()),
            ttft_seconds=1.0 if request.arm.uses_cache else 4.0,
            time_to_completion_seconds=3.0 if request.arm.uses_cache else 8.0,
            metadata={"arm": request.arm.arm_id},
        )


class EmptyMessageFailureEngine:
    def generate(self, request: BenchmarkEngineRequest) -> BenchmarkGeneration:
        raise TimeoutError()


class InvalidGenerationEngine:
    def __init__(self, **generation_overrides) -> None:
        self.generation_overrides = generation_overrides

    def generate(self, request: BenchmarkEngineRequest) -> BenchmarkGeneration:
        kwargs = {
            "output_text": "Ada Lovelace",
            "prompt_tokens": len(request.prompt_text.split()),
            "completion_tokens": 2,
            "ttft_seconds": 1.0,
            "time_to_completion_seconds": 2.0,
            "metadata": {"arm": request.arm.arm_id},
        }
        kwargs.update(self.generation_overrides)
        return BenchmarkGeneration(**kwargs)


def example(
    dataset: str = "biography",
    *,
    example_id: str | None = None,
    kv_transfer_params=None,
) -> BenchmarkExample:
    return BenchmarkExample(
        example_id=example_id or f"{dataset}-1",
        dataset=dataset,
        documents=(
            SourceDocument.from_texts(
                document_id="doc-1",
                static_text="Ada Lovelace biography",
                chunks={"p1": "Lovelace wrote notes on the Analytical Engine."},
            ),
        ),
        query="Who wrote notes on the Analytical Engine?",
        expected_answer="Ada Lovelace",
        kv_transfer_params={} if kv_transfer_params is None else kv_transfer_params,
    )


def inline_handoff_record(*, request_id: str = "cachet-bio-1", payload_uri: str | None = None):
    layout = KVLayout(
        model_id="tiny-test-model",
        lora_id="base",
        layout_version="standard-v1",
        dtype="int8",
        num_layers=1,
        block_size=2,
        bytes_per_token=4,
    )
    handle = KVCacheHandle(
        request_id=request_id,
        handle_uri=f"document-kv://{request_id}",
        layout=layout,
        segments=(KVSegment("doc-1", "document_static", "static", 0, 1, 0, 4),),
        total_tokens=1,
        total_bytes=4,
    )
    ready = EngineReadyRequest(handle=handle, payload=b"data", estimated_gpu_bytes=4)
    adapter_request = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    return engine_adapter_request_to_record(adapter_request, payload_uri=payload_uri or f"disk:/tmp/{request_id}.kv")


def test_run_benchmark_suite_records_baseline_and_cache_measurements():
    suite = BenchmarkSuite(suite_id="v1-smoke", examples=(example(),))
    baseline = RecordingEngine()
    cache = RecordingEngine()

    result = run_benchmark_suite(
        suite,
        {
            BASELINE_PREFILL_ARM: baseline,
            CACHE_REUSE_ARM: cache,
        },
    )

    assert [measurement.arm_id for measurement in result.measurements] == [
        BASELINE_PREFILL_ARM,
        CACHE_REUSE_ARM,
    ]
    assert result.report_rows[0].requests == 1
    assert result.comparisons[0].ttft_speedup == pytest.approx(4.0)
    assert baseline.requests[0].logical_prompt_text == cache.requests[0].logical_prompt_text
    assert baseline.requests[0].prompt_text == baseline.requests[0].logical_prompt_text
    assert cache.requests[0].prompt_text == cache.requests[0].cache_suffix_text
    assert cache.requests[0].cache_prefix_text + cache.requests[0].cache_suffix_text == baseline.requests[0].logical_prompt_text
    assert cache.requests[0].model_id == "qwen3:4b-instruct"
    assert cache.requests[0].hardware_target == "aws-g6-l4"


def test_run_benchmark_suite_attaches_kv_transfer_params_to_cache_arm_only():
    kv_transfer_params = {
        DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-bio-1",
        DOCUMENT_KV_HANDOFF_JSON_PARAM: "/Volumes/catalog/schema/volume/cachet/bio-1.handoff.json",
        DOCUMENT_KV_PAYLOAD_URI_PARAM: "uc-volume:/catalog/schema/volume/cachet/bio-1.kv",
    }
    suite = BenchmarkSuite(suite_id="v1-smoke", examples=(example(kv_transfer_params=kv_transfer_params),))
    baseline = RecordingEngine()
    cache = RecordingEngine()

    run_benchmark_suite(
        suite,
        {
            BASELINE_PREFILL_ARM: baseline,
            CACHE_REUSE_ARM: cache,
        },
    )

    assert baseline.requests[0].kv_transfer_params == {}
    assert baseline.requests[0].request_id is None
    assert cache.requests[0].request_id == "cachet-bio-1"
    assert cache.requests[0].kv_transfer_params == kv_transfer_params


def test_benchmark_generation_validates_output_timing_tokens_and_metadata():
    generation = BenchmarkGeneration(
        output_text="",
        prompt_tokens=0,
        completion_tokens=0,
        ttft_seconds=0.0,
        time_to_completion_seconds=0.0,
        metadata={"source": "unit-test"},
    )

    assert generation.output_text == ""
    assert generation.metadata == {"source": "unit-test"}

    base_kwargs = {
        "output_text": "Ada Lovelace",
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "ttft_seconds": 0.1,
        "time_to_completion_seconds": 0.2,
    }
    with pytest.raises(ValueError, match="output_text must be a string"):
        BenchmarkGeneration(**{**base_kwargs, "output_text": object()})
    with pytest.raises(ValueError, match="prompt_tokens must be a non-negative integer"):
        BenchmarkGeneration(**{**base_kwargs, "prompt_tokens": True})
    with pytest.raises(ValueError, match="completion_tokens must be a non-negative integer"):
        BenchmarkGeneration(**{**base_kwargs, "completion_tokens": -1})
    with pytest.raises(ValueError, match="ttft_seconds must be a non-negative finite number"):
        BenchmarkGeneration(**{**base_kwargs, "ttft_seconds": math.nan})
    with pytest.raises(ValueError, match="time_to_completion_seconds must be a non-negative finite number"):
        BenchmarkGeneration(**{**base_kwargs, "time_to_completion_seconds": math.inf})
    with pytest.raises(ValueError, match="time_to_completion_seconds must be greater than or equal"):
        BenchmarkGeneration(**{**base_kwargs, "ttft_seconds": 2.0, "time_to_completion_seconds": 1.0})
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        BenchmarkGeneration(**{**base_kwargs, "metadata": ()})
    with pytest.raises(ValueError, match="metadata.source must be a string"):
        BenchmarkGeneration(**{**base_kwargs, "metadata": {"source": 1}})


def test_benchmark_example_validates_kv_transfer_params():
    with pytest.raises(TypeError, match="kv_transfer_params must be a mapping"):
        example(kv_transfer_params=[])

    with pytest.raises(ValueError, match="kv_transfer_params.document_kv.handoff_json"):
        example(kv_transfer_params={"document_kv.handoff_json": math.nan})

    with pytest.raises(ValueError, match="kv_transfer_params.document_kv.request_id is required"):
        example(kv_transfer_params={DOCUMENT_KV_HANDOFF_JSON_PARAM: "/tmp/cachet.handoff.json"})

    with pytest.raises(ValueError, match="kv_transfer_params.document_kv.request_id must be a non-empty string"):
        example(kv_transfer_params={DOCUMENT_KV_REQUEST_ID_PARAM: ""})

    with pytest.raises(ValueError, match="kv_transfer_params must include document_kv.handoff_json"):
        example(kv_transfer_params={DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-bio-1"})

    with pytest.raises(ValueError, match="only one"):
        example(
            kv_transfer_params={
                DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-bio-1",
                DOCUMENT_KV_HANDOFF_JSON_PARAM: "/tmp/cachet.handoff.json",
                DOCUMENT_KV_HANDOFF_RECORD_PARAM: {},
            }
        )

    with pytest.raises(ValueError, match="kv_transfer_params.document_kv.handoff_json must be a non-empty string"):
        example(
            kv_transfer_params={
                DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-bio-1",
                DOCUMENT_KV_HANDOFF_JSON_PARAM: "",
            }
        )

    with pytest.raises(ValueError, match="kv_transfer_params.document_kv.payload_uri: payload_uri must be an absolute"):
        example(
            kv_transfer_params={
                DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-bio-1",
                DOCUMENT_KV_HANDOFF_JSON_PARAM: "/tmp/cachet.handoff.json",
                DOCUMENT_KV_PAYLOAD_URI_PARAM: "not-a-uri-or-absolute-path",
            }
        )

    with pytest.raises(ValueError, match="kv_transfer_params.document_kv.handoff_record must be an object"):
        example(
            kv_transfer_params={
                DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-bio-1",
                DOCUMENT_KV_HANDOFF_RECORD_PARAM: "not-an-object",
            }
        )

    example(
        kv_transfer_params={
            DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-bio-1",
            DOCUMENT_KV_HANDOFF_RECORD_PARAM: inline_handoff_record(request_id="cachet-bio-1"),
        }
    )

    with pytest.raises(ValueError, match="handoff_record.request_id must match"):
        example(
            kv_transfer_params={
                DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-bio-1",
                DOCUMENT_KV_HANDOFF_RECORD_PARAM: inline_handoff_record(request_id="different"),
            }
        )

    with pytest.raises(ValueError, match="kv_transfer_params.document_kv.handoff_record.payload_source.uri"):
        example(
            kv_transfer_params={
                DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-bio-1",
                DOCUMENT_KV_HANDOFF_RECORD_PARAM: inline_handoff_record(
                    request_id="cachet-bio-1",
                    payload_uri="s3://bucket/cachet-bio-1.kv",
                ),
            }
        )

    with pytest.raises(ValueError, match="Unsupported engine adapter handoff record_type"):
        example(
            kv_transfer_params={
                DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-bio-1",
                DOCUMENT_KV_HANDOFF_RECORD_PARAM: {"request_id": "cachet-bio-1"},
            }
        )


def test_run_benchmark_suite_records_engine_errors_without_aborting():
    suite = BenchmarkSuite(suite_id="v1-smoke", examples=(example(),))
    result = run_benchmark_suite(
        suite,
        {
            BASELINE_PREFILL_ARM: RecordingEngine(),
            CACHE_REUSE_ARM: RecordingEngine(fail=True),
        },
    )
    cache_measurement = next(measurement for measurement in result.measurements if measurement.arm_id == CACHE_REUSE_ARM)

    assert cache_measurement.error == "engine unavailable"
    assert cache_measurement.metadata == {"error_type": "RuntimeError"}
    assert result.comparisons[0].ttft_speedup is None


def test_run_benchmark_suite_records_empty_message_engine_errors_without_aborting():
    suite = BenchmarkSuite(suite_id="v1-smoke", examples=(example(),))
    result = run_benchmark_suite(
        suite,
        {
            BASELINE_PREFILL_ARM: RecordingEngine(),
            CACHE_REUSE_ARM: EmptyMessageFailureEngine(),
        },
    )
    cache_measurement = next(measurement for measurement in result.measurements if measurement.arm_id == CACHE_REUSE_ARM)

    assert cache_measurement.error == "TimeoutError"
    assert cache_measurement.metadata == {"error_type": "TimeoutError"}
    assert result.comparisons[0].ttft_speedup is None


def test_run_benchmark_suite_captures_invalid_generation_schema_as_error_measurement():
    suite = BenchmarkSuite(suite_id="v1-smoke", examples=(example(),))
    result = run_benchmark_suite(
        suite,
        {
            BASELINE_PREFILL_ARM: RecordingEngine(),
            CACHE_REUSE_ARM: InvalidGenerationEngine(ttft_seconds=math.nan),
        },
    )
    cache_measurement = next(measurement for measurement in result.measurements if measurement.arm_id == CACHE_REUSE_ARM)

    assert cache_measurement.error == "ttft_seconds must be a non-negative finite number"
    assert cache_measurement.metadata == {"error_type": "ValueError"}
    assert result.comparisons[0].ttft_speedup is None


def test_run_benchmark_suite_captures_invalid_generation_metadata_as_error_measurement():
    suite = BenchmarkSuite(suite_id="v1-smoke", examples=(example(),))
    result = run_benchmark_suite(
        suite,
        {
            BASELINE_PREFILL_ARM: RecordingEngine(),
            CACHE_REUSE_ARM: InvalidGenerationEngine(metadata={"arm": CACHE_REUSE_ARM, "usage": 3}),
        },
    )
    cache_measurement = next(measurement for measurement in result.measurements if measurement.arm_id == CACHE_REUSE_ARM)

    assert cache_measurement.error == "metadata.usage must be a string"
    assert cache_measurement.metadata == {"error_type": "ValueError"}
    assert result.comparisons[0].ttft_speedup is None


def test_benchmark_run_result_to_record_serializes_latency_quality_and_comparison():
    suite = BenchmarkSuite(suite_id="v1-smoke", examples=(example(),))
    result = run_benchmark_suite(
        suite,
        {
            BASELINE_PREFILL_ARM: RecordingEngine(output="Charles Babbage"),
            CACHE_REUSE_ARM: RecordingEngine(),
        },
    )

    record = benchmark_run_result_to_record(result)

    assert record["record_type"] == BENCHMARK_RUN_RECORD_TYPE
    assert record["suite"] == {
        "suite_id": "v1-smoke",
        "model_id": "qwen3:4b-instruct",
        "hardware_target": "aws-g6-l4",
        "datasets": ["biography", "hotpotqa", "musique", "niah"],
        "examples": 1,
    }
    assert record["measurements"][0]["exact_match"] is False
    assert record["measurements"][1]["answer_found"] is True
    assert record["report_rows"][0]["ttft"]["p50"] == pytest.approx(4.0)
    assert record["comparisons"][0]["ttft_speedup"] == pytest.approx(4.0)
    assert record["v1_evidence"]["ok"] is False
    assert record["v1_evidence"]["required_datasets"] == ["biography", "hotpotqa", "musique", "niah"]
    assert record["v1_evidence"]["duplicate_required_datasets"] == []
    assert record["v1_evidence"]["duplicate_report_rows"] == []
    assert record["v1_evidence"]["duplicate_comparisons"] == []
    assert "hotpotqa:baseline_prefill" in record["v1_evidence"]["missing_report_rows"]
    assert "hotpotqa" in record["v1_evidence"]["missing_comparisons"]
    assert record["v1_evidence"]["comparisons_without_metrics"] == []
    assert record["v1_evidence"]["unexpected_arms"] == []


def test_benchmark_run_result_to_record_uses_result_arm_ids_for_v1_evidence():
    suite = BenchmarkSuite(suite_id="v1-custom", examples=(example(),))
    arms = (
        BenchmarkArm(arm_id="full_prefill", uses_cache=False, description="baseline"),
        BenchmarkArm(arm_id="kv_reuse", uses_cache=True, description="cache"),
    )

    result = run_benchmark_suite(
        suite,
        {
            "full_prefill": RecordingEngine(),
            "kv_reuse": RecordingEngine(),
        },
        arms=arms,
    )

    record = benchmark_run_result_to_record(result)

    assert record["comparisons"][0]["baseline_arm_id"] == "full_prefill"
    assert record["comparisons"][0]["cache_arm_id"] == "kv_reuse"
    assert record["v1_evidence"]["baseline_arm_id"] == "full_prefill"
    assert record["v1_evidence"]["cache_arm_id"] == "kv_reuse"
    assert "biography:full_prefill" not in record["v1_evidence"]["missing_report_rows"]
    assert "biography:kv_reuse" not in record["v1_evidence"]["missing_report_rows"]
    assert "hotpotqa:full_prefill" in record["v1_evidence"]["missing_report_rows"]
    assert all("baseline_prefill" not in row for row in record["v1_evidence"]["missing_report_rows"])
    assert record["v1_evidence"]["unexpected_arms"] == []


def test_run_openai_compatible_v1_benchmark_uses_factory_for_baseline_and_cache(tmp_path):
    path = tmp_path / "biography.jsonl"
    path.write_text(
        json.dumps({"query": "Who wrote notes?", "documents": ["Ada wrote notes."], "answer": "Ada Lovelace"})
        + "\n",
        encoding="utf-8",
    )
    built: list[tuple[str, str | None, bool]] = []

    def factory(arm, config):
        built.append((arm.arm_id, config.cache_base_url, config.cache_runtime_prompt))
        return RecordingEngine()

    result = run_openai_compatible_v1_benchmark(
        OpenAICompatibleBenchmarkConfig(
            suite_id="v1-openai",
            dataset_paths={"biography": path},
            base_url="http://baseline",
            cache_base_url="http://cache",
            cache_runtime_prompt=True,
            repeats=2,
        ),
        engine_factory=factory,
    )

    assert built == [
        (BASELINE_PREFILL_ARM, "http://cache", True),
        (CACHE_REUSE_ARM, "http://cache", True),
    ]
    assert len(result.measurements) == 4
    assert {measurement.dataset for measurement in result.measurements} == {"biography"}
    cache_measurement = next(measurement for measurement in result.measurements if measurement.arm_id == CACHE_REUSE_ARM)
    baseline_measurement = next(
        measurement for measurement in result.measurements if measurement.arm_id == BASELINE_PREFILL_ARM
    )
    assert cache_measurement.prompt_tokens < baseline_measurement.prompt_tokens


def test_openai_compatible_benchmark_config_validates_dataset_paths():
    with pytest.raises(ValueError, match="dataset_paths"):
        OpenAICompatibleBenchmarkConfig(suite_id="v1", dataset_paths={}, base_url="http://server")

    with pytest.raises(ValueError, match="Unsupported V1 dataset"):
        OpenAICompatibleBenchmarkConfig(
            suite_id="v1",
            dataset_paths={"natural-questions": "nq.jsonl"},
            base_url="http://server",
        )


def test_openai_compatible_benchmark_config_rejects_empty_limit_and_unsafe_runtime_prompt():
    with pytest.raises(ValueError, match="limit_per_dataset"):
        OpenAICompatibleBenchmarkConfig(
            suite_id="v1",
            dataset_paths={"biography": "biography.jsonl"},
            base_url="http://server",
            limit_per_dataset=0,
        )

    with pytest.raises(ValueError, match="cache_runtime_prompt requires cache_base_url"):
        OpenAICompatibleBenchmarkConfig(
            suite_id="v1",
            dataset_paths={"biography": "biography.jsonl"},
            base_url="http://server",
            cache_runtime_prompt=True,
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("suite_id", 123, "suite_id must be non-empty"),
        ("base_url", "", "base_url must be non-empty"),
        ("cache_base_url", "", "cache_base_url must be non-empty"),
        ("endpoint", "", "endpoint must be non-empty"),
        ("cache_endpoint", "", "cache_endpoint must be non-empty"),
        ("model_id", "", "model_id must be non-empty"),
        ("hardware_target", "", "hardware_target must be non-empty"),
        ("hardware_target", "aws-g6e", "Unsupported V1 hardware target"),
        ("limit_per_dataset", True, "limit_per_dataset must be positive"),
        ("repeats", True, "repeats must be positive"),
        ("seed", True, "seed must be an integer"),
        ("shuffle", 1, "shuffle must be a boolean"),
        ("max_tokens", True, "max_tokens must be positive"),
        ("temperature", math.nan, "temperature must be a non-negative finite number"),
        ("timeout_seconds", math.inf, "timeout_seconds must be a positive finite number"),
        ("stream", 1, "stream must be a boolean"),
        ("cache_runtime_prompt", 0, "cache_runtime_prompt must be a boolean"),
        ("api_key", 123, "api_key must be a string"),
        ("prefix_cache_salt_mode", "dynamic", "prefix_cache_salt_mode"),
    ],
)
def test_openai_compatible_benchmark_config_rejects_invalid_public_fields(field_name, value, message):
    kwargs = {
        "suite_id": "v1",
        "dataset_paths": {"biography": "biography.jsonl"},
        "base_url": "http://server",
    }
    if field_name == "cache_runtime_prompt":
        kwargs["cache_base_url"] = "http://cache"

    with pytest.raises(ValueError, match=message):
        OpenAICompatibleBenchmarkConfig(**{**kwargs, field_name: value})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dataset_paths": []}, "dataset_paths must be a mapping"),
        ({"dataset_paths": {"": "bio.jsonl"}}, "dataset_paths keys"),
        ({"dataset_paths": {"biography": 3}}, "dataset_paths.biography"),
        ({"baseline_extra_body": []}, "baseline_extra_body must be a mapping"),
        ({"baseline_extra_body": {"": 1}}, "baseline_extra_body keys"),
        ({"baseline_extra_body": {"temperature": math.nan}}, "baseline_extra_body.temperature"),
        ({"baseline_extra_body": {"bad": object()}}, "baseline_extra_body.bad"),
        ({"cache_extra_body": {"nested": {"bad": object()}}}, "cache_extra_body.nested.bad"),
    ],
)
def test_openai_compatible_benchmark_config_rejects_invalid_mappings(overrides, message):
    kwargs = {
        "suite_id": "v1",
        "dataset_paths": {"biography": "biography.jsonl"},
        "base_url": "http://server",
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        OpenAICompatibleBenchmarkConfig(**kwargs)


def test_openai_compatible_benchmark_config_normalizes_json_body_tuples():
    config = OpenAICompatibleBenchmarkConfig(
        suite_id="v1",
        dataset_paths={"biography": "biography.jsonl"},
        base_url="http://server",
        baseline_extra_body={"guided_choice": ("yes", "no")},
    )

    assert config.baseline_extra_body == {"guided_choice": ["yes", "no"]}


def test_openai_compatible_engine_derives_per_request_prefix_cache_salts():
    config = OpenAICompatibleBenchmarkConfig(
        suite_id="v1",
        dataset_paths={"biography": "biography.jsonl"},
        base_url="http://server",
        cache_base_url="http://cache",
        cache_extra_body={"cache_salt": "cachet-kv-cache"},
        prefix_cache_salt_mode="per_request",
    )
    engine = legacy_benchmark_runner._openai_compatible_engine(default_benchmark_arms()[1], config)
    request = BenchmarkEngineRequest(
        suite_id="v1",
        model_id="qwen3:4b-instruct",
        hardware_target="aws-g6-l4",
        example=example("biography", example_id="bio-1"),
        arm=default_benchmark_arms()[1],
        prompt_parts=legacy_benchmark_runner.build_prompt_parts(example("biography", example_id="bio-1")),
        repeat_index=2,
    )

    assert engine.extra_body_factory is not None
    assert engine.extra_body_factory(request) == {
        "cache_salt": "cachet-kv-cache:v1:biography:bio-1:document_kv_cache:repeat-2"
    }


def test_openai_compatible_engine_normalizes_openai_v1_base_url_for_default_endpoint():
    config = OpenAICompatibleBenchmarkConfig(
        suite_id="v1",
        dataset_paths={"biography": "biography.jsonl"},
        base_url="http://server/v1",
    )

    engine = legacy_benchmark_runner._openai_compatible_engine(default_benchmark_arms()[0], config)

    assert engine.config.base_url == "http://server"
    assert engine.config.endpoint == "/v1/completions"


def test_openai_compatible_engine_appends_custom_endpoint_to_exact_base_url():
    config = OpenAICompatibleBenchmarkConfig(
        suite_id="v1",
        dataset_paths={"biography": "biography.jsonl"},
        base_url="http://server/v1",
        endpoint="/completions",
    )

    engine = legacy_benchmark_runner._openai_compatible_engine(default_benchmark_arms()[0], config)

    assert engine.config.base_url == "http://server/v1"
    assert engine.config.endpoint == "/completions"


def test_main_writes_v1_benchmark_json(monkeypatch, tmp_path):
    path = tmp_path / "biography.jsonl"
    path.write_text(json.dumps({"query": "Bio?", "documents": ["Bio context"], "answer": "Bio"}) + "\n")
    output_path = tmp_path / "result.json"

    def fake_run(config):
        assert config.dataset_paths == {"biography": path}
        assert config.base_url == "http://server"
        assert config.cache_base_url == "http://cache"
        assert config.cache_runtime_prompt
        return run_benchmark_suite(
            BenchmarkSuite(suite_id=config.suite_id, examples=(example(),)),
            {
                BASELINE_PREFILL_ARM: RecordingEngine(),
                CACHE_REUSE_ARM: RecordingEngine(),
            },
        )

    monkeypatch.setattr("restaurant_kv_serving.benchmark_runner.run_openai_compatible_v1_benchmark", fake_run)

    exit_code = legacy_benchmark_runner.main(
        [
            "--dataset",
            f"biography={path}",
            "--base-url",
            "http://server",
            "--cache-base-url",
            "http://cache",
            "--cache-runtime-prompt",
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["record_type"] == BENCHMARK_RUN_RECORD_TYPE
    assert record["suite"]["suite_id"] == "v1-openai-compatible"
    assert [row["arm_id"] for row in record["report_rows"]] == [BASELINE_PREFILL_ARM, CACHE_REUSE_ARM]


def test_main_accepts_file_uri_dataset_and_output_paths(monkeypatch, tmp_path):
    path = tmp_path / "biography.jsonl"
    path.write_text(json.dumps({"query": "Bio?", "documents": ["Bio context"], "answer": "Bio"}) + "\n")
    output_path = tmp_path / "nested" / "result.json"

    def fake_run(config):
        assert config.dataset_paths == {"biography": path}
        return run_benchmark_suite(
            BenchmarkSuite(suite_id=config.suite_id, examples=(example(),)),
            {
                BASELINE_PREFILL_ARM: RecordingEngine(),
                CACHE_REUSE_ARM: RecordingEngine(),
            },
        )

    monkeypatch.setattr("restaurant_kv_serving.benchmark_runner.run_openai_compatible_v1_benchmark", fake_run)

    exit_code = legacy_benchmark_runner.main(
        [
            "--dataset",
            f"biography=file:{path}",
            "--base-url",
            "http://server",
            "--output-json",
            f"file:{output_path}",
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["suite"]["suite_id"] == "v1-openai-compatible"


def test_public_benchmark_runner_main_respects_document_namespace_monkeypatch(monkeypatch, tmp_path):
    path = tmp_path / "biography.jsonl"
    path.write_text(json.dumps({"query": "Bio?", "documents": ["Bio context"], "answer": "Bio"}) + "\n")
    output_path = tmp_path / "public-result.json"
    original_legacy_run = legacy_benchmark_runner.run_openai_compatible_v1_benchmark

    def fake_run(config):
        assert config.dataset_paths == {"biography": path}
        return run_benchmark_suite(
            BenchmarkSuite(suite_id=config.suite_id, examples=(example(),)),
            {
                BASELINE_PREFILL_ARM: RecordingEngine(),
                CACHE_REUSE_ARM: RecordingEngine(),
            },
        )

    monkeypatch.setattr(public_benchmark_runner, "run_openai_compatible_v1_benchmark", fake_run)

    exit_code = public_benchmark_runner.main(
        [
            "--dataset",
            f"biography={path}",
            "--base-url",
            "http://server",
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["suite"]["suite_id"] == "v1-openai-compatible"
    assert legacy_benchmark_runner.run_openai_compatible_v1_benchmark is original_legacy_run


def test_public_benchmark_runner_main_restores_legacy_hook_after_error(monkeypatch, tmp_path):
    path = tmp_path / "biography.jsonl"
    path.write_text(json.dumps({"query": "Bio?", "documents": ["Bio context"], "answer": "Bio"}) + "\n")
    original_legacy_run = legacy_benchmark_runner.run_openai_compatible_v1_benchmark

    def fake_run(config):
        raise RuntimeError("public hook failed")

    monkeypatch.setattr(public_benchmark_runner, "run_openai_compatible_v1_benchmark", fake_run)

    exit_code = public_benchmark_runner.main(
        [
            "--dataset",
            f"biography={path}",
            "--base-url",
            "http://server",
        ]
    )

    assert exit_code == 1
    assert legacy_benchmark_runner.run_openai_compatible_v1_benchmark is original_legacy_run


def test_run_benchmark_suite_requires_one_engine_per_arm():
    with pytest.raises(ValueError, match="Missing benchmark engines"):
        run_benchmark_suite(BenchmarkSuite(suite_id="v1", examples=(example(),)), {})


def test_run_benchmark_suite_supports_repeats_and_seeded_shuffle():
    suite = BenchmarkSuite(suite_id="v1-smoke", examples=(example(),))
    baseline = RecordingEngine()
    cache = RecordingEngine()

    result = run_benchmark_suite(
        suite,
        {
            BASELINE_PREFILL_ARM: baseline,
            CACHE_REUSE_ARM: cache,
        },
        repeats=3,
        shuffle=True,
        seed=7,
    )

    assert len(result.measurements) == 6
    assert sum(1 for measurement in result.measurements if measurement.arm_id == BASELINE_PREFILL_ARM) == 3
    assert sum(1 for measurement in result.measurements if measurement.arm_id == CACHE_REUSE_ARM) == 3
    assert [request.repeat_index for request in baseline.requests] == [1, 2, 3]
    assert [request.repeat_index for request in cache.requests] == [1, 2, 3]


def test_seeded_shuffle_uses_dataset_and_example_identity():
    suite = BenchmarkSuite(
        suite_id="v1-shared-local-id",
        examples=(
            example("biography", example_id="shared-1"),
            example("hotpotqa", example_id="shared-1"),
        ),
        datasets=("biography", "hotpotqa"),
    )

    result = run_benchmark_suite(
        suite,
        {
            BASELINE_PREFILL_ARM: RecordingEngine(),
            CACHE_REUSE_ARM: RecordingEngine(),
        },
        repeats=4,
        shuffle=True,
        seed=1,
    )

    arm_order_by_dataset = {
        dataset: [measurement.arm_id for measurement in result.measurements if measurement.dataset == dataset]
        for dataset in ("biography", "hotpotqa")
    }

    assert arm_order_by_dataset["biography"] != arm_order_by_dataset["hotpotqa"]


def test_run_benchmark_suite_rejects_non_positive_repeats():
    with pytest.raises(ValueError, match="repeats"):
        run_benchmark_suite(
            BenchmarkSuite(suite_id="v1", examples=(example(),)),
            {
                BASELINE_PREFILL_ARM: RecordingEngine(),
                CACHE_REUSE_ARM: RecordingEngine(),
            },
            repeats=0,
        )


def test_run_benchmark_suite_rejects_duplicate_arm_ids():
    duplicate_arms = (
        BenchmarkArm(arm_id="same", uses_cache=False, description="baseline"),
        BenchmarkArm(arm_id="same", uses_cache=True, description="cache"),
    )

    with pytest.raises(ValueError, match="Duplicate benchmark arm ids"):
        run_benchmark_suite(
            BenchmarkSuite(suite_id="v1", examples=(example(),)),
            {"same": RecordingEngine()},
            arms=duplicate_arms,
        )


def test_default_benchmark_arms_are_baseline_then_cache():
    arms = default_benchmark_arms()

    assert [arm.arm_id for arm in arms] == [BASELINE_PREFILL_ARM, CACHE_REUSE_ARM]
    assert [arm.uses_cache for arm in arms] == [False, True]


def test_load_benchmark_jsonl_accepts_canonical_schema(tmp_path):
    path = tmp_path / "biography.jsonl"
    path.write_text(
        json.dumps(
            {
                "example_id": "bio-1",
                "dataset": "biography",
                "query": "Who wrote notes?",
                "expected_answer": "Ada Lovelace",
                "kv_transfer_params": {
                    DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-bio-1",
                    DOCUMENT_KV_HANDOFF_JSON_PARAM: "/Volumes/catalog/schema/volume/cachet/bio-1.handoff.json",
                    "document_kv.payload_uri": "uc-volume:/catalog/schema/volume/cachet/bio-1.kv",
                },
                "documents": [
                    {
                        "document_id": "ada",
                        "title": "Ada",
                        "static_text": "Biography",
                        "chunks": [
                            {"chunk_id": "p1", "text": "Notes", "metadata": {"source": 1}},
                            "String chunk",
                        ],
                    }
                ],
                "metadata": {"split": "dev"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_benchmark_jsonl(path)

    assert loaded[0].example_id == "bio-1"
    assert loaded[0].dataset == "biography"
    assert loaded[0].metadata == {"split": "dev"}
    assert loaded[0].kv_transfer_params == {
        DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-bio-1",
        DOCUMENT_KV_HANDOFF_JSON_PARAM: "/Volumes/catalog/schema/volume/cachet/bio-1.handoff.json",
        DOCUMENT_KV_PAYLOAD_URI_PARAM: "uc-volume:/catalog/schema/volume/cachet/bio-1.kv",
    }
    assert loaded[0].documents[0].metadata["title"] == "Ada"
    assert [chunk.chunk_id for chunk in loaded[0].documents[0].chunks] == ["static", "p1", "chunk-1"]
    assert loaded[0].documents[0].chunks[1].metadata == {"source": "1"}


def test_load_benchmark_jsonl_accepts_dataset_default_and_static_only_documents(tmp_path):
    path = tmp_path / "niah.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "needle-1",
                "question": "What is the needle?",
                "target": "blue lantern",
                "documents": [{"id": "haystack", "static_text": "The needle is blue lantern."}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_benchmark_jsonl(path, dataset="niah")

    assert loaded[0].example_id == "needle-1"
    assert loaded[0].dataset == "niah"
    assert loaded[0].expected_answer == "blue lantern"
    assert loaded[0].documents[0].chunks[0].chunk_id == "static"


def test_load_benchmark_jsonl_keeps_logical_default_ids_when_file_has_blank_lines(tmp_path):
    path = tmp_path / "biography.jsonl"
    path.write_text("\n" + json.dumps({"query": "Bio?", "documents": ["Bio context"]}) + "\n", encoding="utf-8")

    loaded = load_benchmark_jsonl(path, dataset="biography")

    assert loaded[0].example_id == "biography-1"


def test_load_v1_jsonl_suite_combines_dataset_files(tmp_path):
    bio_path = tmp_path / "biography.jsonl"
    hotpot_path = tmp_path / "hotpotqa.jsonl"
    bio_path.write_text(
        json.dumps({"query": "Bio?", "documents": ["Bio context"], "answer": "Bio"}) + "\n",
        encoding="utf-8",
    )
    hotpot_path.write_text(
        json.dumps({"query": "Hotpot?", "documents": [{"text": "Hotpot context"}], "answer": "Hotpot"}) + "\n",
        encoding="utf-8",
    )

    suite = load_v1_jsonl_suite(
        suite_id="v1-jsonl",
        paths={"biography": bio_path, "hotpotqa": hotpot_path},
        limit_per_dataset=1,
    )

    assert suite.datasets == ("biography", "hotpotqa")
    assert [example.dataset for example in suite.examples] == ["biography", "hotpotqa"]
    assert [example.example_id for example in suite.examples] == ["biography-1", "hotpotqa-1"]


def test_load_v1_jsonl_suite_rejects_dataset_mismatch(tmp_path):
    path = tmp_path / "hotpotqa.jsonl"
    path.write_text(
        "\n"
        + json.dumps({"dataset": "biography", "query": "Bio?", "documents": ["Bio context"], "answer": "Bio"})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Benchmark JSONL line 2: dataset 'biography' does not match expected dataset 'hotpotqa'",
    ):
        load_v1_jsonl_suite(suite_id="v1-jsonl", paths={"hotpotqa": path})


def test_load_v1_jsonl_suite_rejects_empty_suite(tmp_path):
    path = tmp_path / "biography.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="at least one example"):
        load_v1_jsonl_suite(suite_id="v1-jsonl", paths={"biography": path})


def test_load_v1_jsonl_suite_rejects_empty_requested_dataset(tmp_path):
    bio_path = tmp_path / "biography.jsonl"
    hotpot_path = tmp_path / "hotpotqa.jsonl"
    bio_path.write_text(json.dumps({"query": "Bio?", "documents": ["Bio context"], "answer": "Bio"}) + "\n")
    hotpot_path.write_text("")

    with pytest.raises(ValueError, match="hotpotqa"):
        load_v1_jsonl_suite(suite_id="v1-jsonl", paths={"biography": bio_path, "hotpotqa": hotpot_path})


def test_load_benchmark_jsonl_accepts_hotpotqa_context_pairs(tmp_path):
    path = tmp_path / "hotpotqa.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "Who wrote notes?",
                "answer": "Ada Lovelace",
                "context": [["Ada", ["Ada was a writer.", "She wrote notes."]]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_benchmark_jsonl(path, dataset="hotpotqa")

    assert loaded[0].dataset == "hotpotqa"
    assert loaded[0].documents[0].document_id == "Ada"
    assert loaded[0].documents[0].metadata["title"] == "Ada"
    assert [chunk.text for chunk in loaded[0].documents[0].chunks] == ["Ada was a writer.", "She wrote notes."]


def test_load_benchmark_jsonl_accepts_musique_paragraphs(tmp_path):
    path = tmp_path / "musique.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "Where?",
                "answer": "Paris",
                "paragraphs": [
                    {"idx": 0, "title": "France", "paragraph_text": "Paris is in France."},
                    {"id": "p2", "paragraph_text": "Berlin is in Germany."},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_benchmark_jsonl(path, dataset="musique")

    assert loaded[0].dataset == "musique"
    assert [document.document_id for document in loaded[0].documents] == ["France", "p2"]
    assert loaded[0].documents[0].chunks[0].text == "Paris is in France."


def test_load_benchmark_jsonl_validates_records(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("\n" + json.dumps({"dataset": "unknown", "query": "Bad?", "documents": ["x"]}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Benchmark JSONL line 2: Unsupported V1 dataset"):
        load_benchmark_jsonl(path)


def test_load_benchmark_jsonl_rejects_invalid_kv_transfer_params(tmp_path):
    path = tmp_path / "bad-kv-transfer.jsonl"
    path.write_text(
        json.dumps(
            {
                "dataset": "biography",
                "query": "Bad?",
                "documents": ["x"],
                "kv_transfer_params": {"document_kv.handoff_json": "/tmp/cachet.handoff.json"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Benchmark JSONL line 1: kv_transfer_params.document_kv.request_id"):
        load_benchmark_jsonl(path)


def test_load_benchmark_jsonl_reports_invalid_json_line(tmp_path):
    path = tmp_path / "bad-json.jsonl"
    path.write_text(json.dumps({"query": "Ok?", "documents": ["x"]}) + "\n{not json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Benchmark JSONL line 2 is not valid JSON"):
        load_benchmark_jsonl(path, dataset="biography")


def test_load_benchmark_jsonl_limit_does_not_parse_rows_after_limit(tmp_path):
    path = tmp_path / "limited.jsonl"
    path.write_text(json.dumps({"query": "Ok?", "documents": ["x"]}) + "\n{not json}\n", encoding="utf-8")

    loaded = load_benchmark_jsonl(path, dataset="biography", limit=1)

    assert len(loaded) == 1
    assert loaded[0].example_id == "biography-1"
