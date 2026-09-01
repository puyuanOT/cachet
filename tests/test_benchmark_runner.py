import json
import math
import threading
import time
from dataclasses import replace
from hashlib import sha256

import pytest

from document_kv_cache.benchmark_gates import (
    CacheStateAttestation,
    evaluate_benchmark_evidence_gate,
)

from document_kv_cache.benchmark_runner import (
    BENCHMARK_RUN_RECORD_TYPE,
    BenchmarkEngineRequest,
    BenchmarkGeneration,
    BenchmarkManifestContext,
    OpenAICompatibleBenchmarkConfig,
    PUBLICATION_LATENCY_DEPLOYMENT_BLOCK_METADATA_KEY,
    PUBLICATION_LATENCY_INPUT_BUNDLE_SHA256_METADATA_KEY,
    PUBLICATION_LATENCY_LANE_METADATA_KEY,
    PUBLICATION_LATENCY_LANE_POSITION_METADATA_KEY,
    PUBLICATION_LATENCY_REQUEST_ID_METADATA_KEY,
    PUBLICATION_LATENCY_REQUEST_INDEX_METADATA_KEY,
    PUBLICATION_LATENCY_REQUESTS_SHA256_METADATA_KEY,
    PUBLICATION_LATENCY_SCHEDULE_SHA256_METADATA_KEY,
    PUBLICATION_LATENCY_SEED_SHA256_METADATA_KEY,
    benchmark_run_result_to_record,
    benchmark_run_result_to_evidence_record,
    benchmark_run_result_from_record,
    benchmark_record_aggregate_issues,
    default_benchmark_arms,
    load_benchmark_jsonl,
    load_v1_jsonl_suite,
    merge_isolated_benchmark_run_records,
    parse_benchmark_arm_specs,
    run_benchmark_suite,
    run_openai_compatible_benchmark,
    run_openai_compatible_v1_benchmark,
)
from document_kv_cache.benchmarks import (
    BASELINE_PREFILL_ARM,
    CACHE_REUSE_ARM,
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_HANDOFF_RECORD_PARAM,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
    DOCUMENT_KV_CACHE_METHOD_PARAM,
    DOCUMENT_KV_RUNTIME_PREFIX_TEXT_PARAM,
    DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM,
    FINAL_ANSWER_EXTRACTED_METADATA_KEY,
    FINAL_ANSWER_NO_EXTRACTION_VALUE,
    FINAL_ANSWER_PARSER_DIGEST,
    FINAL_ANSWER_PARSER_DIGEST_METADATA_KEY,
    FINAL_ANSWER_PARSER_STATUS_METADATA_KEY,
    FINAL_ANSWER_PARSER_VALID_METADATA_KEY,
    SUPPORTED_V1_DATASETS,
    BenchmarkArm,
    BenchmarkExample,
    BenchmarkSuite,
    BenchmarkPromptParts,
    DatasetScorer,
    DatasetScorerRegistry,
    document_kv_cache_arm,
)
from document_kv_cache.engine import EngineReadyRequest
from document_kv_cache.engine_adapters import (
    build_engine_adapter_request,
    engine_adapter_request_to_record,
    vllm_adapter_spec,
)
from document_kv_cache.engine_protocol import KVCacheHandle, KVLayout, KVSegment
from document_kv_cache.methods import default_method_registry
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
)
from document_kv_cache.publication_inputs import (
    PublicationLatencyExample,
    build_publication_latency_block_schedule,
    project_publication_latency_request_order,
)
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


class SlowRecordingEngine(RecordingEngine):
    def __init__(self, *, delay_seconds: float = 0.05) -> None:
        super().__init__()
        self.delay_seconds = delay_seconds
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def generate(self, request: BenchmarkEngineRequest) -> BenchmarkGeneration:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            time.sleep(self.delay_seconds)
            return super().generate(request)
        finally:
            with self._lock:
                self._active -= 1


class TimingSkewPublicationEngine:
    def __init__(
        self,
        request_indices,
        *,
        slow_request_index,
        slow_delay_seconds=0.12,
    ):
        self.request_indices = request_indices
        self.slow_request_index = slow_request_index
        self.slow_delay_seconds = slow_delay_seconds
        self._lock = threading.Lock()
        self._active_identities = set()
        self._slow_active = False
        self.identity_overlap_detected = False
        self.later_request_started_during_slow = False
        self.max_active = 0
        self.requests = []

    def generate(self, request):
        key = (
            request.example.dataset,
            request.example.example_id,
            request.repeat_index,
        )
        identity = key[:2]
        request_index = self.request_indices[key]
        with self._lock:
            if identity in self._active_identities:
                self.identity_overlap_detected = True
            self._active_identities.add(identity)
            self.requests.append(request)
            if request_index == self.slow_request_index:
                self._slow_active = True
            elif self._slow_active and request_index >= len(self.request_indices) // 2:
                self.later_request_started_during_slow = True
            self.max_active = max(self.max_active, len(self._active_identities))
        try:
            time.sleep(
                self.slow_delay_seconds
                if request_index == self.slow_request_index
                else 0.0002
            )
            return BenchmarkGeneration(
                output_text="<final_answer>Ada Lovelace</final_answer>",
                prompt_tokens=len(request.prompt_text.split()),
                completion_tokens=2,
                ttft_seconds=0.001,
                time_to_completion_seconds=0.002,
                metadata={"arm": request.arm.arm_id},
            )
        finally:
            with self._lock:
                self._active_identities.remove(identity)
                if request_index == self.slow_request_index:
                    self._slow_active = False


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


def _cachet_arm_spec(method_id: str, *, method_version: str = "1") -> dict[str, object]:
    registry = default_method_registry()
    connector_mode = (
        registry.get(method_id).connector_mode
        if method_id in registry.specs
        else "cachet"
    )
    return {
        "arm_id": f"cachet:{method_id}",
        "uses_cache": True,
        "description": f"Cachet method {method_id}",
        "cache_method": method_id,
        "connector_mode": connector_mode,
        "implementation_kind": "cachet",
        "method_version": method_version,
        "method_config_digest": "0" * 64,
        "physical_transform_id": f"cachet.{method_id}",
        "requires_cachet_handoff": True,
    }


@pytest.mark.parametrize(
    "method_id",
    ("kv_packet", "cacheblend", "infoflow_kv", "unknown_method"),
)
def test_arbitrary_cachet_arm_specs_reject_planned_and_unknown_methods(
    method_id: str,
) -> None:
    with pytest.raises(ValueError, match="registered runnable Cachet method"):
        parse_benchmark_arm_specs((_cachet_arm_spec(method_id),))


def test_arbitrary_cachet_arm_specs_accept_explicit_runnable_custom_registry() -> None:
    registry = default_method_registry()
    vanilla = registry.get("vanilla_prefill", require_implemented=True)
    custom = replace(
        vanilla,
        method="vendor_custom_prefill",
        arm_id="document_kv_cache:vendor_custom_prefill",
        display_name="Vendor custom prefill",
    )
    custom_registry = registry.with_spec(custom)

    arms, _, _, _ = parse_benchmark_arm_specs(
        (
            {
                **_cachet_arm_spec(
                    custom.method_id,
                    method_version=custom.artifact_version,
                ),
                "connector_mode": custom.connector_mode,
            },
        ),
        method_registry=custom_registry,
    )

    assert arms[0].cache_method == custom.method_id


def test_arbitrary_upstream_arm_spec_does_not_require_cachet_registration() -> None:
    arms, _, _, _ = parse_benchmark_arm_specs(
        (
            {
                "arm_id": "upstream:author_method",
                "uses_cache": True,
                "description": "Author implementation",
                "cache_method": "author_method",
                "implementation_kind": "upstream",
                "method_version": "paper-revision",
                "method_config_digest": "1" * 64,
                "physical_transform_id": "author.packetization",
                "source_revision": "author-commit",
                "checkpoint_identity": "checkpoint-sha256",
                "requires_cachet_handoff": False,
            },
        )
    )

    assert arms[0].implementation_kind == "upstream"


def test_arm_spec_parses_only_typed_runtime_environment_overrides() -> None:
    spec = {
        "arm_id": "upstream",
        "uses_cache": True,
        "description": "upstream",
        "cache_method": "upstream_method",
        "implementation_kind": "upstream",
        "source_revision": "source",
        "checkpoint_identity": "checkpoint",
        "requires_cachet_handoff": False,
        "runtime_environment_overrides": {
            "serving_platform": "vllm",
            "tensor_parallel_size": 2,
        },
    }

    arms, _, _, _ = parse_benchmark_arm_specs((spec,))

    assert dict(arms[0].runtime_environment_overrides) == {
        "serving_platform": "vllm",
        "tensor_parallel_size": 2,
    }
    with pytest.raises(ValueError, match="unknown fields"):
        parse_benchmark_arm_specs(
            ({**spec, "runtime_environment_overrides": {"label_only": "bad"}},)
        )


def test_arbitrary_engine_native_cachet_arm_does_not_require_cachet_handoff() -> None:
    lmcache = default_method_registry().get("lmcache", require_implemented=True)
    spec = _cachet_arm_spec("lmcache", method_version=lmcache.artifact_version)
    spec["requires_cachet_handoff"] = False

    arms, _, _, _ = parse_benchmark_arm_specs((spec,))

    assert arms[0].cache_method == "lmcache"
    assert arms[0].requires_cachet_handoff is False


def test_programmatic_runner_rejects_planned_cachet_arm_before_execution() -> None:
    planned = default_method_registry().get("kv_packet")
    arm = BenchmarkArm(
        arm_id="cachet:kv_packet",
        uses_cache=True,
        description="planned KV Packet",
        cache_method=planned.method_id,
        connector_mode=planned.connector_mode,
        implementation_kind="cachet",
        method_version=planned.artifact_version,
        method_config_digest="0" * 64,
        physical_transform_id="cachet.kv_packet",
        requires_cachet_handoff=True,
    )

    with pytest.raises(ValueError, match="registered runnable Cachet method"):
        run_benchmark_suite(
            BenchmarkSuite(suite_id="planned-method", examples=(example(),)),
            {arm.arm_id: RecordingEngine()},
            arms=(arm,),
        )


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
        cache_method="full_prefix_prefill",
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


def test_runner_persists_raw_and_extracted_answers_and_zeroes_invalid_structure():
    suite = BenchmarkSuite(
        suite_id="answer-parser",
        examples=(example(),),
        datasets=("biography",),
    )
    valid = run_benchmark_suite(
        suite,
        {BASELINE_PREFILL_ARM: RecordingEngine(output="<final_answer>Ada Lovelace</final_answer>")},
        arms=(BenchmarkArm(BASELINE_PREFILL_ARM, False, "baseline"),),
    ).measurements[0]
    invalid = run_benchmark_suite(
        suite,
        {BASELINE_PREFILL_ARM: RecordingEngine(output="Ada Lovelace")},
        arms=(BenchmarkArm(BASELINE_PREFILL_ARM, False, "baseline"),),
    ).measurements[0]

    assert valid.output_text == "<final_answer>Ada Lovelace</final_answer>"
    assert valid.metadata[FINAL_ANSWER_EXTRACTED_METADATA_KEY] == "Ada Lovelace"
    assert valid.metadata[FINAL_ANSWER_PARSER_VALID_METADATA_KEY] == "true"
    assert valid.metadata[FINAL_ANSWER_PARSER_STATUS_METADATA_KEY] == "ok"
    assert valid.metadata[FINAL_ANSWER_PARSER_DIGEST_METADATA_KEY] == (
        FINAL_ANSWER_PARSER_DIGEST
    )
    assert valid.quality_scores == {"exact_match": 1.0}
    assert invalid.output_text == "Ada Lovelace"
    assert invalid.metadata[FINAL_ANSWER_EXTRACTED_METADATA_KEY] == (
        FINAL_ANSWER_NO_EXTRACTION_VALUE
    )
    assert invalid.metadata[FINAL_ANSWER_PARSER_VALID_METADATA_KEY] == "false"
    assert invalid.quality_scores == {"exact_match": 0.0}


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
    assert cache.requests[0].request_id == (
        "v1-smoke:biography:biography-1:document_kv_cache:repeat-1:cachet-bio-1"
    )
    assert cache.requests[0].kv_transfer_params == kv_transfer_params


def test_n_way_cache_arms_receive_distinct_physical_handoffs_for_same_logical_example():
    vanilla_method = default_method_registry().get(
        "vanilla_prefill",
        require_implemented=True,
    )
    baseline_arm = BenchmarkArm(
        arm_id="baseline",
        uses_cache=False,
        description="baseline",
    )
    first_arm = BenchmarkArm(
        arm_id="vanilla",
        uses_cache=True,
        description="vanilla",
        cache_method="vanilla_prefill",
        connector_mode=vanilla_method.connector_mode,
        variant_id="default",
        method_version=vanilla_method.artifact_version,
        method_config_digest="0" * 64,
        physical_transform_id="cachet.vanilla",
    )
    second_arm = BenchmarkArm(
        arm_id="upstream",
        uses_cache=True,
        description="upstream",
        cache_method="upstream_method",
        variant_id="author",
        implementation_kind="upstream",
        physical_transform_id="author.upstream",
        source_revision="author-commit-1",
        checkpoint_identity="author-checkpoint-1",
        requires_cachet_handoff=False,
    )
    logical_example = BenchmarkExample(
        example_id="biography-n-way",
        dataset="biography",
        documents=example().documents,
        query="Who wrote notes on the Analytical Engine?",
        expected_answer="Ada Lovelace",
        arm_kv_transfer_params={
            "vanilla": {
                DOCUMENT_KV_REQUEST_ID_PARAM: "vanilla-handoff",
                DOCUMENT_KV_HANDOFF_JSON_PARAM: "/tmp/vanilla.json",
                DOCUMENT_KV_CACHE_METHOD_PARAM: "vanilla_prefill",
            },
        },
    )
    suite = BenchmarkSuite(
        suite_id="n-way",
        examples=(logical_example,),
        datasets=("biography",),
    )
    baseline = RecordingEngine()
    vanilla = RecordingEngine()
    upstream = RecordingEngine()

    result = run_benchmark_suite(
        suite,
        {"baseline": baseline, "vanilla": vanilla, "upstream": upstream},
        arms=(baseline_arm, first_arm, second_arm),
    )

    assert vanilla.requests[0].kv_transfer_params[DOCUMENT_KV_REQUEST_ID_PARAM] == "vanilla-handoff"
    assert upstream.requests[0].kv_transfer_params == {}
    assert vanilla.requests[0].logical_prompt_text == upstream.requests[0].logical_prompt_text
    assert len(result.comparisons) == 2
    assert result.experiment_manifest is not None
    transform_digests = {
        arm.arm_id: arm.physical_transform_config_digest
        for arm in result.experiment_manifest.arms
    }
    assert transform_digests["vanilla"] != transform_digests["upstream"]


def test_benchmark_rejects_handoff_method_that_disagrees_with_arm_before_execution():
    method = default_method_registry().get(
        "vanilla_prefill",
        require_implemented=True,
    )
    arm = BenchmarkArm(
        arm_id="vanilla",
        uses_cache=True,
        description="vanilla",
        cache_method=method.method_id,
        connector_mode=method.connector_mode,
        variant_id="default",
        method_version=method.artifact_version,
        method_config_digest="1" * 64,
        requires_cachet_handoff=True,
    )
    mismatched = example(
        kv_transfer_params={
            DOCUMENT_KV_REQUEST_ID_PARAM: "mismatched-method",
            DOCUMENT_KV_HANDOFF_JSON_PARAM: "/tmp/mismatched-method.json",
            DOCUMENT_KV_CACHE_METHOD_PARAM: "kv_packet",
        }
    )
    engine = RecordingEngine()

    with pytest.raises(ValueError, match="declare cache method 'kv_packet'"):
        run_benchmark_suite(
            BenchmarkSuite(suite_id="mismatched-method", examples=(mismatched,)),
            {arm.arm_id: engine},
            arms=(arm,),
        )

    assert engine.requests == []


def test_merge_independent_one_arm_records_rebuilds_union_comparison():
    examples = (
        example(example_id="biography-1"),
        example(example_id="biography-2"),
    )
    suite = BenchmarkSuite(
        suite_id="physical-jobs",
        examples=examples,
        datasets=("biography",),
    )
    baseline = BenchmarkArm(
        arm_id="baseline",
        uses_cache=False,
        description="full prefill",
    )
    first = BenchmarkArm(
        arm_id="author_a",
        uses_cache=True,
        description="author method A",
        cache_method="author_a",
        variant_id="default",
        implementation_kind="upstream",
        method_version="1",
        method_config_digest="a" * 64,
        physical_transform_id="author.a",
        source_revision="commit-a",
        checkpoint_identity="checkpoint-a",
        requires_cachet_handoff=False,
    )
    second = BenchmarkArm(
        arm_id="author_b",
        uses_cache=True,
        description="author method B",
        cache_method="author_b",
        variant_id="default",
        implementation_kind="upstream",
        method_version="1",
        method_config_digest="b" * 64,
        physical_transform_id="author.b",
        source_revision="commit-b",
        checkpoint_identity="checkpoint-b",
        requires_cachet_handoff=False,
    )

    records = []
    for index, arm in enumerate((baseline, first, second), start=1):
        context = BenchmarkManifestContext(
            model_revision="model-revision",
            tokenizer_id="tokenizer",
            tokenizer_revision="tokenizer-revision",
            engine_id="test-engine",
            engine_version="1",
            package_revisions=(("cachet", "revision"),),
            max_output_tokens=16,
            temperature=0.0,
            stream=False,
            hardware_fingerprint="same-hardware",
            runtime_id=f"ephemeral-cluster-{index}",
            runtime_version="runtime-1",
            storage_identity="same-storage",
            cache_state="warm",
            measurement_scopes=("latency",),
        )
        result = run_benchmark_suite(
            suite,
            {arm.arm_id: RecordingEngine(output="TOP_SECRET_OUTPUT")},
            arms=(arm,),
            manifest_context=context,
            evidence_policy="canary",
        )
        assert result.reference_arm_id == arm.arm_id
        records.append(benchmark_run_result_to_evidence_record(result))

    tampered_window = json.loads(json.dumps(records[0]))
    tampered_window["execution_windows"][0]["successful_requests"] = 999
    assert any(
        "successful_requests does not match raw measurements" in issue
        for issue in benchmark_record_aggregate_issues(tampered_window)
    )
    contradictory_diagnostic = json.loads(json.dumps(records[0]))
    original_exact = contradictory_diagnostic["measurements"][0]["exact_match"]
    contradictory_diagnostic["measurements"][0]["exact_match"] = not original_exact
    assert any(
        "exact_match contradicts quality_scores/output" in issue
        for issue in benchmark_record_aggregate_issues(contradictory_diagnostic)
    )

    merged = merge_isolated_benchmark_run_records(
        records,
        reference_arm_id="baseline",
        policy="canary",
    )

    assert merged["experiment_manifest"]["comparison"]["reference_arm_id"] == "baseline"
    assert merged["experiment_manifest"]["execution"]["isolation_mode"] == (
        "separate_process_or_job"
    )
    assert merged["experiment_manifest"]["environment"]["runtime_id"].startswith(
        "separate_jobs:"
    )
    assert {item["cache_arm_id"] for item in merged["comparisons"]} == {
        "author_a",
        "author_b",
    }
    assert len(merged["paired_statistics"]["rows"]) == 2
    assert merged["evidence_sanitized"] is True
    serialized = json.dumps(merged)
    assert "TOP_SECRET_OUTPUT" not in serialized
    assert "Ada Lovelace" not in serialized

    duplicate_execution = json.loads(json.dumps(records))
    duplicate_execution[1]["experiment_manifest"]["environment"]["runtime_id"] = (
        duplicate_execution[0]["experiment_manifest"]["environment"]["runtime_id"]
    )
    with pytest.raises(ValueError, match="distinct execution-instance runtime_id"):
        merge_isolated_benchmark_run_records(
            duplicate_execution,
            reference_arm_id="baseline",
            policy="canary",
        )

    source_execution_ids = merged["experiment_manifest"]["execution"][
        "source_execution_ids"
    ]
    assert {item["arm_id"] for item in source_execution_ids} == {
        "baseline",
        "author_a",
        "author_b",
    }
    assert len({item["execution_id_digest"] for item in source_execution_ids}) == 3


def test_sanitized_evidence_drops_untrusted_generation_metadata():
    class SecretMetadataEngine:
        def generate(self, request):
            return BenchmarkGeneration(
                output_text="TOP_SECRET_OUTPUT",
                prompt_tokens=8,
                completion_tokens=1,
                ttft_seconds=0.1,
                time_to_completion_seconds=0.2,
                metadata={
                    "raw_prompt": "TOP_SECRET_PROMPT",
                    "raw_response": "TOP_SECRET_RESPONSE",
                    "prefix_cache_salt": "TOP_SECRET_SALT",
                    "request_payload_prompt_sha256": "b" * 64,
                    "request_body": "TOP_SECRET_REQUEST_BODY",
                    "payload_path": "/tmp/TOP_SECRET_PATH",
                    "authorization": "Bearer TOP_SECRET_TOKEN",
                },
            )

    baseline = BenchmarkArm(
        arm_id="baseline",
        uses_cache=False,
        description="baseline",
    )
    result = run_benchmark_suite(
        BenchmarkSuite(
            suite_id="sanitized-metadata",
            examples=(example(),),
            datasets=("biography",),
        ),
        {baseline.arm_id: SecretMetadataEngine()},
        arms=(baseline,),
    )
    result = replace(
        result,
        measurements=(
            replace(
                result.measurements[0],
                completion_tokens=0,
                expected_answer="TOP_SECRET_EXPECTED_ANSWER",
                references=("TOP_SECRET_EXPECTED_ANSWER",),
                error="TOP_SECRET_ENGINE_ERROR",
                request_id="TOP_SECRET_REQUEST_ID",
            ),
        ),
        execution_windows=tuple(
            replace(
                window,
                completion_tokens=0,
                successful_requests=0,
            )
            for window in result.execution_windows
        ),
    )

    record = benchmark_run_result_to_evidence_record(
        result,
        cache_state_attestations=(
            CacheStateAttestation(
                request_id="TOP_SECRET_ATTESTATION_REQUEST",
                cache_method="vanilla_prefill",
                artifact_id="a" * 64,
                source="local_path",
                bytes_read=1,
                payload_cache_hit=False,
                eviction_requested=True,
                eviction_succeeded=True,
            ),
        ),
    )
    serialized = json.dumps(record, sort_keys=True)

    assert "TOP_SECRET" not in serialized
    assert set(record["measurements"][0]["metadata"]) == {
        "logical_prompt_sha256",
        "runtime_prompt_sha256",
        "request_payload_prompt_sha256",
        "prefix_cache_salt_sha256",
    }
    measurement = record["measurements"][0]
    assert measurement["output_text"] == ""
    assert measurement["expected_answer"] is None
    assert measurement["references"] == []
    assert measurement["error"] == "redacted"
    assert len(measurement["request_id"]) == 64
    assert len(record["gate_inputs"]["cache_state_attestations"][0]["request_id"]) == 64


def test_n_way_cache_arms_reject_ambiguous_legacy_handoff_mapping():
    suite = BenchmarkSuite(
        suite_id="ambiguous-n-way",
        examples=(example(kv_transfer_params={}),),
        datasets=("biography",),
    )
    arms = (
        BenchmarkArm(arm_id="baseline", uses_cache=False, description="baseline"),
        BenchmarkArm(arm_id="cache-a", uses_cache=True, description="cache a"),
        BenchmarkArm(arm_id="cache-b", uses_cache=True, description="cache b"),
    )

    with pytest.raises(ValueError, match="distinct arm_kv_transfer_params"):
        run_benchmark_suite(
            suite,
            {arm.arm_id: RecordingEngine() for arm in arms},
            arms=arms,
        )


def test_setting_variation_binds_declared_value_to_actual_arm_environment():
    fp16 = BenchmarkArm(
        arm_id="fp16",
        uses_cache=True,
        description="author fp16",
        cache_method="author_method",
        variant_id="fp16",
        implementation_kind="upstream",
        method_version="1",
        method_config_digest="1" * 64,
        physical_transform_id="author.method",
        source_revision="author-commit",
        checkpoint_identity="author-checkpoint",
        setting_overrides={"model_quantization": "fp16"},
        runtime_environment_overrides={"model_quantization": "fp16"},
        requires_cachet_handoff=False,
    )
    int8 = BenchmarkArm(
        arm_id="int8",
        uses_cache=True,
        description="author int8",
        cache_method="author_method",
        variant_id="int8",
        implementation_kind="upstream",
        method_version="1",
        method_config_digest="1" * 64,
        physical_transform_id="author.method",
        source_revision="author-commit",
        checkpoint_identity="author-checkpoint",
        setting_overrides={"model_quantization": "int8"},
        runtime_environment_overrides={"model_quantization": "int8"},
        requires_cachet_handoff=False,
    )
    suite = BenchmarkSuite(
        suite_id="setting-variation",
        examples=(example(),),
        datasets=("biography",),
    )
    context = BenchmarkManifestContext(
        comparison_mode="single_method_setting_variation",
        varied_setting="model_quantization",
        reference_arm_id="fp16",
        model_quantization="fp16",
    )

    result = run_benchmark_suite(
        suite,
        {"fp16": RecordingEngine(), "int8": RecordingEngine()},
        arms=(fp16, int8),
        manifest_context=context,
        reference_arm_id="fp16",
    )
    record = benchmark_run_result_to_record(result)

    assert result.reference_arm_id == "fp16"
    assert result.cache_arm_ids == ("int8",)
    assert result.comparisons[0].baseline_arm_id == "fp16"
    serialized_setting = record["experiment_manifest"]["arms"][0][
        "setting_overrides"
    ]
    assert serialized_setting == {"model_quantization": "fp16"}
    assert record["experiment_manifest"]["arms"][1]["runtime_environment"][
        "model_quantization"
    ] == "int8"
    json.dumps(record)
    with pytest.raises(TypeError):
        fp16.runtime_environment_overrides["model_quantization"] = "unsafe"


def test_legacy_v1_manifest_without_per_arm_environment_remains_readable() -> None:
    suite = BenchmarkSuite(
        suite_id="legacy-environment",
        examples=(example(),),
        datasets=("biography",),
    )
    result = run_benchmark_suite(
        suite,
        {BASELINE_PREFILL_ARM: RecordingEngine()},
        arms=(BenchmarkArm("baseline_prefill", False, "baseline"),),
        manifest_context=BenchmarkManifestContext(
            model_revision="model-revision",
            tokenizer_id="tokenizer",
            tokenizer_revision="tokenizer-revision",
            engine_id="engine",
            engine_version="1",
            hardware_fingerprint="hardware",
            runtime_id="runtime",
            runtime_version="1",
            storage_identity="storage",
            cache_state="warm",
        ),
    )
    record = benchmark_run_result_to_record(result)
    manifest = record["experiment_manifest"]
    manifest["arms"][0].pop("runtime_environment")
    for field_name in (
        "canonical_model_id",
        "lora_id",
        "prompt_template_version",
        "serving_platform",
        "model_dtype",
        "model_quantization",
        "runtime_kv_dtype",
        "layout_version",
        "payload_axis_order",
        "block_size",
        "key_position_encoding",
        "tensor_parallel_size",
        "pipeline_parallel_size",
    ):
        manifest["model_runtime"].pop(field_name)

    reconstructed = benchmark_run_result_from_record(record)

    assert reconstructed.experiment_manifest is not None
    environment = reconstructed.experiment_manifest.arms[0].runtime_environment
    assert environment.served_model_id == suite.model_id
    assert environment.canonical_model_id == suite.model_id
    assert environment.serving_platform == "unresolved"


@pytest.mark.parametrize(
    ("varied_setting", "reference_value", "candidate_value"),
    (
        ("hardware_target", "g5.8xlarge", "g6.8xlarge"),
        ("serving_platform", "vllm", "sglang"),
    ),
)
def test_setting_variation_accepts_one_typed_actual_environment_field(
    varied_setting: str,
    reference_value: str,
    candidate_value: str,
) -> None:
    def arm(arm_id: str, value: str) -> BenchmarkArm:
        return BenchmarkArm(
            arm_id=arm_id,
            uses_cache=True,
            description=arm_id,
            cache_method="author_method",
            variant_id=arm_id,
            implementation_kind="upstream",
            method_version="1",
            method_config_digest="1" * 64,
            physical_transform_id="author.method",
            source_revision="author-commit",
            checkpoint_identity="author-checkpoint",
            setting_overrides={varied_setting: value},
            runtime_environment_overrides={varied_setting: value},
            requires_cachet_handoff=False,
        )

    suite = BenchmarkSuite(
        suite_id=f"variation-{varied_setting}",
        examples=(example(),),
        datasets=("biography",),
    )
    result = run_benchmark_suite(
        suite,
        {"reference": RecordingEngine(), "candidate": RecordingEngine()},
        arms=(arm("reference", reference_value), arm("candidate", candidate_value)),
        manifest_context=BenchmarkManifestContext(
            comparison_mode="single_method_setting_variation",
            varied_setting=varied_setting,
            reference_arm_id="reference",
        ),
        reference_arm_id="reference",
    )

    assert result.experiment_manifest is not None
    environments = {
        candidate.arm_id: candidate.runtime_environment
        for candidate in result.experiment_manifest.arms
    }
    assert getattr(environments["reference"], varied_setting) == reference_value
    assert getattr(environments["candidate"], varied_setting) == candidate_value


def test_setting_variation_rejects_a_second_actual_environment_drift() -> None:
    reference = BenchmarkArm(
        arm_id="reference",
        uses_cache=True,
        description="reference",
        cache_method="author_method",
        implementation_kind="upstream",
        method_version="1",
        method_config_digest="1" * 64,
        source_revision="author-commit",
        checkpoint_identity="author-checkpoint",
        setting_overrides={"model_quantization": "fp16"},
        runtime_environment_overrides={"model_quantization": "fp16"},
        requires_cachet_handoff=False,
    )
    candidate = replace(
        reference,
        arm_id="candidate",
        description="candidate",
        setting_overrides={"model_quantization": "int8"},
        runtime_environment_overrides={
            "model_quantization": "int8",
            "serving_platform": "sglang",
        },
    )

    with pytest.raises(ValueError, match="typed dependent fields"):
        run_benchmark_suite(
            BenchmarkSuite(
                suite_id="second-drift",
                examples=(example(),),
                datasets=("biography",),
            ),
            {"reference": RecordingEngine(), "candidate": RecordingEngine()},
            arms=(reference, candidate),
            manifest_context=BenchmarkManifestContext(
                comparison_mode="single_method_setting_variation",
                varied_setting="model_quantization",
                reference_arm_id="reference",
            ),
            reference_arm_id="reference",
        )


def test_hardware_setting_dimension_allows_honest_dependent_provenance() -> None:
    reference = BenchmarkArm(
        arm_id="g5",
        uses_cache=True,
        description="g5",
        cache_method="author_method",
        implementation_kind="upstream",
        method_version="1",
        method_config_digest="1" * 64,
        source_revision="author-commit",
        checkpoint_identity="author-checkpoint",
        setting_overrides={"hardware_target": "g5.8xlarge"},
        runtime_environment_overrides={
            "hardware_target": "g5.8xlarge",
            "hardware_fingerprint": "a10g-24gb",
            "storage_identity": "g5-local-nvme",
        },
        requires_cachet_handoff=False,
    )
    candidate = replace(
        reference,
        arm_id="g6",
        description="g6",
        setting_overrides={"hardware_target": "g6.8xlarge"},
        runtime_environment_overrides={
            "hardware_target": "g6.8xlarge",
            "hardware_fingerprint": "l4-24gb",
            "storage_identity": "g6-local-nvme",
        },
    )
    suite = BenchmarkSuite(
        suite_id="honest-hardware-variation",
        examples=(example(),),
        hardware_target="g5.8xlarge",
        datasets=("biography",),
    )
    context = BenchmarkManifestContext(
        comparison_mode="single_method_setting_variation",
        varied_setting="hardware_target",
        reference_arm_id="g5",
        hardware_fingerprint="a10g-24gb",
        storage_identity="g5-local-nvme",
    )

    result = run_benchmark_suite(
        suite,
        {"g5": RecordingEngine(), "g6": RecordingEngine()},
        arms=(reference, candidate),
        manifest_context=context,
        reference_arm_id="g5",
    )

    assert result.reference_arm_id == "g5"

    with pytest.raises(ValueError, match="typed dependent fields"):
        run_benchmark_suite(
            suite,
            {"g5": RecordingEngine(), "g6": RecordingEngine()},
            arms=(
                reference,
                replace(
                    candidate,
                    runtime_environment_overrides={
                        **dict(candidate.runtime_environment_overrides),
                        "model_revision": "different-model-revision",
                    },
                ),
            ),
            manifest_context=context,
            reference_arm_id="g5",
        )


def test_serving_platform_dimension_allows_engine_identity_dependencies() -> None:
    reference = BenchmarkArm(
        arm_id="vllm",
        uses_cache=True,
        description="vLLM",
        cache_method="author_method",
        implementation_kind="upstream",
        method_version="1",
        method_config_digest="1" * 64,
        source_revision="author-commit",
        checkpoint_identity="author-checkpoint",
        setting_overrides={"serving_platform": "vllm"},
        runtime_environment_overrides={
            "serving_platform": "vllm",
            "engine_id": "vllm",
            "engine_version": "0.27.1",
            "runtime_version": "vllm-runtime",
        },
        requires_cachet_handoff=False,
    )
    candidate = replace(
        reference,
        arm_id="sglang",
        description="SGLang",
        setting_overrides={"serving_platform": "sglang"},
        runtime_environment_overrides={
            "serving_platform": "sglang",
            "engine_id": "sglang",
            "engine_version": "0.5.10.post1",
            "runtime_version": "sglang-runtime",
        },
    )

    result = run_benchmark_suite(
        BenchmarkSuite(
            suite_id="serving-platform-variation",
            examples=(example(),),
            datasets=("biography",),
        ),
        {"vllm": RecordingEngine(), "sglang": RecordingEngine()},
        arms=(reference, candidate),
        manifest_context=BenchmarkManifestContext(
            comparison_mode="single_method_setting_variation",
            varied_setting="serving_platform",
            reference_arm_id="vllm",
            serving_platform="vllm",
            engine_id="vllm",
            engine_version="0.27.1",
            runtime_version="vllm-runtime",
        ),
        reference_arm_id="vllm",
    )

    assert result.reference_arm_id == "vllm"


@pytest.mark.parametrize(
    ("field_name", "different_value"),
    (
        ("implementation_kind", "external"),
        ("method_version", "999"),
        ("method_config_digest", "2" * 64),
        ("connector_mode", "other-connector"),
        ("requires_cachet_handoff", True),
        ("physical_transform_id", "other.transform"),
        ("physical_transform_version", "2"),
        ("physical_transform_config_digest", "2" * 64),
        ("scorer_plugin_path", "other.module:score"),
        ("source_revision", "other-commit"),
        ("checkpoint_identity", "other-checkpoint"),
    ),
)
def test_setting_variation_rejects_non_setting_method_contract_drift(
    field_name: str,
    different_value: object,
) -> None:
    reference = BenchmarkArm(
        arm_id="reference",
        uses_cache=True,
        description="reference",
        cache_method="author_method",
        connector_mode="author-connector",
        implementation_kind="upstream",
        method_version="1",
        method_config_digest="1" * 64,
        physical_transform_id="author.method",
        physical_transform_version="1",
        scorer_plugin_path="author.module:score",
        source_revision="author-commit",
        checkpoint_identity="author-checkpoint",
        setting_overrides={"hardware_target": "g5.8xlarge"},
        runtime_environment_overrides={"hardware_target": "g5.8xlarge"},
        requires_cachet_handoff=False,
    )
    candidate = replace(
        reference,
        arm_id="candidate",
        description="candidate",
        setting_overrides={"hardware_target": "g6.8xlarge"},
        runtime_environment_overrides={"hardware_target": "g6.8xlarge"},
        **{field_name: different_value},
    )

    with pytest.raises(ValueError, match="invariant method and implementation"):
        run_benchmark_suite(
            BenchmarkSuite(
                suite_id="contract-drift",
                examples=(example(),),
                datasets=("biography",),
            ),
            {"reference": RecordingEngine(), "candidate": RecordingEngine()},
            arms=(reference, candidate),
            manifest_context=BenchmarkManifestContext(
                comparison_mode="single_method_setting_variation",
                varied_setting="hardware_target",
                reference_arm_id="reference",
            ),
            reference_arm_id="reference",
        )


def test_quantization_variation_allows_explicit_checkpoint_identity_change() -> None:
    reference = BenchmarkArm(
        arm_id="fp16",
        uses_cache=True,
        description="fp16",
        cache_method="author_method",
        implementation_kind="upstream",
        method_version="1",
        method_config_digest="1" * 64,
        source_revision="author-commit",
        checkpoint_identity="fp16-checkpoint",
        setting_overrides={"model_quantization": "fp16"},
        runtime_environment_overrides={"model_quantization": "fp16"},
        requires_cachet_handoff=False,
    )
    candidate = replace(
        reference,
        arm_id="int8",
        description="int8",
        checkpoint_identity="int8-checkpoint",
        setting_overrides={"model_quantization": "int8"},
        runtime_environment_overrides={"model_quantization": "int8"},
    )

    result = run_benchmark_suite(
        BenchmarkSuite(
            suite_id="quantization-checkpoint",
            examples=(example(),),
            datasets=("biography",),
        ),
        {"fp16": RecordingEngine(), "int8": RecordingEngine()},
        arms=(reference, candidate),
        manifest_context=BenchmarkManifestContext(
            comparison_mode="single_method_setting_variation",
            varied_setting="model_quantization",
            reference_arm_id="fp16",
        ),
        reference_arm_id="fp16",
    )

    assert result.reference_arm_id == "fp16"


def test_methods_same_setting_rejects_actual_environment_drift() -> None:
    first = BenchmarkArm(
        arm_id="first",
        uses_cache=True,
        description="first method",
        cache_method="author_a",
        implementation_kind="upstream",
        source_revision="author-a",
        checkpoint_identity="checkpoint-a",
        runtime_environment_overrides={"serving_platform": "vllm"},
        requires_cachet_handoff=False,
    )
    second = BenchmarkArm(
        arm_id="second",
        uses_cache=True,
        description="second method",
        cache_method="author_b",
        implementation_kind="upstream",
        source_revision="author-b",
        checkpoint_identity="checkpoint-b",
        runtime_environment_overrides={"serving_platform": "sglang"},
        requires_cachet_handoff=False,
    )

    with pytest.raises(ValueError, match="identical actual runtime environments"):
        run_benchmark_suite(
            BenchmarkSuite(
                suite_id="method-environment-drift",
                examples=(example(),),
                datasets=("biography",),
            ),
            {"first": RecordingEngine(), "second": RecordingEngine()},
            arms=(first, second),
            manifest_context=BenchmarkManifestContext(reference_arm_id="first"),
            reference_arm_id="first",
        )


def test_isolated_setting_merge_preserves_each_actual_hardware_environment() -> None:
    records = []
    values = (("g5", "g5.8xlarge"), ("g6", "g6.8xlarge"))
    for index, (arm_id, hardware_target) in enumerate(values, start=1):
        arm = BenchmarkArm(
            arm_id=arm_id,
            uses_cache=True,
            description=arm_id,
            cache_method="author_method",
            implementation_kind="upstream",
            method_version="1",
            method_config_digest="1" * 64,
            source_revision="author-commit",
            checkpoint_identity="author-checkpoint",
            setting_overrides={"hardware_target": hardware_target},
            requires_cachet_handoff=False,
        )
        result = run_benchmark_suite(
            BenchmarkSuite(
                suite_id="isolated-hardware",
                examples=(example(),),
                hardware_target=hardware_target,
                datasets=("biography",),
            ),
            {arm_id: RecordingEngine()},
            arms=(arm,),
            manifest_context=BenchmarkManifestContext(
                comparison_mode="single_method_setting_variation",
                varied_setting="hardware_target",
                reference_arm_id=arm_id,
                runtime_id=f"runtime-{index}",
            ),
            reference_arm_id=arm_id,
            evidence_policy="canary",
        )
        records.append(benchmark_run_result_to_evidence_record(result))

    merged = merge_isolated_benchmark_run_records(
        records,
        reference_arm_id="g5",
        comparison_mode="single_method_setting_variation",
        varied_setting="hardware_target",
        policy="canary",
    )

    arm_environments = {
        arm["arm_id"]: arm["runtime_environment"]
        for arm in merged["experiment_manifest"]["arms"]
    }
    assert arm_environments["g5"]["hardware_target"] == "g5.8xlarge"
    assert arm_environments["g6"]["hardware_target"] == "g6.8xlarge"
    assert merged["experiment_manifest"]["environment"]["hardware_target"] == (
        "varies_by_arm"
    )
    assert merged["suite"]["hardware_target"] == "varies_by_arm"


def test_isolated_setting_merge_rejects_method_contract_drift() -> None:
    records = []
    values = (
        ("g5", "g5.8xlarge", "1"),
        ("g6", "g6.8xlarge", "999"),
    )
    for index, (arm_id, hardware_target, method_version) in enumerate(
        values,
        start=1,
    ):
        arm = BenchmarkArm(
            arm_id=arm_id,
            uses_cache=True,
            description=arm_id,
            cache_method="author_method",
            implementation_kind="upstream",
            method_version=method_version,
            method_config_digest="1" * 64,
            source_revision="author-commit",
            checkpoint_identity="author-checkpoint",
            setting_overrides={"hardware_target": hardware_target},
            requires_cachet_handoff=False,
        )
        result = run_benchmark_suite(
            BenchmarkSuite(
                suite_id="isolated-contract-drift",
                examples=(example(),),
                hardware_target=hardware_target,
                datasets=("biography",),
            ),
            {arm_id: RecordingEngine()},
            arms=(arm,),
            manifest_context=BenchmarkManifestContext(
                comparison_mode="single_method_setting_variation",
                varied_setting="hardware_target",
                reference_arm_id=arm_id,
                runtime_id=f"runtime-{index}",
            ),
            reference_arm_id=arm_id,
            evidence_policy="canary",
        )
        records.append(benchmark_run_result_to_evidence_record(result))

    with pytest.raises(ValueError, match="invariant method and implementation"):
        merge_isolated_benchmark_run_records(
            records,
            reference_arm_id="g5",
            comparison_mode="single_method_setting_variation",
            varied_setting="hardware_target",
            policy="canary",
        )


def test_isolated_arm_environment_override_must_equal_source_context() -> None:
    arm = BenchmarkArm(
        arm_id="g6",
        uses_cache=True,
        description="g6",
        cache_method="author_method",
        implementation_kind="upstream",
        setting_overrides={"hardware_target": "g6.8xlarge"},
        runtime_environment_overrides={"hardware_target": "g6.8xlarge"},
        requires_cachet_handoff=False,
    )

    with pytest.raises(ValueError, match="must equal its source manifest context"):
        run_benchmark_suite(
            BenchmarkSuite(
                suite_id="isolated-context-mismatch",
                examples=(example(),),
                hardware_target="g5.8xlarge",
                datasets=("biography",),
            ),
            {"g6": RecordingEngine()},
            arms=(arm,),
            manifest_context=BenchmarkManifestContext(
                comparison_mode="single_method_setting_variation",
                varied_setting="hardware_target",
                reference_arm_id="g6",
            ),
            reference_arm_id="g6",
        )


def test_run_benchmark_suite_prepends_runtime_prefix_text_for_cache_prompt():
    kv_transfer_params = {
        DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-bio-1",
        DOCUMENT_KV_HANDOFF_JSON_PARAM: "/Volumes/catalog/schema/volume/cachet/bio-1.handoff.json",
        DOCUMENT_KV_RUNTIME_PREFIX_TEXT_PARAM: "tail-prefix",
    }
    suite = BenchmarkSuite(suite_id="v1-smoke", examples=(example(kv_transfer_params=kv_transfer_params),))
    cache = RecordingEngine()

    run_benchmark_suite(
        suite,
        {CACHE_REUSE_ARM: cache},
        arms=(document_kv_cache_arm(),),
    )

    assert cache.requests[0].prompt_text == "tail-prefix" + cache.requests[0].cache_suffix_text


def test_run_benchmark_suite_uses_unique_engine_request_ids_for_cache_repeats():
    kv_transfer_params = {
        DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-bio-1",
        DOCUMENT_KV_HANDOFF_JSON_PARAM: "/tmp/cachet-bio-1.handoff.json",
    }
    suite = BenchmarkSuite(suite_id="v1-smoke", examples=(example(kv_transfer_params=kv_transfer_params),))
    cache = RecordingEngine()

    run_benchmark_suite(
        suite,
        {CACHE_REUSE_ARM: cache},
        arms=(document_kv_cache_arm(),),
        repeats=2,
        request_parallelism=1,
    )

    assert [request.request_id for request in cache.requests] == [
        "v1-smoke:biography:biography-1:document_kv_cache:repeat-1:cachet-bio-1",
        "v1-smoke:biography:biography-1:document_kv_cache:repeat-2:cachet-bio-1",
    ]
    assert all(request.kv_transfer_params == kv_transfer_params for request in cache.requests)


def test_run_benchmark_suite_groups_example_repeats_by_default():
    suite = BenchmarkSuite(
        suite_id="v1-smoke",
        examples=(
            example(example_id="biography-1"),
            example(example_id="biography-2"),
            example(example_id="biography-3"),
        ),
    )
    baseline_arm = BenchmarkArm(arm_id=BASELINE_PREFILL_ARM, uses_cache=False, description="baseline")
    engine = RecordingEngine()

    run_benchmark_suite(
        suite,
        {BASELINE_PREFILL_ARM: engine},
        arms=(baseline_arm,),
        repeats=2,
        request_parallelism=1,
    )

    assert [(request.example.example_id, request.repeat_index) for request in engine.requests] == [
        ("biography-1", 1),
        ("biography-1", 2),
        ("biography-2", 1),
        ("biography-2", 2),
        ("biography-3", 1),
        ("biography-3", 2),
    ]


def test_run_benchmark_suite_interleave_examples_round_robins_across_examples():
    suite = BenchmarkSuite(
        suite_id="v1-smoke",
        examples=(
            example(example_id="biography-1"),
            example(example_id="biography-2"),
            example(example_id="biography-3"),
        ),
    )
    baseline_arm = BenchmarkArm(arm_id=BASELINE_PREFILL_ARM, uses_cache=False, description="baseline")
    engine = RecordingEngine()

    run_benchmark_suite(
        suite,
        {BASELINE_PREFILL_ARM: engine},
        arms=(baseline_arm,),
        repeats=2,
        request_parallelism=1,
        interleave_examples=True,
    )

    ordered = [(request.example.example_id, request.repeat_index) for request in engine.requests]
    # Each repeat cycle visits every example once before repeating, so a
    # request_parallelism=3 wave would draw from three distinct examples.
    assert ordered == [
        ("biography-1", 1),
        ("biography-2", 1),
        ("biography-3", 1),
        ("biography-1", 2),
        ("biography-2", 2),
        ("biography-3", 2),
    ]
    # Interleaving is a pure reordering: identical membership to the grouped order.
    assert sorted(ordered) == [
        ("biography-1", 1),
        ("biography-1", 2),
        ("biography-2", 1),
        ("biography-2", 2),
        ("biography-3", 1),
        ("biography-3", 2),
    ]


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
            DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM: ["page-a", "page-b"],
        }
    )

    with pytest.raises(ValueError, match="document_kv.sglang_hicache_page_keys must be a sequence"):
        example(
            kv_transfer_params={
                DOCUMENT_KV_REQUEST_ID_PARAM: "cachet-bio-1",
                DOCUMENT_KV_HANDOFF_RECORD_PARAM: inline_handoff_record(request_id="cachet-bio-1"),
                DOCUMENT_KV_SGLANG_HICACHE_PAGE_KEYS_PARAM: "page-a",
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
        "request_parallelism": 1,
        "isolate_arms": True,
        "repeats": 1,
        "shuffle": False,
        "seed": None,
        "interleave_examples": False,
        "prefix_cache_salt_mode": "static",
        "arms": [
            {
                "arm_id": "baseline_prefill",
                "uses_cache": False,
                "cache_method": "",
                "connector_mode": "",
                    "variant_id": "",
                "description": "Standard inference prefill that recomputes all document tokens.",
            },
            {
                "arm_id": "document_kv_cache",
                "uses_cache": True,
                "cache_method": "",
                "connector_mode": "cachet",
                    "variant_id": "",
                "description": "Inference path that reuses precomputed document KV cache.",
            },
        ],
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


def test_manifest_records_decode_settings_preimage_and_rejects_digest_tampering():
    result = run_benchmark_suite(
        BenchmarkSuite(suite_id="decode-settings", examples=(example(),)),
        {
            BASELINE_PREFILL_ARM: RecordingEngine(),
            CACHE_REUSE_ARM: RecordingEngine(),
        },
        manifest_context=BenchmarkManifestContext(
            max_output_tokens=8,
            temperature=0.0,
            stream=True,
            generation_seed=7,
            decode_settings={
                "ignore_eos": True,
                "top_p": 0.9,
                "stop": ["END"],
            },
        ),
    )

    record = benchmark_run_result_to_record(result)

    assert record["experiment_manifest"]["decoding"]["settings"] == {
        "ignore_eos": True,
        "stop": ["END"],
        "top_p": 0.9,
    }
    benchmark_run_result_from_record(record)
    record["experiment_manifest"]["decoding"]["settings"]["top_p"] = 0.8
    with pytest.raises(ValueError, match="decoding_config_digest"):
        benchmark_run_result_from_record(record)


def test_manifest_context_rejects_unknown_or_invalid_decode_settings():
    with pytest.raises(ValueError, match="unsupported settings"):
        BenchmarkManifestContext(decode_settings={"raw_request_body": "secret"})
    with pytest.raises(ValueError, match="ignore_eos must be a boolean"):
        BenchmarkManifestContext(decode_settings={"ignore_eos": 1})


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


def test_openai_request_customization_identity_is_digest_only_and_authenticated(
    tmp_path,
):
    path = tmp_path / "biography.jsonl"
    path.write_text(
        json.dumps(
            {
                "query": "Who wrote notes?",
                "documents": ["Ada wrote notes."],
                "answer": "Ada Lovelace",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def run(extra_body, *, salt_mode="static"):
        return run_openai_compatible_v1_benchmark(
            OpenAICompatibleBenchmarkConfig(
                suite_id="request-customization",
                dataset_paths={"biography": path},
                base_url="http://unused",
                arm_ids=(BASELINE_PREFILL_ARM,),
                baseline_extra_body=extra_body,
                cache_extra_body=extra_body,
                prefix_cache_salt_mode=salt_mode,
            ),
            engine_factory=lambda _arm, _config: RecordingEngine(),
        )

    first = run({"custom_params": {"vendor_profile": "TOP_SECRET_A"}})
    second = run({"custom_params": {"vendor_profile": "TOP_SECRET_B"}})
    assert first.experiment_manifest is not None
    assert second.experiment_manifest is not None
    first_arm = first.experiment_manifest.arms[0]
    second_arm = second.experiment_manifest.arms[0]
    assert first_arm.request_customization_digest != (
        second_arm.request_customization_digest
    )
    assert first_arm.physical_transform_config_digest != (
        second_arm.physical_transform_config_digest
    )

    record = benchmark_run_result_to_evidence_record(first)
    serialized = json.dumps(record, sort_keys=True)
    assert "TOP_SECRET_A" not in serialized
    assert record["experiment_manifest"]["arms"][0][
        "request_customization"
    ] == {"config_digest": first_arm.request_customization_digest}
    reconstructed = benchmark_run_result_from_record(record)
    assert reconstructed.experiment_manifest is not None
    assert (
        reconstructed.experiment_manifest.arms[0].request_customization_digest
        == first_arm.request_customization_digest
    )

    dynamic_a = run(
        {"top_p": 0.2, "cache_salt": "dynamic-prefix-a"},
        salt_mode="per_request",
    )
    dynamic_b = run(
        {"top_p": 0.8, "cache_salt": "dynamic-prefix-b"},
        salt_mode="per_request",
    )
    empty_dynamic_salt = run({"cache_salt": ""}, salt_mode="per_request")
    static_salt = run({"cache_salt": "static-prefix"})
    assert dynamic_a.experiment_manifest is not None
    assert dynamic_b.experiment_manifest is not None
    assert empty_dynamic_salt.experiment_manifest is not None
    assert static_salt.experiment_manifest is not None
    dynamic_digest = (
        dynamic_a.experiment_manifest.arms[0].request_customization_digest
    )
    assert (
        dynamic_b.experiment_manifest.arms[0].request_customization_digest
        == dynamic_digest
    )
    assert (
        static_salt.experiment_manifest.arms[0].request_customization_digest
        != dynamic_digest
    )
    assert (
        empty_dynamic_salt.experiment_manifest.arms[0].request_customization_digest
        != dynamic_digest
    )


def test_programmatic_request_customization_identity_requires_exact_arm_coverage():
    arms = default_benchmark_arms()
    with pytest.raises(ValueError, match="cover every benchmark arm exactly"):
        run_benchmark_suite(
            BenchmarkSuite(suite_id="request-customization", examples=(example(),)),
            {arm.arm_id: RecordingEngine() for arm in arms},
            arms=arms,
            request_customization_digests={BASELINE_PREFILL_ARM: "1" * 64},
        )


def test_setting_variation_request_customization_identity_is_invariant_except_platform():
    reference = BenchmarkArm(
        arm_id="reference",
        uses_cache=True,
        description="reference",
        cache_method="author_method",
        implementation_kind="upstream",
        method_version="1",
        method_config_digest="1" * 64,
        source_revision="author-commit",
        checkpoint_identity="author-checkpoint",
        setting_overrides={"hardware_target": "g5.8xlarge"},
        runtime_environment_overrides={"hardware_target": "g5.8xlarge"},
        requires_cachet_handoff=False,
    )
    candidate = replace(
        reference,
        arm_id="candidate",
        description="candidate",
        setting_overrides={"hardware_target": "g6.8xlarge"},
        runtime_environment_overrides={"hardware_target": "g6.8xlarge"},
    )
    suite = BenchmarkSuite(
        suite_id="request-customization-setting",
        examples=(example(),),
        hardware_target="g5.8xlarge",
    )
    with pytest.raises(ValueError, match="invariant static request customizations"):
        run_benchmark_suite(
            suite,
            {"reference": RecordingEngine(), "candidate": RecordingEngine()},
            arms=(reference, candidate),
            manifest_context=BenchmarkManifestContext(
                comparison_mode="single_method_setting_variation",
                varied_setting="hardware_target",
                reference_arm_id="reference",
            ),
            reference_arm_id="reference",
            request_customization_digests={
                "reference": "1" * 64,
                "candidate": "2" * 64,
            },
        )

    platform_reference = replace(
        reference,
        setting_overrides={"serving_platform": "vllm"},
        runtime_environment_overrides={"serving_platform": "vllm"},
    )
    platform_candidate = replace(
        candidate,
        setting_overrides={"serving_platform": "sglang"},
        runtime_environment_overrides={"serving_platform": "sglang"},
    )
    result = run_benchmark_suite(
        suite,
        {"reference": RecordingEngine(), "candidate": RecordingEngine()},
        arms=(platform_reference, platform_candidate),
        manifest_context=BenchmarkManifestContext(
            serving_platform="vllm",
            comparison_mode="single_method_setting_variation",
            varied_setting="serving_platform",
            reference_arm_id="reference",
        ),
        reference_arm_id="reference",
        request_customization_digests={
            "reference": "1" * 64,
            "candidate": "2" * 64,
        },
    )
    assert result.experiment_manifest is not None
    assert {
        arm.request_customization_digest for arm in result.experiment_manifest.arms
    } == {"1" * 64, "2" * 64}


def test_run_openai_compatible_v1_benchmark_can_select_single_arm(tmp_path):
    path = tmp_path / "biography.jsonl"
    path.write_text(
        json.dumps({"query": "Who wrote notes?", "documents": ["Ada wrote notes."], "answer": "Ada Lovelace"})
        + "\n",
        encoding="utf-8",
    )
    built: list[str] = []

    def factory(arm, config):
        built.append(arm.arm_id)
        return RecordingEngine()

    result = run_openai_compatible_v1_benchmark(
        OpenAICompatibleBenchmarkConfig(
            suite_id="v1-openai-baseline-only",
            dataset_paths={"biography": path},
            base_url="http://server",
            arm_ids=(BASELINE_PREFILL_ARM,),
        ),
        engine_factory=factory,
    )

    assert built == [BASELINE_PREFILL_ARM]
    assert [measurement.arm_id for measurement in result.measurements] == [BASELINE_PREFILL_ARM]
    assert result.comparisons == ()


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
        ("request_parallelism", 0, "request_parallelism must be positive"),
        ("arm_ids", ("unknown",), "Unknown benchmark arm ids"),
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


def test_run_benchmark_suite_issues_requests_concurrently():
    suite = BenchmarkSuite(suite_id="v1-smoke", examples=(example(),))
    baseline = SlowRecordingEngine()
    baseline_arm = BenchmarkArm(arm_id=BASELINE_PREFILL_ARM, uses_cache=False, description="baseline")

    result = run_benchmark_suite(
        suite,
        {BASELINE_PREFILL_ARM: baseline},
        arms=(baseline_arm,),
        repeats=4,
        request_parallelism=4,
    )

    assert len(result.measurements) == 4
    assert baseline.max_active > 1
    assert benchmark_run_result_to_record(result)["suite"]["request_parallelism"] == 4


def _publication_latency_suite():
    examples = tuple(
        BenchmarkExample(
            example_id=f"{dataset}-{index:02d}",
            dataset=dataset,
            documents=(
                SourceDocument.from_texts(
                    document_id=f"document-{dataset}-{index:02d}",
                    static_text="Ada Lovelace biography",
                    chunks={
                        "p1": "Lovelace wrote notes on the Analytical Engine."
                    },
                ),
            ),
            query="Who wrote notes on the Analytical Engine?",
            expected_answer="Ada Lovelace",
        )
        for dataset in SUPPORTED_V1_DATASETS
        for index in range(PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET)
    )
    return BenchmarkSuite(suite_id="publication-latency", examples=examples)


def test_publication_schedule_uses_identity_sticky_lanes_under_timing_skew():
    suite = _publication_latency_suite()
    bundle_sha256 = sha256(b"verified-publication-input-bundle").hexdigest()
    schedule_examples = tuple(
        PublicationLatencyExample(example.dataset, example.example_id)
        for example in suite.examples
    )
    schedule = build_publication_latency_block_schedule(
        campaign_id="publication-2026",
        deployment_block=3,
        input_bundle_sha256=bundle_sha256,
        examples=schedule_examples,
    )
    projection = project_publication_latency_request_order(
        schedule,
        examples=schedule_examples,
        expected_input_bundle_sha256=bundle_sha256,
    )
    request_indices = {key: index for index, key in enumerate(projection)}
    lanes = schedule["lanes"]["4"]
    baseline = TimingSkewPublicationEngine(
        request_indices,
        slow_request_index=lanes[0][0],
    )
    cache = TimingSkewPublicationEngine(
        request_indices,
        slow_request_index=lanes[-1][0],
    )

    result = run_benchmark_suite(
        suite,
        {
            BASELINE_PREFILL_ARM: baseline,
            CACHE_REUSE_ARM: cache,
        },
        repeats=2,
        request_parallelism=4,
        publication_latency_schedule_record=schedule,
        publication_latency_expected_input_bundle_sha256=bundle_sha256,
    )

    for engine in (baseline, cache):
        assert engine.max_active > 1
        assert engine.later_request_started_during_slow is True
        assert engine.identity_overlap_detected is False

    measurements_by_arm = {
        arm_id: [
            measurement
            for measurement in result.measurements
            if measurement.arm_id == arm_id
        ]
        for arm_id in (BASELINE_PREFILL_ARM, CACHE_REUSE_ARM)
    }
    for measurements in measurements_by_arm.values():
        assert [
            (
                measurement.dataset,
                measurement.example_id,
                measurement.repeat_index,
            )
            for measurement in measurements
        ] == list(projection)
        for request_index, measurement in enumerate(measurements):
            metadata = measurement.metadata
            assert metadata[PUBLICATION_LATENCY_SCHEDULE_SHA256_METADATA_KEY] == (
                schedule["closed_record_sha256"]
            )
            assert metadata[PUBLICATION_LATENCY_REQUESTS_SHA256_METADATA_KEY] == (
                schedule["requests_sha256"]
            )
            assert metadata[
                PUBLICATION_LATENCY_INPUT_BUNDLE_SHA256_METADATA_KEY
            ] == bundle_sha256
            assert metadata[PUBLICATION_LATENCY_SEED_SHA256_METADATA_KEY] == (
                schedule["seed_sha256"]
            )
            assert metadata[PUBLICATION_LATENCY_DEPLOYMENT_BLOCK_METADATA_KEY] == "3"
            assert metadata[PUBLICATION_LATENCY_REQUEST_INDEX_METADATA_KEY] == str(
                request_index
            )
            assert metadata[PUBLICATION_LATENCY_REQUEST_ID_METADATA_KEY] == (
                schedule["requests"][request_index]["request_id"]
            )

    lane_by_request_index = {
        request_index: lane_index
        for lane_index, lane in enumerate(lanes)
        for request_index in lane
    }
    for engine in (baseline, cache):
        actual_by_lane = {lane_index: [] for lane_index in range(4)}
        for request in engine.requests:
            key = (
                request.example.dataset,
                request.example.example_id,
                request.repeat_index,
            )
            actual_by_lane[lane_by_request_index[request_indices[key]]].append(key)
        assert actual_by_lane == {
            lane_index: [projection[request_index] for request_index in lane]
            for lane_index, lane in enumerate(lanes)
        }

    baseline_metadata = measurements_by_arm[BASELINE_PREFILL_ARM]
    for request_index, measurement in enumerate(baseline_metadata):
        lane_index = lane_by_request_index[request_index]
        assert measurement.metadata[PUBLICATION_LATENCY_LANE_METADATA_KEY] == str(
            lane_index
        )
        assert measurement.metadata[
            PUBLICATION_LATENCY_LANE_POSITION_METADATA_KEY
        ] == str(lanes[lane_index].index(request_index))


def test_publication_schedule_fails_before_execution_on_membership_mismatch():
    suite = _publication_latency_suite()
    bundle_sha256 = sha256(b"verified-publication-input-bundle").hexdigest()
    schedule_examples = tuple(
        PublicationLatencyExample(example.dataset, example.example_id)
        for example in suite.examples
    )
    schedule = build_publication_latency_block_schedule(
        campaign_id="publication-2026",
        deployment_block=1,
        input_bundle_sha256=bundle_sha256,
        examples=schedule_examples,
    )
    replacement = replace(suite.examples[0], example_id="replacement-example")
    mismatched_suite = replace(
        suite,
        examples=(replacement, *suite.examples[1:]),
    )
    engine = RecordingEngine(output="<final_answer>Ada Lovelace</final_answer>")

    with pytest.raises(ValueError, match="verified input bundle"):
        run_benchmark_suite(
            mismatched_suite,
            {BASELINE_PREFILL_ARM: engine},
            arms=(default_benchmark_arms()[0],),
            repeats=2,
            request_parallelism=4,
            publication_latency_schedule_record=schedule,
            publication_latency_expected_input_bundle_sha256=bundle_sha256,
        )

    assert engine.requests == []


@pytest.mark.parametrize("request_parallelism", (1, 2, 4))
def test_publication_schedule_executes_each_closed_concurrency_cell(
    request_parallelism,
):
    suite = _publication_latency_suite()
    bundle_sha256 = sha256(b"verified-publication-input-bundle").hexdigest()
    schedule_examples = tuple(
        PublicationLatencyExample(example.dataset, example.example_id)
        for example in suite.examples
    )
    schedule = build_publication_latency_block_schedule(
        campaign_id="publication-2026",
        deployment_block=2,
        input_bundle_sha256=bundle_sha256,
        examples=schedule_examples,
    )
    engine = RecordingEngine(output="<final_answer>Ada Lovelace</final_answer>")

    result = run_benchmark_suite(
        suite,
        {BASELINE_PREFILL_ARM: engine},
        arms=(default_benchmark_arms()[0],),
        repeats=2,
        request_parallelism=request_parallelism,
        publication_latency_schedule_record=schedule,
        publication_latency_expected_input_bundle_sha256=bundle_sha256,
    )

    assert len(result.measurements) == 256
    assert {
        int(measurement.metadata[PUBLICATION_LATENCY_LANE_METADATA_KEY])
        for measurement in result.measurements
    } == set(range(request_parallelism))


def test_publication_schedule_config_requires_one_record_or_path_and_bundle_sha(
    tmp_path,
):
    bundle_sha256 = sha256(b"verified-publication-input-bundle").hexdigest()
    common = {
        "suite_id": "publication-latency",
        "dataset_paths": {"biography": "biography.jsonl"},
        "base_url": "http://server",
        "repeats": 2,
        "request_parallelism": 4,
    }

    with pytest.raises(ValueError, match="must be provided together"):
        OpenAICompatibleBenchmarkConfig(
            **common,
            publication_latency_schedule_path=tmp_path / "schedule.json",
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        OpenAICompatibleBenchmarkConfig(
            **common,
            publication_latency_schedule_record={"record_type": "test"},
            publication_latency_schedule_path=tmp_path / "schedule.json",
            publication_latency_expected_input_bundle_sha256=bundle_sha256,
        )

    config = OpenAICompatibleBenchmarkConfig(
        **common,
        publication_latency_schedule_path=tmp_path / "schedule.json",
        publication_latency_expected_input_bundle_sha256=bundle_sha256,
    )
    assert config.publication_latency_schedule_path == tmp_path / "schedule.json"


def test_run_benchmark_suite_isolates_arms_by_default():
    suite = BenchmarkSuite(suite_id="v1-smoke", examples=(example(),))

    result = run_benchmark_suite(
        suite,
        {
            BASELINE_PREFILL_ARM: RecordingEngine(),
            CACHE_REUSE_ARM: RecordingEngine(),
        },
        repeats=3,
    )

    # With isolation the measurements are grouped by arm (all of one arm, then the
    # other) rather than interleaved per example/repeat.
    assert [measurement.arm_id for measurement in result.measurements] == [
        BASELINE_PREFILL_ARM,
        BASELINE_PREFILL_ARM,
        BASELINE_PREFILL_ARM,
        CACHE_REUSE_ARM,
        CACHE_REUSE_ARM,
        CACHE_REUSE_ARM,
    ]
    assert result.isolate_arms is True
    assert benchmark_run_result_to_record(result)["suite"]["isolate_arms"] is True


def test_run_benchmark_suite_no_isolate_arms_interleaves_arms():
    suite = BenchmarkSuite(suite_id="v1-smoke", examples=(example(),))

    result = run_benchmark_suite(
        suite,
        {
            BASELINE_PREFILL_ARM: RecordingEngine(),
            CACHE_REUSE_ARM: RecordingEngine(),
        },
        repeats=3,
        isolate_arms=False,
    )

    assert [measurement.arm_id for measurement in result.measurements] == [
        BASELINE_PREFILL_ARM,
        CACHE_REUSE_ARM,
        BASELINE_PREFILL_ARM,
        CACHE_REUSE_ARM,
        BASELINE_PREFILL_ARM,
        CACHE_REUSE_ARM,
    ]
    assert result.isolate_arms is False
    assert benchmark_run_result_to_record(result)["suite"]["isolate_arms"] is False


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
        # Observe the interleaved shuffle order directly; arm isolation would regroup
        # measurements by arm and hide the per-example shuffle ordering under test here.
        isolate_arms=False,
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


def test_run_benchmark_suite_rejects_non_positive_request_parallelism():
    with pytest.raises(ValueError, match="request_parallelism"):
        run_benchmark_suite(
            BenchmarkSuite(suite_id="v1", examples=(example(),)),
            {
                BASELINE_PREFILL_ARM: RecordingEngine(),
                CACHE_REUSE_ARM: RecordingEngine(),
            },
            request_parallelism=0,
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

    loaded = load_benchmark_jsonl(path)
    assert loaded[0].dataset == "unknown"

    with pytest.raises(ValueError, match="Unsupported V1 dataset"):
        load_v1_jsonl_suite(
            suite_id="v1",
            paths={"unknown": path},
        )


def test_generalized_openai_path_uses_explicit_versioned_scorer_and_prompt(tmp_path):
    path = tmp_path / "custom.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "custom-1",
                "query": "Evaluate without a reference.",
                "documents": ["Custom source text."],
                "metadata": {"judge": "pass"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def custom_prompt(example):
        return BenchmarkPromptParts(
            system_prompt="CUSTOM SYSTEM",
            document_context=example.documents[0].chunks[0].text,
            user_prompt=example.query,
        )

    scorer = DatasetScorer(
        scorer_id="tests.custom",
        version="2026.1",
        metric_names=("judge_score",),
        score_function=lambda context: {
            "judge_score": 1.0 if context.metadata["judge"] == "pass" else 0.0
        },
        plugin_path="tests.test_benchmark_runner:custom_score",
        prompt_function=custom_prompt,
        prompt_plugin_path="tests.test_benchmark_runner:custom_prompt",
        prompt_template_version="custom-prompt-v1",
    )
    registry = DatasetScorerRegistry().register("custom-dataset", scorer)
    arm = BenchmarkArm(
        arm_id="baseline",
        uses_cache=False,
        description="baseline",
    )
    config = OpenAICompatibleBenchmarkConfig(
        suite_id="custom-suite",
        dataset_paths={"custom-dataset": path},
        base_url="http://unused",
        hardware_target="custom-gpu",
        arms=(arm,),
        stream=False,
        suite_contract="generalized",
        prompt_template_version="custom-prompt-v1",
    )
    engine = RecordingEngine(output="reference-free output")

    result = run_openai_compatible_benchmark(
        config,
        scorer_registry=registry,
        engine_factory=lambda _arm, _config: engine,
    )

    assert result.suite.datasets == ("custom-dataset",)
    assert result.suite.hardware_target == "custom-gpu"
    assert result.measurements[0].references == ()
    assert result.measurements[0].quality_scores == {"judge_score": 1.0}
    assert engine.requests[0].logical_prompt_text.startswith("CUSTOM SYSTEM")
    assert result.experiment_manifest is not None
    assert result.experiment_manifest.scorer_identities[0].prompt_plugin_path == (
        "tests.test_benchmark_runner:custom_prompt"
    )
    scorer_manifest = result.experiment_manifest.scorer_identities[0]
    bad_prompt_manifest = replace(
        result.experiment_manifest,
        scorer_identities=(
            replace(
                scorer_manifest,
                prompt_plugin_path="",
                prompt_template_version="",
            ),
        ),
    )
    gate = evaluate_benchmark_evidence_gate(
        replace(result, experiment_manifest=bad_prompt_manifest),
        policy="publication",
    )
    assert any(
        "custom dataset 'custom-dataset' requires a versioned prompt" in issue
        for issue in gate.issues
    )


def test_runner_rejects_mismatched_or_mixed_scorer_prompt_template_versions():
    def scorer(template_version):
        return DatasetScorer(
            scorer_id=f"tests.custom.{template_version}",
            version="1",
            metric_names=("score",),
            score_function=lambda _context: {"score": 1.0},
            prompt_function=lambda item: BenchmarkPromptParts(
                system_prompt="CUSTOM",
                document_context=item.documents[0].chunks[0].text,
                user_prompt=item.query,
            ),
            prompt_plugin_path="tests.test_benchmark_runner:custom_prompt",
            prompt_template_version=template_version,
        )

    arm = BenchmarkArm(
        arm_id="baseline",
        uses_cache=False,
        description="baseline",
    )
    single_suite = BenchmarkSuite(
        suite_id="custom-prompt-version",
        examples=(example(dataset="custom-a"),),
        datasets=("custom-a",),
    )
    single_registry = DatasetScorerRegistry().register(
        "custom-a",
        scorer("custom-v1"),
    )
    with pytest.raises(ValueError, match="must match manifest_context"):
        run_benchmark_suite(
            single_suite,
            {"baseline": RecordingEngine()},
            arms=(arm,),
            scorer_registry=single_registry,
            manifest_context=BenchmarkManifestContext(
                prompt_template_version="custom-v2"
            ),
        )

    mixed_suite = BenchmarkSuite(
        suite_id="mixed-prompt-versions",
        examples=(
            example(dataset="custom-a"),
            example(dataset="custom-b"),
        ),
        datasets=("custom-a", "custom-b"),
    )
    mixed_registry = single_registry.register(
        "custom-b",
        scorer("custom-v2"),
    )
    with pytest.raises(ValueError, match="one shared prompt_template_version"):
        run_benchmark_suite(
            mixed_suite,
            {"baseline": RecordingEngine()},
            arms=(arm,),
            scorer_registry=mixed_registry,
            manifest_context=BenchmarkManifestContext(
                prompt_template_version="custom-v1"
            ),
        )


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
