from typing import Iterator

from lumiseval_agent.runners import GraphDatasetRunner
from lumiseval_core.types import EvalCase, EvalJobConfig, EvalReport, QualityScore
from lumiseval_ingest.adapters import DatasetAdapter


class _StaticAdapter(DatasetAdapter):
    def __init__(self, cases: list[EvalCase]) -> None:
        self._cases = cases

    @property
    def name(self) -> str:
        return "static"

    def iter_cases(
        self,
        split: str = "train",
        limit: int | None = None,
        seed: int = 42,
    ) -> Iterator[EvalCase]:
        del split, seed
        for idx, case in enumerate(self._cases):
            if limit is not None and idx >= limit:
                break
            yield case


def _fake_graph_fn(
    generation: str,
    job_config: EvalJobConfig,
    question=None,
    ground_truth=None,
    rubric_rules=None,
    reference_files=None,
) -> EvalReport:
    del question, ground_truth, rubric_rules, reference_files
    if generation == "raise":
        raise RuntimeError("synthetic failure")
    return EvalReport(
        job_id=job_config.job_id,
        composite_score=0.8,
        retrieval_score=QualityScore(score=0.7),
        answer_score=QualityScore(score=0.9),
    )


def test_graph_dataset_runner_summarizes_success_and_failure(tmp_path) -> None:
    cases = [
        EvalCase(case_id="ok-1", generation="good output", dataset="demo"),
        EvalCase(case_id="bad-1", generation="raise", dataset="demo"),
    ]
    adapter = _StaticAdapter(cases)
    runner = GraphDatasetRunner(graph_fn=_fake_graph_fn)
    base_cfg = EvalJobConfig(job_id="batch-job")

    summary = runner.run_adapter(
        adapter=adapter,
        base_job_config=base_cfg,
        continue_on_error=True,
        output_dir=tmp_path,
    )

    assert summary.total_cases == 2
    assert summary.succeeded == 1
    assert summary.failed == 1
    assert summary.average_composite_score == 0.8
    assert summary.average_retrieval_score == 0.7
    assert summary.average_answer_score == 0.9
    assert any(result.ok for result in summary.case_results)
    assert any((not result.ok) and "synthetic failure" in (result.error or "") for result in summary.case_results)


def test_graph_dataset_runner_fail_fast_stops_on_first_error() -> None:
    cases = [
        EvalCase(case_id="bad-1", generation="raise", dataset="demo"),
        EvalCase(case_id="ok-2", generation="good output", dataset="demo"),
    ]
    adapter = _StaticAdapter(cases)
    runner = GraphDatasetRunner(graph_fn=_fake_graph_fn)
    base_cfg = EvalJobConfig(job_id="batch-job")

    summary = runner.run_adapter(
        adapter=adapter,
        base_job_config=base_cfg,
        continue_on_error=False,
    )

    assert summary.total_cases == 1
    assert summary.succeeded == 0
    assert summary.failed == 1
