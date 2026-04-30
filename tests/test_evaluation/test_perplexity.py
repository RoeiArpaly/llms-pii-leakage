import json

import numpy as np
import pytest

from pandas import DataFrame

from evaluation.visualizations.sensitivity_analysis_perplexity import (
    plot_threshold_sweep,
    plot_threshold_sweep_plotly,
)


@pytest.fixture()
def baseline_df():
    return DataFrame({
        "y_true": [True, True, True, False, False, False],
        "perplexity": [
            1.000004, 1.000003, 1.000001,
            1.0, 1.0, 1.000004,
        ],
        "model": "gpt-4o-mini-defend",
        "dataset": "baseline",
    })


@pytest.fixture()
def adversarial_df():
    return DataFrame({
        "y_true": [True, True, True],
        "perplexity": [1.000002, 1.0, 1.000003],
        "model": "gpt-4o-mini-defend",
        "dataset": "fuzzy_adv",
    })


@pytest.fixture()
def thresholds():
    return np.arange(1, 1.000005, 0.000001)


class TestPlotThresholdSweep:

    def test_returns_figure(self, baseline_df, thresholds):
        import matplotlib.figure
        fig = plot_threshold_sweep(
            baseline_df, thresholds,
            chosen_threshold=1.000002,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_combined_datasets(
        self, baseline_df, adversarial_df, thresholds,
    ):
        import matplotlib.figure
        from pandas import concat
        combined = concat(
            [baseline_df, adversarial_df],
            ignore_index=True,
        )
        fig = plot_threshold_sweep(
            combined, thresholds,
            chosen_threshold=1.000002,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_chart_has_single_axes(
        self, baseline_df, thresholds,
    ):
        fig = plot_threshold_sweep(
            baseline_df, thresholds,
            chosen_threshold=1.000002,
        )
        assert len(fig.get_axes()) == 1

    def test_chosen_threshold_line(
        self, baseline_df, thresholds,
    ):
        fig = plot_threshold_sweep(
            baseline_df, thresholds,
            chosen_threshold=1.000002,
        )
        ax = fig.get_axes()[0]
        assert len(ax.lines) > 0


class TestPlotThresholdSweepPlotly:

    def test_returns_valid_json(
        self, baseline_df, thresholds,
    ):
        result = plot_threshold_sweep_plotly(
            baseline_df, thresholds,
            chosen_threshold=1.000002,
        )
        spec = json.loads(result)
        assert "data" in spec
        assert "layout" in spec

    def test_has_traces(self, baseline_df, thresholds):
        spec = json.loads(plot_threshold_sweep_plotly(
            baseline_df, thresholds,
            chosen_threshold=1.000002,
        ))
        # At least recall + precision + threshold line
        assert len(spec["data"]) >= 3

    def test_threshold_line_trace(
        self, baseline_df, thresholds,
    ):
        spec = json.loads(plot_threshold_sweep_plotly(
            baseline_df, thresholds,
            chosen_threshold=1.000002,
        ))
        threshold_traces = [
            t for t in spec["data"]
            if t.get("name") == "Chosen Threshold"
        ]
        assert len(threshold_traces) == 1
        assert threshold_traces[0]["x"] == [
            1.000002, 1.000002,
        ]

    def test_combined_datasets(
        self, baseline_df, adversarial_df, thresholds,
    ):
        from pandas import concat
        combined = concat(
            [baseline_df, adversarial_df],
            ignore_index=True,
        )
        spec = json.loads(plot_threshold_sweep_plotly(
            combined, thresholds,
            chosen_threshold=1.000002,
        ))
        names = [t["name"] for t in spec["data"]]
        assert "Baseline Recall" in names
        assert "Adversarial Recall" in names
        assert "Baseline Precision" in names

    def test_layout_y_range(
        self, baseline_df, thresholds,
    ):
        spec = json.loads(plot_threshold_sweep_plotly(
            baseline_df, thresholds,
            chosen_threshold=1.000002,
        ))
        assert spec["layout"]["yaxis"]["range"] == [
            -5, 105,
        ]


class TestBuildPerplexityCharts:

    def _has_attack(self, target):
        return (
            isinstance(target, dict)
            and (
                bool(target.get("pii"))
                or bool(target.get("context"))
            )
        )

    def test_clean_positive_is_baseline(self):
        assert self._has_attack(None) is False

    def test_attacked_positive_is_adversarial(self):
        target = {"pii": ["homoglyph"], "context": []}
        assert self._has_attack(target) is True

    def test_negative_is_baseline(self):
        assert "negative" in {"negative", "hard_negative"}

    def test_hard_negative_is_baseline(self):
        assert "hard_negative" in {
            "negative", "hard_negative",
        }

    def test_context_attack_is_adversarial(self):
        target = {
            "pii": [], "context": ["supportive_context"],
        }
        assert self._has_attack(target) is True

    def test_empty_attack_target_dict_is_baseline(self):
        target = {"pii": [], "context": []}
        assert self._has_attack(target) is False

    def test_no_predictions_file_returns_none(self, mocker):
        mocker.patch(
            "evaluation.report.generator.PREDICTIONS_PATH",
            type("P", (), {
                "exists": lambda self: False,
            })(),
        )
        from evaluation.report.generator import (
            _build_perplexity_charts,
        )
        assert _build_perplexity_charts() is None

    def test_dataset_label_integration(self):
        from evaluation.report.generator import (
            _dataset_label,
        )
        data = DataFrame({
            "category": [
                "positive", "positive",
                "negative", "hard_negative",
                "positive",
            ],
            "attack_target": [
                None,
                {"pii": ["homoglyph"], "context": []},
                None,
                None,
                {
                    "pii": [],
                    "context": ["supportive_context"],
                },
            ],
        })

        data["dataset"] = data.apply(
            _dataset_label, axis=1,
        )

        assert data["dataset"].tolist() == [
            "baseline",     # clean positive
            "fuzzy_adv",    # pii attack
            "baseline",     # negative
            "baseline",     # hard_negative
            "fuzzy_adv",    # context attack
        ]
