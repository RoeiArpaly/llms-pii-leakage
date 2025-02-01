from pandas import DataFrame

from data_generation.pii_generator import presidio_inject_pii
from data_generation.llm_input_generator import generate_llm_input
from data_manipulation.rule_based import (
    pii_fuzzer,
    pii_fuzzer_type,
    adversarial_content,
)
from detectors.llm import llm_pii_detector
from detectors.presidio import presidio_pii_analyzer
from evaluation import spans_scorer


def llm_input_generation(contains_pii: bool):

    results = generate_llm_input(contains_pii)
    llm_input = results["llm_input"]
    contains_pii = results["contains_pii"]

    fake_record = presidio_inject_pii(llm_input) if contains_pii else {"spans": []}

    llm_input_result = fake_record["text"] if contains_pii else llm_input
    analyzer_result = presidio_pii_analyzer(text=llm_input_result)
    valid_sample = all(span in analyzer_result for span in fake_record["spans"])

    return {
        "llm_input": llm_input_result,
        "llm_input_template": llm_input if contains_pii else None,
        "contains_pii_generator": contains_pii,
        "contains_pii_analyzer": bool(analyzer_result),
        "pii_amount_generator": len(fake_record["spans"]),
        "pii_amount_analyzer": len(analyzer_result),
        "pii_spans_generator": fake_record["spans"] if contains_pii else None,
        "pii_spans_analyzer": analyzer_result,
        "valid_sample": valid_sample,
    }


def validate_llm_input_generation_results(data: DataFrame):
    data = data[data["valid_sample"]]
    return data


def llm_detector(data: DataFrame):

    data["pii_spans_llm_detector"] = data["llm_input"].apply(
        llm_pii_detector, mode="spans"
    )
    data["pii_amount_llm_detector"] = data["pii_spans_llm_detector"].apply(len)
    data["spans_score"] = data.apply(
        lambda row: spans_scorer(
            spans_true=row["pii_spans_generator"],
            spans_pred=row["pii_spans_llm_detector"]
        ),
        axis=1,
    )
    return data


def fuzzy_pii_generation(data: DataFrame):
    """

    Parameters
    ----------
    data : DataFrame
        dataset with the following columns:
            - llm_input
            - llm_input_template

    Returns
    -------
    DataFrame
    """

    prefix = "fuzzy"
    data["fuzzy_techniques"] = data.apply(
        lambda row: pii_fuzzer_type() if row["llm_input_template"] else None,
        axis=1,
    )
    data[f"{prefix}_llm_input"] = data.apply(
        lambda row: pii_fuzzer(
            llm_input=row["llm_input"],
            spans=row["pii_spans_generator"],
            chosen_techniques=row["fuzzy_techniques"],
        ),
        axis=1,
    )
    data[f"{prefix}_analyzer"] = data[f"{prefix}_llm_input"].apply(
        presidio_pii_analyzer
    )
    data[f"{prefix}_llm_restored"] = data[f"{prefix}_llm_input"].apply(llm_pii_detector)
    data[f"{prefix}_llm_restored_analyzer"] = data[f"{prefix}_llm_restored"].apply(
        presidio_pii_analyzer
    )
    data[f"{prefix}_pii_amount_analyzer"] = data[f"{prefix}_analyzer"].apply(len)
    data[f"{prefix}_pii_amount_llm_restored_analyzer"] = data[
        f"{prefix}_llm_restored_analyzer"
    ].apply(len)
    data["spans_score"] = data.apply(
        lambda row: spans_scorer(
            row["pii_spans_generator"], row[f"{prefix}_llm_restored_analyzer"]
        ),
        axis=1,
    )
    return data


def fuzzy_pii_adv_content_generation(data: DataFrame):

    prefix = "fuzzy_adv_content"
    data[f"{prefix}_llm_input"] = data.apply(
        lambda row: adversarial_content(
            llm_input=row["fuzzy_llm_input"],
            spans=row["pii_spans_generator"],
            chosen_techniques=row["fuzzy_techniques"],
        ),
        axis=1,
    )

    data[f"{prefix}_analyzer"] = data[f"{prefix}_llm_input"].apply(
        presidio_pii_analyzer
    )
    data[f"{prefix}_llm_restored"] = data[f"{prefix}_llm_input"].apply(llm_pii_detector)
    data[f"{prefix}_llm_restored_analyzer"] = data[f"{prefix}_llm_restored"].apply(
        presidio_pii_analyzer
    )
    data[f"{prefix}_pii_amount_analyzer"] = data[f"{prefix}_analyzer"].apply(len)
    data[f"{prefix}_pii_amount_llm_restored_analyzer"] = data[
        f"{prefix}_llm_restored_analyzer"
    ].apply(len)
    return data
