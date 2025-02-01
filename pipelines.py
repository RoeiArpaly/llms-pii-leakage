from pandas import DataFrame

from data_generator import presidio_inject_pii
from data_validators import (
    contain_pii_template,
    presidio_pii_analyzer,
)
from evaluation import spans_scorer
from models import (
    adversarial_content_generator,
    AdversarialContent,
    generate_llm_input,
    llm_pii_detector,
    pii_fuzzer_v2,
    pii_fuzzer_type,
)


def llm_input_generation(contains_pii: bool):

    llm_input = generate_llm_input(contains_pii)

    contains_pii = contain_pii_template(llm_input)
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
    data = data[data["valid_sample"] == True]
    return data


def llm_detector(data: DataFrame):

    cols = [
        "llm_input",
        "pii_spans_generator",
        "pii_spans_llm_detector",
        "pii_amount_llm_detector",
        "spans_score",
    ]

    data["pii_spans_llm_detector"] = data["llm_input"].apply(
        llm_pii_detector, mode="spans"
    )
    data["pii_amount_llm_detector"] = data["pii_spans_llm_detector"].apply(len)

    data["spans_score"] = data.apply(
        lambda row: spans_scorer(
            row["pii_spans_generator"], row["pii_spans_llm_detector"]
        ),
        axis=1,
    )
    return data[cols]


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
    cols = [
        "llm_input",
        "llm_input_template",
        "pii_spans_generator",
        "fuzzy_techniques",
        f"{prefix}_llm_input",
        f"{prefix}_analyzer",
        f"{prefix}_llm_restored",
        f"{prefix}_llm_restored_analyzer",
        f"{prefix}_pii_amount_analyzer",
        f"{prefix}_pii_amount_llm_restored_analyzer",
        "spans_score",
    ]

    data["fuzzy_techniques"] = data.apply(
        lambda row: pii_fuzzer_type() if row["llm_input_template"] else None,
        axis=1,
    )
    data[f"{prefix}_llm_input"] = data.apply(
        lambda row: pii_fuzzer_v2(
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
    return data[cols]


def fuzzy_pii_adv_content_generation(data: DataFrame):
    prefix = "fuzzy_adv_content"

    data[f"{prefix}_llm_input"] = data.apply(
        lambda row: adversarial_content_generator(
            llm_input=row["llm_input"],
            spans=row["pii_spans_generator"],
            adv_content=AdversarialContent.ThisIsMyLuckyNumber,
            prefix=True,
        )
        if row["llm_input_template"]
        else None,
        axis=1,
    ).apply(pii_fuzzer_v2)
    data = data.drop(columns=["llm_input_template", "pii_spans_generator"])

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
