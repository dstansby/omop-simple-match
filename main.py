import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import bm25s
import inquirer
import pandas as pd
import pooch


def load_concepts(
    *,
    domain_ids: list[str] | None = None,
    vocabulary_ids: list[str] | None = None,
    concept_class_ids: list[str] | None = None,
    only_standard_concepts: bool = False,
) -> pd.DataFrame:
    """
    Load OMOP concepts.

    Parameters
    ----------
    domain_ids, vocabularly_ids, concept_class_ids :
        Provide a list of valid domain/vocab/concept IDs to filter the vocabulary on.
        By default no filtering is done.
    only_standard_concepts :
        If *True*, only return standard concepts.
    """
    print("Loading concepts...")
    concept_path = pooch.retrieve(
        url="https://github.com/SAFEHR-data/omop-vocabs-processed/raw/refs/tags/v20260227/data/concept.parquet",
        known_hash="md5:da01db95f561a1edaa78b5dce037eb47",
        progressbar=True,
    )
    concepts = pd.read_parquet(concept_path)
    print("Concepts loaded 🎉")

    if domain_ids is not None:
        domain_filter = concepts["domain_id"].isin(domain_ids)
        if domain_filter.sum() == 0:
            raise ValueError(
                f"Zero rows left after filtering on {domain_ids=}\n"
                f"Valid domain ids are {concepts['domain_id'].unique().tolist()}"
            )
        concepts = concepts[concepts["domain_id"].isin(domain_ids)]
    if vocabulary_ids is not None:
        concepts = concepts[concepts["vocabulary_id"].isin(vocabulary_ids)]
    if concept_class_ids is not None:
        concepts = concepts[concepts["concept_class_id"].isin(concept_class_ids)]
    if only_standard_concepts:
        concepts = concepts[concepts["standard_concept"] == "S"]
    return concepts


def create_retriever(
    *,
    omop_concepts: pd.DataFrame,
    mutate_concept_id: Callable[[str], str] | None = None,
) -> bm25s.BM25:
    """
    Create text retriever for OMOP concept names.

    Parameters
    ----------
    omop_concepts:
        OMOP concept dataframe.
    mutate_concept_id:
        Optional function that can be used to mutate the OMOP concept names
        before they're fed into the retriever. This can be useful to massage
        the OMOP concept names into forms that are more likely to match the
        values being searched.

    """
    concept_names: list[str] = omop_concepts["concept_name"].tolist()
    if mutate_concept_id:
        concept_names = [mutate_concept_id(c) for c in concept_names]
    corpus_tokens = bm25s.tokenize(concept_names, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    return retriever


def main(
    *,
    csv_path: Path,
    id_col_name: str,
    source_value_col_name: str,
    extra_to_add: dict[str, Any] | None = None,
    mutate_concept_id: Callable[[str], str] | None = None,
):

    concepts = load_concepts(
        domain_ids=["Regimen"], vocabulary_ids=["HemOnc"], only_standard_concepts=True
    )

    retriever = create_retriever(
        omop_concepts=concepts, mutate_concept_id=mutate_concept_id
    )

    df = pd.read_csv(csv_path)
    df[id_col_name] = df[id_col_name].fillna(value=0).astype(int)
    df_new = df.copy()
    df_new.to_csv(csv_path, index=False)

    for idx, row in df.iterrows():
        if row[id_col_name] != 0:
            continue

        source_value = str(row[source_value_col_name])
        query_tokens = bm25s.tokenize(source_value)
        result_idxs, _ = retriever.retrieve(query_tokens, k=5)
        result_idxs = result_idxs[0, :]
        possible_names = concepts["concept_name"].iloc[result_idxs].values.tolist()

        none_found = "None of the above"
        questions = [
            inquirer.List(
                "concept_name",
                message=f"{source_value} 👈 source value",
                choices=possible_names + [none_found],
                carousel=True,
            ),
        ]

        answer = inquirer.prompt(questions)
        if answer is None:
            sys.exit()
        answer = answer["concept_name"]
        if answer == none_found:
            continue

        answer_idx = int(result_idxs[possible_names.index(answer)])
        concept_id = int(concepts["concept_id"].iloc[answer_idx])

        df_new.loc[idx, id_col_name] = concept_id
        if extra_to_add is not None:
            for k, v in extra_to_add.items():
                df_new.loc[idx, k] = v
        print(f"🌳 Assigning concept ID {concept_id} to '{source_value}'")
        print()
        df_new.to_csv(csv_path, index=False)


if __name__ == "__main__":
    # When feeding in names to the corpus, strip 'monotherapy'. This is present in OMOP,
    # but not in Caboodle, so hopefully that should help matching
    strip_monotherapy = lambda s: s.lower().replace("monotherapy").strip()
    csv_path = Path(
        r"C:\Users\stansbyd\software\omop_es\mapping\CUH\local_mappings\epic\chemo_protocols.csv"
    )
    main(
        csv_path=csv_path,
        id_col_name="episode_object_concept_id",
        source_value_col_name="ProtocolName",
        extra_to_add={"quality_control_scale": "L2_HUMAN_REVIEW"},
        mutate_concept_id=strip_monotherapy,
    )
