import pandas as pd

from toxic_comments.config import LABEL_COLUMNS, TEXT_COLUMN
from toxic_comments.repositories import InMemoryDatasetRepository, validate_training_data


def test_in_memory_repository_round_trip():
    data = pd.DataFrame(
        {
            TEXT_COLUMN: ["hello", "bad comment"],
            **{label: [0, 1] for label in LABEL_COLUMNS},
        }
    )
    repository = InMemoryDatasetRepository(data)

    loaded = repository.load()
    loaded.loc[0, TEXT_COLUMN] = "changed"

    assert repository.load().loc[0, TEXT_COLUMN] == "hello"


def test_validate_training_data_fills_missing_text_and_labels():
    data = pd.DataFrame(
        {
            TEXT_COLUMN: [None],
            **{label: [None] for label in LABEL_COLUMNS},
        }
    )

    validated = validate_training_data(data)

    assert validated.loc[0, TEXT_COLUMN] == ""
    assert validated[LABEL_COLUMNS].iloc[0].tolist() == [0, 0, 0, 0, 0, 0]
