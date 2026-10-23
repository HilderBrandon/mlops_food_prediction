import pytest

from src.load_data import load_config, load_data


def test_load_data_quita_columna_sin_nombre(tmp_path):
    csv = tmp_path / "food.csv"
    csv.write_text("Age,Gender,Output,\n20,Female,Yes,Yes\n24,Male,No,No\n", encoding="utf-8")
    df = load_data(csv, load_config())
    assert list(df.columns) == ["Age", "Gender", "Output"]
    assert df.shape == (2, 3)


def test_load_data_falla_sin_target(tmp_path):
    csv = tmp_path / "food.csv"
    csv.write_text("Age,Gender\n20,Female\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_data(csv, load_config())


def test_load_data_archivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "no_existe.csv", load_config())
