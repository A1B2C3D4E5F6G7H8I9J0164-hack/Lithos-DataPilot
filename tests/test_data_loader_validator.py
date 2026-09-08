import io
import pytest
import pandas as pd
from app.data.loader import DataLoader
from app.data.validator import DataValidator, ValidationResult
from app.data.profiler import DataProfiler
from app.data.quality import DataQualityAuditor, QualitySeverity


def test_loader_from_string():
    csv_data = "col1,col2,target\n1,a,0\n2,b,1\n3,c,0\n"
    df = DataLoader.load_csv(io.StringIO(csv_data))
    assert len(df) == 3
    assert list(df.columns) == ["col1", "col2", "target"]


def test_loader_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        DataLoader.load_csv(io.StringIO(""))


def test_loader_delimiter_sniffing():
    tsv_data = "colA\tcolB\ttarget\n10\tx\t0\n20\ty\t1\n"
    df = DataLoader.load_csv(io.StringIO(tsv_data))
    assert len(df) == 2
    assert "colA" in df.columns


def test_validator_min_rows():
    # Less than 25 rows should trigger error
    small_df = pd.DataFrame({"a": range(10), "target": [0, 1] * 5})
    res = DataValidator.validate(small_df, target_column="target")
    assert not res.is_valid
    assert any("Minimum 25 rows" in err for err in res.errors)


def test_validator_missing_target():
    df = pd.DataFrame({"a": range(30), "b": range(30)})
    res = DataValidator.validate(df, target_column="non_existent")
    assert not res.is_valid
    assert any("not found" in err for err in res.errors)


def test_validator_single_value_target():
    df = pd.DataFrame({"a": range(30), "target": [1] * 30})
    res = DataValidator.validate(df, target_column="target")
    assert not res.is_valid
    assert any("at least 2 distinct values" in err for err in res.errors)


def test_validator_valid_dataset():
    df = pd.DataFrame({"a": range(35), "b": ["x"] * 35, "target": [0, 1] * 17 + [0]})
    res = DataValidator.validate(df, target_column="target")
    assert res.is_valid
    assert res.row_count == 35


def test_profiler_and_quality_audit():
    df = pd.DataFrame({
        "id_col": [f"ID_{i}" for i in range(50)],
        "const_col": [42] * 50,
        "missing_col": [None if i < 15 else float(i) for i in range(50)],
        "num_col": list(range(50)),
        "target": [0, 1] * 25
    })
    profile = DataProfiler.profile(df)
    assert profile.row_count == 50
    assert "const_col" in profile.constant_columns
    
    report = DataQualityAuditor.audit(df, profile, target_column="target")
    assert report.overall_score < 100
    rule_ids = [issue.rule_id for issue in report.issues]
    assert "ZERO_VARIANCE" in rule_ids
