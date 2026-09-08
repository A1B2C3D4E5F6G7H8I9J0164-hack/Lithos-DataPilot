"""
Robust dataset loader supporting file paths, file-like buffers, and multiple encodings/delimiters.
"""
import io
import os
from typing import Union, BinaryIO, Optional
import pandas as pd


class DataLoader:
    """Loads CSV tabular datasets with automated format and encoding detection."""

    SUPPORTED_ENCODINGS = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]

    @classmethod
    def load_csv(
        cls,
        source: Union[str, BinaryIO, io.BytesIO, io.StringIO],
        filename: Optional[str] = None,
        max_rows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load a CSV into a pandas DataFrame with automatic dialect and encoding detection.

        Args:
            source: Filepath, bytes, or file-like stream.
            filename: Optional filename for error reporting.
            max_rows: Optional limit on rows to load.

        Returns:
            pd.DataFrame: Loaded and sanitized dataset.

        Raises:
            ValueError: If file is empty, unreadable, or not valid tabular CSV.
        """
        if isinstance(source, str):
            if not os.path.exists(source):
                raise ValueError(f"Dataset file not found: {source}")
            if os.path.getsize(source) == 0:
                raise ValueError(f"Dataset file is empty: {source}")
            
            with open(source, "rb") as f:
                content = f.read()
        elif hasattr(source, "read"):
            content = source.read()
            if isinstance(content, str):
                content = content.encode("utf-8")
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        if not content or len(content.strip()) == 0:
            raise ValueError("Provided dataset buffer is empty (0 bytes).")

        # Attempt decoding across fallback encodings
        decoded_text = None
        used_encoding = None
        for enc in cls.SUPPORTED_ENCODINGS:
            try:
                decoded_text = content.decode(enc)
                used_encoding = enc
                break
            except UnicodeDecodeError:
                continue

        if decoded_text is None:
            raise ValueError("Unable to decode CSV file using supported encodings (UTF-8, Latin-1, CP1252).")

        # Sniff delimiter
        sample_lines = decoded_text.strip().split("\n")[:10]
        sample_text = "\n".join(sample_lines)
        
        delimiter = ","
        candidate_delimiters = [",", "\t", ";", "|"]
        delimiter_counts = {delim: sample_text.count(delim) for delim in candidate_delimiters}
        best_delim = max(delimiter_counts, key=delimiter_counts.get)
        if delimiter_counts[best_delim] > 0:
            delimiter = best_delim

        try:
            df = pd.read_csv(
                io.StringIO(decoded_text),
                sep=delimiter,
                nrows=max_rows,
                low_memory=False
            )
        except Exception as e:
            raise ValueError(f"Failed to parse CSV data: {str(e)}") from e

        if df.empty:
            raise ValueError("Parsed dataset contains 0 rows.")

        # Sanitize column names
        df.columns = [str(c).strip() for c in df.columns]
        
        # Drop entirely empty unnamed columns
        unnamed_cols = [c for c in df.columns if c.startswith("Unnamed:") and df[c].isna().all()]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)

        return df
