import io

import httpx
import polars as pl
import pytest

from ffa.ingest import nflverse


def _client(body: bytes, status: int = 200) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(status, content=body))
    )


def test_csv_dataset_keeps_ids_as_strings():
    csv = (
        b"season,week,player_id,game_id,receptions_exp\n"
        b"2026,1,00-0037744,2026_01_ARI_LAC,7.4\n"
        b"2026,1,NA,x,NA\n"
    )
    df = nflverse.fetch("expected_points", 2026, client=_client(csv))
    assert df.schema["player_id"] == pl.Utf8
    assert df["player_id"].to_list() == ["00-0037744", None]  # "NA" is null, not a string
    assert df["receptions_exp"].to_list() == [7.4, None]


def test_parquet_dataset_and_missing_season():
    buf = io.BytesIO()
    pl.DataFrame({"season": [2026], "week": [1]}).write_parquet(buf)
    assert nflverse.fetch("snap_counts", 2026, client=_client(buf.getvalue())).height == 1
    with pytest.raises(nflverse.NotPublishedError):
        nflverse.fetch("snap_counts", 2099, client=_client(b"", status=404))
