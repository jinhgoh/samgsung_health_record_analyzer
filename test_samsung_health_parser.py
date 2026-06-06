from pathlib import Path
import unittest

import pandas as pd

from samsung_health_parser import (
    combine_frames,
    load_health_csv,
    prepare_body_records,
    prepare_sleep_records,
    prepare_sleep_stages,
    time_weighted_mean,
)


class SleepParserTests(unittest.TestCase):
    def test_sample_sleep_stage_records_are_prepared(self) -> None:
        records = [load_health_csv(path) for path in Path("sample_dataset").glob("*.csv")]

        sleep = prepare_sleep_stages(combine_frames(records, "sleep_stage"))

        self.assertGreater(len(sleep), 0)
        self.assertIn("duration_min", sleep.columns)

    def test_regular_sleep_record_is_prepared(self) -> None:
        record = load_health_csv(
            "\n".join(
                [
                    "com.samsung.health.sleep,1,2",
                    "start_time,end_time,sleep_score,datauuid",
                    "2026-06-01 23:00:00,2026-06-02 06:30:00,82,abc",
                ]
            ).encode("utf-8"),
            source_name="com.samsung.health.sleep.csv",
        )

        sleep = prepare_sleep_records(combine_frames([record], "sleep"))

        self.assertEqual(record.record_type, "sleep")
        self.assertEqual(len(sleep), 1)
        self.assertEqual(round(float(sleep["duration_min"].iloc[0]), 1), 450.0)

    def test_generic_stage_header_infers_sleep_stage(self) -> None:
        record = load_health_csv(
            "start_time,end_time,stage\n00:00:00,00:30:00,40002\n".encode("utf-8"),
            source_name="uploaded.csv",
        )

        self.assertEqual(record.record_type, "sleep_stage")

    def test_weight_records_drop_sparse_rows_without_weight(self) -> None:
        frame = pd.DataFrame(
            {
                "start_time": ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"],
                "weight": ["80.0", pd.NA, "79.5", pd.NA],
                "total_body_water": [pd.NA, "42.0", "41.8", "42.2"],
            }
        )

        body = prepare_body_records(frame)

        self.assertEqual(len(body), 2)
        self.assertTrue(body["weight"].notna().all())

    def test_generic_weight_header_infers_weight(self) -> None:
        record = load_health_csv(
            "start_time,weight\n2026-06-01,80.0\n".encode("utf-8"),
            source_name="uploaded.csv",
        )

        self.assertEqual(record.record_type, "weight")

    def test_sample3_weight_records_are_prepared(self) -> None:
        path = Path("sample_dataset/sample3_com.samsung.health.weight.20260606182543.csv")
        self.assertTrue(path.exists())
        record = load_health_csv(path)

        body = prepare_body_records(combine_frames([record], "weight"))
        expected_weight_rows = record.frame["weight"].dropna().shape[0]

        self.assertEqual(record.record_type, "weight")
        self.assertEqual(len(body), expected_weight_rows)
        self.assertGreater(len(body), 0)
        self.assertTrue(body["weight"].notna().all())
        self.assertTrue(body["record_dt"].notna().all())

    def test_sample3_trailing_empty_cells_do_not_shift_columns(self) -> None:
        path = Path("sample_dataset/sample3_com.samsung.health.weight.20260606182543.csv")
        self.assertTrue(path.exists())
        record = load_health_csv(path)

        self.assertTrue(record.frame["pkg_name"].dropna().eq("com.sec.android.app.shealth").all())
        self.assertTrue(
            record.frame["datauuid"]
            .dropna()
            .str.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
            .all()
        )
        self.assertTrue(record.frame["vfa_level"].dropna().empty)
        self.assertTrue(record.frame["total_body_water"].dropna().empty)

    def test_unquoted_commas_in_comment_do_not_shift_columns(self) -> None:
        record = load_health_csv(
            "\n".join(
                [
                    "com.samsung.health.weight,1,2,,,,,",
                    "start_time,height,weight,comment,pkg_name,datauuid,total_body_water",
                    "2026-06-01,177,80.0,meal,gym,note,com.sec.android.app.shealth,00000000-0000-0000-0000-000000000001,42.0",
                ]
            ).encode("utf-8"),
            source_name="weight.csv",
        )

        row = record.frame.iloc[0]

        self.assertEqual(row["comment"], "meal,gym,note")
        self.assertEqual(row["pkg_name"], "com.sec.android.app.shealth")
        self.assertEqual(row["datauuid"], "00000000-0000-0000-0000-000000000001")
        self.assertEqual(row["total_body_water"], "42.0")

    def test_unquoted_commas_in_custom_do_not_shift_columns(self) -> None:
        record = load_health_csv(
            "\n".join(
                [
                    "com.samsung.health.weight,1,2,,,,",
                    "start_time,custom,height,weight,pkg_name,datauuid",
                    "2026-06-01,{meal:true,source:manual},177,80.0,com.sec.android.app.shealth,00000000-0000-0000-0000-000000000002",
                ]
            ).encode("utf-8"),
            source_name="weight.csv",
        )

        row = record.frame.iloc[0]

        self.assertEqual(row["custom"], "{meal:true,source:manual}")
        self.assertEqual(row["height"], "177")
        self.assertEqual(row["weight"], "80.0")
        self.assertEqual(row["pkg_name"], "com.sec.android.app.shealth")


    def test_time_weighted_mean_weights_by_duration(self) -> None:
        frame = pd.DataFrame(
            {
                "record_dt": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-11"]),
                "weight": [80.0, 90.0, 90.0],
            }
        )

        weighted = time_weighted_mean(frame, "weight", "record_dt")

        # Plain mean is 86.67, but 90 kg was held for 9 of the 10 days.
        self.assertAlmostEqual(weighted, 89.5, places=6)
        self.assertNotAlmostEqual(weighted, frame["weight"].mean(), places=2)

    def test_time_weighted_mean_falls_back_without_time(self) -> None:
        frame = pd.DataFrame({"weight": [80.0, 90.0, 100.0]})

        self.assertAlmostEqual(time_weighted_mean(frame, "weight"), 90.0, places=6)

    def test_time_weighted_mean_single_record(self) -> None:
        frame = pd.DataFrame({"record_dt": pd.to_datetime(["2020-01-01"]), "weight": [83.0]})

        self.assertAlmostEqual(time_weighted_mean(frame, "weight", "record_dt"), 83.0, places=6)


if __name__ == "__main__":
    unittest.main()
