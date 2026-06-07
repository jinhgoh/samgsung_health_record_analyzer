from pathlib import Path
import unittest

import pandas as pd

from samsung_health_parser import (
    combine_frames,
    daily_sleep_summary,
    load_health_csv,
    parse_datetime_series,
    prepare_body_records,
    prepare_sleep_records,
    prepare_sleep_stages,
    time_weighted_mean,
    trailing_rate_per_day,
    weight_band_durations,
    weight_range_by_period,
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

    def test_daily_sleep_summary_reports_sleep_in_and_wake_up(self) -> None:
        record = load_health_csv(
            "\n".join(
                [
                    "com.samsung.health.sleep_stage,1,2",
                    "sleep_id,start_time,end_time,stage",
                    "s1,2026-06-01 23:00:00,2026-06-02 00:30:00,40002",
                    "s1,2026-06-02 00:30:00,2026-06-02 06:00:00,40003",
                    "s1,2026-06-02 06:00:00,2026-06-02 07:00:00,40004",
                ]
            ).encode("utf-8"),
            source_name="com.samsung.health.sleep_stage.csv",
        )

        sleep = prepare_sleep_stages(combine_frames([record], "sleep_stage"))
        daily = daily_sleep_summary(sleep)

        self.assertEqual(len(daily), 1)
        row = daily.iloc[0]
        self.assertTrue(bool(row["has_clock_time"]))
        self.assertEqual(row["sleep_in"], "23:00")
        self.assertEqual(row["wake_up"], "07:00")
        self.assertEqual(pd.Timestamp(row["date"]), pd.Timestamp("2026-06-01"))
        self.assertAlmostEqual(float(row["sleep_minutes"]), 480.0, places=1)

    def test_daily_sleep_summary_uses_time_offset_for_utc_epoch_values(self) -> None:
        start_1 = int(pd.Timestamp("2026-06-01 14:00:00", tz="UTC").timestamp() * 1000)
        end_1 = int(pd.Timestamp("2026-06-01 16:00:00", tz="UTC").timestamp() * 1000)
        start_2 = end_1
        end_2 = int(pd.Timestamp("2026-06-01 22:00:00", tz="UTC").timestamp() * 1000)
        record = load_health_csv(
            "\n".join(
                [
                    "com.samsung.health.sleep_stage,1,2",
                    "sleep_id,start_time,end_time,stage,time_offset",
                    f"s1,{start_1},{end_1},40002,UTC+0900",
                    f"s1,{start_2},{end_2},40003,UTC+0900",
                ]
            ).encode("utf-8"),
            source_name="com.samsung.health.sleep_stage.csv",
        )

        sleep = prepare_sleep_stages(combine_frames([record], "sleep_stage"))
        daily = daily_sleep_summary(sleep)

        row = daily.iloc[0]
        self.assertEqual(row["sleep_in"], "23:00")
        self.assertEqual(row["wake_up"], "07:00")
        self.assertEqual(pd.Timestamp(row["date"]), pd.Timestamp("2026-06-01"))
        self.assertAlmostEqual(float(row["sleep_minutes"]), 480.0, places=1)
        self.assertEqual(sleep["hover_start"].iloc[0], "2026-06-01 23:00")

    def test_explicit_timezone_text_keeps_local_wall_time(self) -> None:
        parsed = parse_datetime_series(pd.Series(["2026-06-01T23:00:00+09:00"]))

        self.assertEqual(parsed.iloc[0], pd.Timestamp("2026-06-01 23:00:00"))

    def test_naive_text_with_time_offset_is_read_as_utc(self) -> None:
        # Samsung Health writes naive (no tz suffix) UTC strings plus a separate
        # time_offset, so 14:00 UTC must surface as 23:00 KST.
        parsed = parse_datetime_series(
            pd.Series(["2026-06-01 14:00:00.000"]),
            pd.Series(["UTC+0900"]),
        )

        self.assertEqual(parsed.iloc[0], pd.Timestamp("2026-06-01 23:00:00"))

    def test_naive_text_without_offset_stays_as_written(self) -> None:
        parsed = parse_datetime_series(pd.Series(["2026-06-01 23:00:00"]))

        self.assertEqual(parsed.iloc[0], pd.Timestamp("2026-06-01 23:00:00"))

    def test_daily_sleep_summary_shifts_utc_text_to_local(self) -> None:
        record = load_health_csv(
            "\n".join(
                [
                    "com.samsung.health.sleep_stage,1,2",
                    "sleep_id,start_time,end_time,stage,time_offset",
                    "s1,2026-06-01 14:00:00.000,2026-06-01 17:00:00.000,40002,UTC+0900",
                    "s1,2026-06-01 17:00:00.000,2026-06-01 22:00:00.000,40003,UTC+0900",
                ]
            ).encode("utf-8"),
            source_name="com.samsung.health.sleep_stage.csv",
        )

        sleep = prepare_sleep_stages(combine_frames([record], "sleep_stage"))
        daily = daily_sleep_summary(sleep)

        row = daily.iloc[0]
        self.assertTrue(bool(row["has_clock_time"]))
        self.assertEqual(row["sleep_in"], "23:00")  # 14:00 UTC + 9h
        self.assertEqual(row["wake_up"], "07:00")  # 22:00 UTC + 9h -> next day
        self.assertEqual(pd.Timestamp(row["date"]), pd.Timestamp("2026-06-01"))
        self.assertAlmostEqual(float(row["sleep_minutes"]), 480.0, places=1)
        self.assertEqual(sleep["hover_start"].iloc[0], "2026-06-01 23:00")

    def test_daily_sleep_summary_without_absolute_time(self) -> None:
        records = [load_health_csv(path) for path in Path("sample_dataset").glob("*.csv")]
        sleep = prepare_sleep_stages(combine_frames(records, "sleep_stage"))

        daily = daily_sleep_summary(sleep)

        self.assertGreater(len(daily), 0)
        row = daily.iloc[0]
        self.assertFalse(bool(row["has_clock_time"]))
        self.assertEqual(row["sleep_in"], "")
        self.assertEqual(row["wake_up"], "")
        self.assertTrue(pd.isna(row["date"]))
        self.assertGreater(float(row["sleep_minutes"]), 0)

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

    def test_weight_band_durations_sums_time_per_band(self) -> None:
        frame = pd.DataFrame(
            {
                "record_dt": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-13"]),
                "weight": [81.0, 81.0, 87.0],
            }
        )

        bands = weight_band_durations(frame, "weight", "record_dt", bin_size=5.0)

        # 81->81 midpoint 81 over 2 days lands in 80-85; 81->87 midpoint 84 over 10 days also 80-85.
        self.assertEqual(list(bands["band_label"]), ["80-85 kg"])
        self.assertAlmostEqual(float(bands["days"].iloc[0]), 12.0, places=6)

    def test_weight_band_durations_splits_across_bands(self) -> None:
        frame = pd.DataFrame(
            {
                "record_dt": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-12"]),
                "weight": [72.0, 72.0, 88.0],
            }
        )

        bands = weight_band_durations(frame, "weight", "record_dt", bin_size=5.0)

        durations = dict(zip(bands["band_label"], bands["days"]))
        # 72->72 midpoint 72 over 1 day -> 70-75; 72->88 midpoint 80 over 10 days -> 80-85.
        self.assertAlmostEqual(durations["70-75 kg"], 1.0, places=6)
        self.assertAlmostEqual(durations["80-85 kg"], 10.0, places=6)

    def test_weight_band_durations_needs_two_timestamps(self) -> None:
        frame = pd.DataFrame({"record_dt": pd.to_datetime(["2020-01-01"]), "weight": [80.0]})

        self.assertTrue(weight_band_durations(frame, "weight", "record_dt").empty)

    def test_weight_range_by_period_month_min_max_mean(self) -> None:
        frame = pd.DataFrame(
            {
                "record_dt": pd.to_datetime(["2020-01-05", "2020-01-20", "2020-02-10"]),
                "weight": [80.0, 84.0, 79.0],
            }
        )

        ranges = weight_range_by_period(frame, "weight", "record_dt", freq="M")

        self.assertEqual(len(ranges), 2)
        jan = ranges.iloc[0]
        self.assertEqual(jan["period_start"], pd.Timestamp("2020-01-01"))
        self.assertAlmostEqual(jan["low"], 80.0)
        self.assertAlmostEqual(jan["high"], 84.0)
        self.assertAlmostEqual(jan["mean"], 82.0)
        self.assertEqual(int(jan["count"]), 2)

    def test_weight_range_by_period_year_buckets(self) -> None:
        frame = pd.DataFrame(
            {
                "record_dt": pd.to_datetime(["2019-03-01", "2019-09-01", "2020-06-01"]),
                "weight": [90.0, 86.0, 88.0],
            }
        )

        ranges = weight_range_by_period(frame, "weight", "record_dt", freq="Y")

        self.assertEqual(
            list(ranges["period_start"]),
            [pd.Timestamp("2019-01-01"), pd.Timestamp("2020-01-01")],
        )
        self.assertAlmostEqual(ranges.iloc[0]["low"], 86.0)
        self.assertAlmostEqual(ranges.iloc[0]["high"], 90.0)

    def test_weight_range_by_period_empty_without_time(self) -> None:
        frame = pd.DataFrame({"weight": [80.0, 81.0]})

        self.assertTrue(weight_range_by_period(frame, "weight", "record_dt").empty)

    def test_trailing_rate_per_day_uses_window_endpoints(self) -> None:
        dates = pd.date_range("2020-01-01", periods=31, freq="D")
        frame = pd.DataFrame({"record_dt": dates, "weight": [80.0 + 0.1 * i for i in range(31)]})

        rate = trailing_rate_per_day(frame, "weight", "record_dt", window_days=7)

        # Steady +0.1 kg/day; the trailing 7-day window keeps that slope.
        self.assertAlmostEqual(rate, 0.1, places=6)

    def test_trailing_rate_per_day_window_isolates_recent_slope(self) -> None:
        # Flat at 80 kg until a 4 kg drop concentrated in the final week.
        frame = pd.DataFrame(
            {
                "record_dt": pd.to_datetime(
                    ["2020-01-01", "2020-01-24", "2020-01-28", "2020-01-31"]
                ),
                "weight": [80.0, 80.0, 78.0, 76.0],
            }
        )

        week_rate = trailing_rate_per_day(frame, "weight", "record_dt", window_days=7)
        month_rate = trailing_rate_per_day(frame, "weight", "record_dt", window_days=30)

        # Week window (from 01-24): -4 kg over 7 days. Month window (from 01-01): -4 kg over 30 days.
        self.assertAlmostEqual(week_rate, -4 / 7, places=6)
        self.assertAlmostEqual(month_rate, -4 / 30, places=6)
        self.assertLess(week_rate, month_rate)  # recent drop is steeper than the monthly average

    def test_trailing_rate_per_day_insufficient_window(self) -> None:
        frame = pd.DataFrame(
            {"record_dt": pd.to_datetime(["2020-01-01", "2020-03-01"]), "weight": [80.0, 82.0]}
        )

        # Only the last point falls inside a 7-day trailing window.
        self.assertIsNone(trailing_rate_per_day(frame, "weight", "record_dt", window_days=7))

    def test_trailing_rate_per_day_rejects_thin_coverage(self) -> None:
        # Two weigh-ins only ~1.6 days apart inside a 7-day window.
        frame = pd.DataFrame(
            {
                "record_dt": pd.to_datetime(["2020-01-14 18:00", "2020-01-16 08:00"]),
                "weight": [83.9, 82.4],
            }
        )

        # Without a guard a slope is returned; requiring half-week coverage rejects it.
        self.assertIsNotNone(trailing_rate_per_day(frame, "weight", "record_dt", window_days=7))
        self.assertIsNone(
            trailing_rate_per_day(frame, "weight", "record_dt", window_days=7, min_span_days=3.5)
        )


if __name__ == "__main__":
    unittest.main()
