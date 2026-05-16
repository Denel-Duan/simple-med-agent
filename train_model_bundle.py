import os
import sys

import pandas as pd

from app import (
    DEFAULT_DB_PATHS,
    MODEL_BUNDLE_PATH,
    choose_final_sheet,
    find_local_db_path,
    load_dataframe_from_path,
    save_model_bundle,
    train_real_ml_models,
)


def _find_training_data() -> str:
    path = find_local_db_path()
    if path:
        return path
    for candidate in DEFAULT_DB_PATHS:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("未在 data/ 目录或默认路径中找到可训练的 Excel/CSV 数据文件。")


def _load_training_dataframe(path: str) -> pd.DataFrame:
    if path.lower().endswith(".xlsx"):
        sheet_names = pd.ExcelFile(path).sheet_names
        sheet_name = choose_final_sheet(sheet_names)
        print(f"读取数据：{path} / sheet={sheet_name}")
        return load_dataframe_from_path(path, sheet_name=sheet_name)
    print(f"读取数据：{path}")
    return load_dataframe_from_path(path)


def main() -> int:
    try:
        data_path = _find_training_data()
        df = _load_training_dataframe(data_path)
        print(f"训练数据行数：{len(df)}，字段数：{len(df.columns)}")

        bundle = train_real_ml_models(df)
        save_model_bundle(bundle)

        models = bundle.get("models", {})
        metrics = bundle.get("metrics", bundle.get("summary"))
        print("\n训练目标列表：")
        for target in models.keys():
            print(f"- {target}")

        print("\n模型评价指标：")
        if isinstance(metrics, pd.DataFrame) and not metrics.empty:
            print(metrics.to_string(index=False))
        else:
            print("无可显示指标。")

        print(f"\n模型已保存：{MODEL_BUNDLE_PATH}")
        return 0
    except Exception as exc:
        print(f"训练失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
