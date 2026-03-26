import pandas as pd
import tempfile
import os


def process_loss_log(file_path):

    try:
        df = pd.read_csv(file_path)

        rows_to_delete = []
        i = 0
        while i < len(df) - 1:
            if df.iloc[i]['iteration'] == df.iloc[i + 1]['iteration']:
                rows_to_delete.append(i + 1)
                i += 1
            i += 1

        df_cleaned = df.drop(rows_to_delete).reset_index(drop=True)

        df_cleaned.to_csv(file_path, index=False)

        print(f"处理完成！删除了 {len(rows_to_delete)} 行重复数据")
        print(f"原文件行数: {len(df)}, 处理后行数: {len(df_cleaned)}")

    except Exception as e:
        print(f"处理文件时出错: {e}")

if __name__ == "__main__":
    file_path = "loss_log.csv"
    process_loss_log(file_path)