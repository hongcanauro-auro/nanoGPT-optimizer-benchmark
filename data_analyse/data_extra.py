import re
import csv


def extract_loss_values(log_text):

    pattern = r'iter \d+: loss ([\d.]+)'

    loss_values = []

    lines = log_text.split('\n')

    for line in lines:
        match = re.search(pattern, line)
        if match:
            loss_value = float(match.group(1))
            loss_values.append(loss_value)

    return loss_values


def save_to_csv(loss_values, filename='loss_values.csv'):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(['loss'])

        for loss in loss_values:
            writer.writerow([loss])


# 主程序
if __name__ == "__main__":

    with open('gpt_adamw.txt', 'r') as f:
         log_content = f.read()

    loss_values = extract_loss_values(log_content)

    save_to_csv(loss_values)

    print(f"成功提取了 {len(loss_values)} 个loss值")
    print("CSV文件已保存为 'loss_values.csv'")