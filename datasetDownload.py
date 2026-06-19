import os

from datasets import load_dataset


def download_data():
    print("1")
    # 1. Скачиваем во временный кэш HF
    dataset = load_dataset("google-research-datasets/go_emotions")
    print("2")

    # 2. Создаем папку в проекте, если её нет
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    print("3")

    # 3. Сохраняем каждую часть (train, validation, test) в CSV-файлы
    for split in dataset.keys():
        dataset[split].to_csv(f"{output_dir}/{split}.csv")

    print(f"Данные успешно сохранены в папку: {output_dir}")
    return dataset


if __name__ == "__main__":
    download_data()
