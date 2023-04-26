from starwhale import Dataset

if __name__ == "__main__":
    with open('alpaca_data_cleaned_archive.json') as f:
        Dataset.from_json("alpaca_data_cleaned_archive",f.read(),)
    with open('alpaca_data_gpt4.json') as f:
        Dataset.from_json("alpaca_data_gpt4",f.read(),)
    with open('alpaca_data.json') as f:
        Dataset.from_json("alpaca_data",f.read(),)
