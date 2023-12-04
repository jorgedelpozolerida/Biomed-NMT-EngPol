from transformers import MBart50Tokenizer, MBartForConditionalGeneration
import datasets
import pandas as pd
import time
from tqdm import tqdm

def main():
    test_data = pd.read_csv("datasets/test_merged.tsv", delimiter="\t")

    device = 'cuda:0'
    model_name = "facebook/mbart-large-50-one-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50Tokenizer.from_pretrained(model_name, src_lang="en_XX")
    model = model.to(device)

    start = time.time()
    predictions = []
    references = []
    for _, row in tqdm(test_data.iterrows()):
        input = row["eng"]
        target = row["pol"]

        model_inputs = tokenizer(input, return_tensors="pt")
        model_inputs = model_inputs.to(device)
        generated_tokens = model.generate(**model_inputs,forced_bos_token_id=tokenizer.lang_code_to_id["pl_PL"])
        prediction = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        predictions.append(prediction)
        references.append([target])

    bleu = datasets.load_metric('sacrebleu')
    bleu.add_batch(predictions=predictions, references=references)
    print(bleu.compute())
    t = time.time() - start
    print(t)

if __name__ == '__main__':
    main()