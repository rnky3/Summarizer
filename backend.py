from flask import Flask, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy
from tqdm.auto import tqdm

# absolute file path for all the models
bart = 'models/bart-base-booksum'
led = 'models/led-base-book-summary'
t5_l = 'models/long-t5-tglobal-base-16384-book-summary'
t5_s = 'models/t5-small-booksum'
nlp = spacy.load("en_core_web_md")


app = Flask(__name__)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


@app.route('/api/predict', methods=['POST'])
def predict():
    input_text = None
    if request.is_json:
        req = request.get_json()
        input_text = req["input"]
    else:
        return None
    
    input_text_words = input_text.split()
    words_per_chunk = 512
    list_of_text_chunks = chunks(input_text_words, n = words_per_chunk)
    input_text_chunks = [" ".join(t) for t in list_of_text_chunks]

    results = list()
    bart_result = list()
    for text in tqdm(input_text_chunks, total=len(input_text_chunks)):
        bart_result.append(
            bart_summarizer(
                text,
                min_length=16,
                max_length=256,
                no_repeat_ngram_size=3,
                clean_up_tokenization_spaces=True,
                early_stopping=True,
                repetition_penalty=2.1,
                num_beams=4,
            )
        )
    results.append(bart_result)
    led_result = list()
    for text in tqdm(input_text_chunks, total=len(input_text_chunks)):
        led_result.append(
            led_summarizer(
                text,
                min_length=16,
                max_length=256,
                no_repeat_ngram_size=3,
                clean_up_tokenization_spaces=True,
                early_stopping=True,
                repetition_penalty=2.1,
                num_beams=4,
            )
        )
    results.append(led_result)
    t5_l_result = list()
    for text in tqdm(input_text_chunks, total=len(input_text_chunks)):
        t5_l_result.append(
            t5_l_summarizer(
                text,
                min_length=16,
                max_length=256,
                no_repeat_ngram_size=3,
                clean_up_tokenization_spaces=True,
                early_stopping=True,
                repetition_penalty=2.1,
                num_beams=4,
            )
        )
    results.append(t5_l_result)
    t5_l_result = list()
    for text in tqdm(input_text_chunks, total=len(input_text_chunks)):
        t5_l_result.append(
            t5_l_summarizer(
                text,
                min_length=16,
                max_length=256,
                no_repeat_ngram_size=3,
                clean_up_tokenization_spaces=True,
                early_stopping=True,
                repetition_penalty=2.1,
                num_beams=4,
            )
        )
    results.append(t5_l_result)

    formatted_results = list()
    for i in results:
        temp = str()
        for j in i:
            temp = temp + j[0]["summary_text"]
        formatted_results.append(temp)

    vectorized_results = list()
    for summary in formatted_results:
        vectorized_results = nlp(summary).vector

    best_summary = None
    n = 4
    average_similarities = list()

    for i in range(n):
        total_similarity = 0

        for j in range(n):
            if i != j:  # Avoid comparing an element with itself
                u = vectorized_results[i].reshape(1, -1)  # Reshape to a 2D array
                v = vectorized_results[j].reshape(1, -1)
                cos = cosine_similarity(u, v)[0, 0]
                similarity = ((1 - numpy.arccos(cos)) / numpy.pi)
                total_similarity += similarity
        
        average_similarity = total_similarity / 4
        average_similarities.append(average_similarity)
    
    best_index = max(average_similarities)

    for i in range(4):
        if best_index == average_similarities[i]:
            best_summary = formatted_results[i]

    return jsonify(message=best_summary)


if __name__ == '__main__':
    bart_model = AutoModelForSeq2SeqLM.from_pretrained(bart)
    bart_tokenizer = AutoTokenizer.from_pretrained(bart)

    bart_summarizer = pipeline(
        "summarization",
        model=bart_model,
        tokenizer=bart_tokenizer,
    )

    led_model = AutoModelForSeq2SeqLM.from_pretrained(led)
    led_tokenizer = AutoTokenizer.from_pretrained(led)

    led_summarizer = pipeline(
        "summarization",
        model=led_model,
        tokenizer=led_tokenizer,
    )

    t5_l_model = AutoModelForSeq2SeqLM.from_pretrained(t5_l)
    t5_l_tokenizer = AutoTokenizer.from_pretrained(t5_l)

    t5_l_summarizer = pipeline(
        "summarization",
        model=t5_l_model,
        tokenizer=t5_l_tokenizer,
    )

    t5_s_model = AutoModelForSeq2SeqLM.from_pretrained(t5_s)
    t5_s_tokenizer = AutoTokenizer.from_pretrained(t5_s)

    t5_s_summarizer = pipeline(
        "summarization",
        model=t5_s_model,
        tokenizer=t5_s_tokenizer,
    )
    app.run(debug=True)
