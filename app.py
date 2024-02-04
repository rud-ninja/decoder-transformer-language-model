import torch
from tokenizers import Tokenizer
from packages.llm_architecture import BigramLanguageModel
from flask import Flask, request, render_template, Response, stream_with_context
import time

app = Flask(__name__)

vocab_size = 500
# tokenizer = Tokenizer.from_file(r"C:\\Users\\hp\\Downloads\\secondapp\\llm_bpe.tokenizer.json")
# model = BigramLanguageModel(vocab_size)
# model.load_state_dict(torch.load(r"C:\\Users\\hp\\Downloads\\secondapp\\trained_lang_model.pt", map_location=torch.device("cpu")))
# model.eval()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/stream', methods=['GET', 'POST'])
def stream():
    tokenizer = Tokenizer.from_file(r"C:\\Users\\hp\\Downloads\\secondapp\\llm_bpe.tokenizer.json")
    model = BigramLanguageModel(vocab_size)
    model.load_state_dict(torch.load(r"C:\\Users\\hp\\Downloads\\secondapp\\trained_lang_model.pt", map_location=torch.device("cpu")))
    model.eval()

    user_input = request.args.get('userInput', '')
    max_tokens = request.args.get('numericInput', '')
    context = torch.tensor(tokenizer.encode(user_input).ids, dtype=torch.long).view(1, -1)

    return Response(model.generate(context, tokenizer, max_new_tokens=int(max_tokens)), content_type='text/event-stream')

if __name__ == "__main__":
    app.run(debug=True)