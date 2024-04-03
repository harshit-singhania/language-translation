from translation import Encoder, Decoder, Seq2Seq, en_vocab, de_vocab, en_nlp, de_nlp, translate
import torch   
import streamlit as st

# Load the model
input_dim = len(de_vocab)
output_dim = len(en_vocab)
encoder_embedding_dim = 256
decoder_embedding_dim = 256
hidden_dim = 512
n_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5

encoder = Encoder(
    input_dim,
    encoder_embedding_dim,
    hidden_dim,
    n_layers,
    encoder_dropout,
)

decoder = Decoder(
    output_dim,
    decoder_embedding_dim,
    hidden_dim,
    n_layers,
    decoder_dropout,
)

lower = True 
sos_token = "<sos>"
eos_token = "<eos>"
max_output_length = 25
model = Seq2Seq(encoder, decoder, device="cpu")
model.eval()
model.load_state_dict(torch.load("translation.pt", map_location=torch.device("cpu")))

def translate_to_english(german_text): 
    translation = translate(
        german_text,
        model,
        en_nlp,
        de_nlp,
        en_vocab,
        de_vocab,
        lower,
        sos_token,
        eos_token,
        device="cpu",
    )

    # remove eos and sos token
    translation = translation[1:-1]
    # remove <unk> and <pad> 
    translation = [word for word in translation if word != "<unk>" and word != "<pad>"]
    english_text = ' '.join(translation) 
    return english_text

# sentence = input('Enter a sentence to translate: ')
# translation_arb = translate(
#     sentence,
#     model,
#     en_nlp,
#     de_nlp,
#     en_vocab,
#     de_vocab,
#     lower,
#     sos_token,
#     eos_token,
#     device="cpu",
# )

# print(translation_arb)

st.set_page_config(page_title="German to Englidh Translation", page_icon="🌍")

st.title("German to English Translation")
sentence = st.text_input("Enter a sentence to translate")
if st.button("Translate"):
    translated = translate_to_english(sentence)
    st.write(translated)

