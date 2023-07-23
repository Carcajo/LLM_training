import torch
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', is_decoder=True)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

contexts = [
"Interesting fact about movies",
"Tell me an interesting movie fact",
"Can you share an interesting detail about a popular movie?"
]

facts = [
"In the movie The <mask> the characters Neo and Trinity never kiss.",
"Alfred Hitchcock made a cameo appearance in almost every one of his <mask>, including Rear Window, Psycho, The Birds.",
...
]
encoded_data = [tokenizer.encode(f"{c} [SEP] {f}", return_tensors='pt')
                for c, f in zip(contexts, facts)]

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
for epoch in range(10):
    for input_ids in encoded_data:
        outputs = model(input_ids)
        loss = torch.mean(outputs[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


context = tokenizer.encode("Interesting fact about movies:", return_tensors='pt')
generated = model.generate(context, max_length=100)


text = tokenizer.decode(generated[0])
sep_idx = text.find('[SEP]')
if sep_idx != -1:
  text = text[sep_idx+6:]
print(text)
