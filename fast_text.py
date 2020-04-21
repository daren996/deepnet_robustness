import fastText

model = fastText.load_model("./.vector_cache/wiki.en.bin")

print(model.get_word_vector('3:26'))
print(model.get('3:26').shape)