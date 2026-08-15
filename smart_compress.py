from smart_compress import smart_expand, LangModel

# Train compressed model
model = LangModel(n_layers=5)
# ... train ...

# Expand to 10 layers
expanded = smart_expand(model, target_layers=10, vocab_size=38)
# ... fine-tune if needed ...
