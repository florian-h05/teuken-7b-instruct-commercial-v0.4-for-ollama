# Deploy to [Ollama.com](https://ollama.com/models)

## Quantizing a model

See [Ollama Docs: Import: Quantizing a Model](https://github.com/ollama/ollama/blob/main/docs/import.md#quantizing-a-model).

## Pushing a model to ollama.com

See [Ollama Docs: Import: Sharing your model on ollama.com](https://github.com/ollama/ollama/blob/main/docs/import.md#sharing-your-model-on-ollamacom).

To push multiple different quantizations, set the tag of the model to push:

```shell
ollama cp mymodel myuser/mymodel:tag
ollama push myuser/mymodel:tag
```

We use the following tags:

- `latest` for `q4_K_M` quantization
- `7b` for `q4_K_M` quantization
- `7b-instruct-fp16` for the original F16 quantization
