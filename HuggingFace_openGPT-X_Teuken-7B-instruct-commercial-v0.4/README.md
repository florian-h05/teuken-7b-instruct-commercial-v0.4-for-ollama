---
language:
- de
- bg
- cs
- da
- el
- en
- es
- et
- fi
- fr
- ga
- hr
- hu
- it
- lt
- lv
- mt
- nl
- pl
- pt
- ro
- sl
- sv
- sk
metrics:
- accuracy
- bleu
pipeline_tag: text-generation
library_name: transformers
base_model:
- openGPT-X/Teuken-7B-base-v0.4
license: apache-2.0
---
# Model Card for Teuken-7B-instruct-commercial-v0.4


[Teuken-7B-instruct-commercial-v0.4](https://huggingface.co/openGPT-X/Teuken-7B-instruct-commercial-v0.4) is an instruction-tuned 7B parameter multilingual large language model (LLM) pre-trained with 4T tokens in all official 24 European languages and released under Apache 2.0 in the research project [OpenGPT-X](https://opengpt-x.de). 
The base model Teuken-7B-base-v0.4 is available on request 📧 <a href="contact@opengpt-x.de">contact@opengpt-x.de</a>.

### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** Fraunhofer, Forschungszentrum Jülich, TU Dresden, DFKI
- **Funded by:** German Federal Ministry of Economics and Climate Protection (BMWK) in the context of the OpenGPT-X project
- **Model type:** Transformer based decoder-only model
- **Language(s) (NLP):** bg, cs, da, de, el, en, es, et, fi, fr, ga, hr, hu, it, lt, lv, mt, nl, pl, pt, ro, sk, sl, sv
- **Shared by:** OpenGPT-X

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->
[Teuken-7B-instruct-commercial-v0.4](https://huggingface.co/openGPT-X/Teuken-7B-instruct-commercial-v0.4) is intended for commercial and research use in all official 24 European languages. Since [Teuken-7B-instruct-commercial-v0.4](https://huggingface.co/openGPT-X/Teuken-7B-instruct-commercial-v0.4) focuses on covering all 24 EU languages, it renders more stable results across these languages and better reflects European values in its answers than English-centric models. It is therefore specialized for use in multilingual tasks.

## Disclaimer Toxic Content:
 
This Large Language Model (LLM) may generate content that is inappropriate, offensive, or harmful. While the dataset has been filtered to minimize such outputs, the model may still produce text that is biased or toxic due to the large scale and diverse nature of the data.


### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

The model is not intended for use in math and coding tasks.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

[Teuken-7B-instruct-commercial-v0.4](https://huggingface.co/openGPT-X/Teuken-7B-instruct-commercial-v0.4) is an instruction-tuned version of Teuken-7B-base-v0.4 (which is available on request 📧 <a href="contact@opengpt-x.de">contact@opengpt-x.de</a>) that is not completely free from biases and hallucinations.

## How to Get Started with the Model

## Usage
The model requires a few libraries that can be installed in your python environment:


```bash
python -m pip install numpy torch huggingface_hub transformers sentencepiece
```

After installation, here's an example of how to use the model:

As this model is a fine-tuned model, it must be used with the provided prompt template. Using the model without the prompt template is not intended and is not recommended. The prompt template is defined as follows:
```python
user="Hi!"
lang_code = "DE"
system_messages={
            "EN": "A chat between a human and an artificial intelligence assistant."
            " The assistant gives helpful and polite answers to the human's questions.",
            "DE": "Ein Gespräch zwischen einem Menschen und einem Assistenten mit künstlicher Intelligenz."
            " Der Assistent gibt hilfreiche und höfliche Antworten auf die Fragen des Menschen.",
        }
 
prompt = f"System: {system_messages[lang_code]}\nUser: {user}\nAssistant:"
```

The prompt template is also directly integrated in the Tokenizer and can be used as follows:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "openGPT-X/Teuken-7B-instruct-commercial-v0.4"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model = model.to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False,
    trust_remote_code=True,
)
messages = [{"role": "User", "content": "Wer bist du?"}]
prompt_ids = tokenizer.apply_chat_template(messages, chat_template="DE", tokenize=True, add_generation_prompt=True, return_tensors="pt")
prediction = model.generate(
    prompt_ids.to(model.device),
    max_length=512,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
    num_return_sequences=1,
)
prediction_text = tokenizer.decode(prediction[0].tolist())
print(prediction_text)
```

This example demonstrates how to load the model and tokenizer, prepare input, generate text, and print the result.

### Usage with vLLM Server
Starting the vLLM Server:
``` shell
vllm serve openGPT-X/Teuken-7B-instruct-commercial-v0.4 --trust-remote-code
```
Use Chat API with vLLM and pass the language of the Chat-Template as extra body:
``` python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)
completion = client.chat.completions.create(
    model="openGPT-X/Teuken-7B-instruct-commercial-v0.4",
    messages=[{"role": "User", "content": "Hallo"}],
    extra_body={"chat_template":"DE"}
)
print(f"Assistant: {completion]")
```
The default language of the Chat-Template can also be set when starting the vLLM Server. For this create a new file with the name `lang` and the content `DE` and start the vLLM Server as follows:
``` shell
vllm serve openGPT-X/Teuken-7B-instruct-commercial-v0.4 --trust-remote-code --chat-template lang
```

### Usage with vLLM Offline Batched Inference
``` python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.01, max_tokens=1024, stop=["</s>"])
llm = LLM(model="openGPT-X/Teuken-7B-instruct-commercial-v0.4", trust_remote_code=True, dtype="bfloat16") 
outputs = llm.chat(
    messages=[{"role": "User", "content": "Hallo"}], 
    sampling_params=sampling_params, 
    chat_template="DE"
)
print(f"Prompt: {outputs[0].prompt}")
print(f"Assistant: {outputs[0].outputs[0].text}")
```

## Training Details

### Pre-Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

Teuken-7B-base-v0.4 was pre-trained on 4 trillion tokens of data from publicly available sources. 
The pretraining data has a cutoff of September 2023.
More information is available in our preprint ["Data Processing for the OpenGPT-X Model Family"](http://arxiv.org/abs/2410.08800).


### Instruction-Tuning Data
The model was fine-tuned on a collection of English- and German-focused instruction-tuning datasets which also contains instructions for 22 official European languages
The dataset composition contains three types of data: multilingual data, English data, and translated German data

#### English data
* We only included a subsample of the OpenOrca dataset.
* To select instruction-tuning examples based on their quality, We calculated the reward scores of all English examples utilizing [Starling-RM-7B-alpha](https://huggingface.co/berkeley-nest/Starling-RM-7B-alpha) (Apache-2.0 license)

We aim to include roughly the same amount of English examples as we have multilingual examples:
  1. Add all multi-turn examples
  2. Add entire `code_alpaca` dataset subset
  4. For the remaining dataset subsets (`open_orca`, `evol_instruct_143k`, `evol_instruct_70k`, `sharegpt_v3`, `ultrachat_200k`), we add the samples with the highest reward scores so that each dataset subset contributes an equal amount of high-quality examples

##### German Data
As we aim for a German- and English-centric, European language dataset and due to the sparsity of large-scale German instruction-tuning data, we translated the English portion of the above-described dataset composition. For this, we applied the [Alma-13B](https://huggingface.co/haoranxu/ALMA-13B) (MIT license) model. As code can be a problematic case for translation, we implemented a regex-based code detection functionality. With it, we exclude code snippets from translation and insert the code snippets after translation again.
As the `alpaca_code` contains many code snippets not detectable by our regex-based code detection implementation, we included this part of the dataset from the translation.

#### Multilingual data
For multilingual data we include the 14 offical European languages contained in the [aya_dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset) and the 21 offical European languages contained in the `translated_flan_cot` dataset of the [aya_collection](https://huggingface.co/datasets/CohereForAI/aya_collection/viewer/translated_flan_cot).

#### Datasets and Licenses

| Name                                                                                                                   | Language     | License                                                                                                           |
| :--------------------------------------------------------------------------------------------------------------------- | :----------- | :---------------------------------------------------------------------------------------------------------------- |
| [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca)                                               | EN           | MIT                                                                                                               |
| [sahil2801/CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)                                   | EN           | CC-BY-4.0                                                                                                         |
| [WizardLM/WizardLM_evol_instruct_V2_196k](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k)     | EN           | MIT                                                                                                               |
| [WizardLM/WizardLM_evol_instruct_70k](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_70k)             | EN           | MIT                                                                                                               |
| [anon8231489123/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) | EN           | Apache-2.0                                                                                                        |
| [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)                           | EN           | MIT                                                                                                               |
| [CohereForAI/aya_dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset)                                     | Multilingual | Apache-2.0                                                                                                        |
| [CohereForAI/aya_collection](https://huggingface.co/datasets/CohereForAI/aya_collection)                               | Multilingual | Apache-2.0                                                                                                        |
| [FreedomIntelligence/sharegpt-deutsch](https://huggingface.co/datasets/FreedomIntelligence/sharegpt-deutsch)           | DE           | Apache-2.0                                                                                                        |
| [bjoernp/ultrachat_de](https://huggingface.co/datasets/bjoernp/ultrachat_de)                                           | DE           | MIT                                                                                                               |



Dataset contribution per language:

|    |   total |   de_freedomintelligence_sharegpt |   de_ultrachat_de |   translated_flan_cot |   aya_dataset |   ultrachat_200k_translated_to_de |   sharegpt_v3_unfiltered_translated_to_de |   evol_instruct_143k_translated_to_de |   evol_instruct_70k_translated_to_de |   open_orca_translated_to_de |   ultrachat_200k |   sharegpt_v3_unfiltered |   code_alpaca |   open_orca |   evol_instruct_143k |   evol_instruct_70k |
|:---|--------:|----------------------------------:|------------------:|----------------------:|--------------:|----------------------------------:|------------------------------------------:|--------------------------------------:|-------------------------------------:|-----------------------------:|-----------------:|-------------------------:|--------------:|------------:|---------------------:|--------------------:|
| BG |    1909 |                                 0 |                 0 |                  1909 |             0 |                                 0 |                                         0 |                                     0 |                                    0 |                            0 |                0 |                        0 |             0 |           0 |                    0 |                   0 |
| CS |    1885 |                                 0 |                 0 |                  1885 |             0 |                                 0 |                                         0 |                                     0 |                                    0 |                            0 |                0 |                        0 |             0 |           0 |                    0 |                   0 |
| DA |    2001 |                                 0 |                 0 |                  1906 |            95 |                                 0 |                                         0 |                                     0 |                                    0 |                            0 |                0 |                        0 |             0 |           0 |                    0 |                   0 |
| DE |   77628 |                              5818 |               898 |                  1896 |           231 |                              6940 |                                     37555 |                                  8116 |                                 8065 |                         8109 |                0 |                        0 |             0 |           0 |                    0 |                   0 |
| ET |    1901 |                                 0 |                 0 |                  1901 |             0 |                                 0 |                                         0 |                                     0 |                                    0 |                            0 |                0 |                        0 |             0 |           0 |                    0 |                   0 |
| EL |    2472 |                                 0 |                 0 |                  1881 |           591 |                                 0 |                                         0 |                                     0 |                                    0 |                            0 |                0 |                        0 |             0 |           0 |                    0 |                   0 |
| ES |    3800 |                                 0 |                 0 |                  1898 |          1902 |                                 0 |                                         0 |                                     0 |                                    0 |                            0 |                0 |                        0 |             0 |           0 |                    0 |                   0 |
| EN |   80806 |                                 0 |                 0 |                     0 |             0 |                                 0 |                                         0 |                                     0 |                                    0 |                            0 |             6915 |                    37600 |         12013 |        8074 |                 8099 |                8105 |
| FI |    2598 |                                 0 |                 0 |                  1890 |           708 |                                 0 |                                         0 |                                     0 |                                    0 |                            0 |                0 |                        0 |             0 |           0 |                    0 |                   0 |
| FR |    3250 |                                 0 |                 0 |                  1890 |          1360 |                                 0 |                                         0 |                                     0 |                                    0 |                            0 |                0 |                        0 |             0 |           0 |                    0 |                   0 |
| HU |    1985 |                                 0 |                 0 |                  1892 |            93 |                                 0 |                                         0 |                                     0 |                                    0 |                            0 |                0 |                        0 |             0 |           0 |                    0 |                   0 |
| MT |    1918 |                                 0 |                 0 |                  1918 |             0 |                                 0 |                                         0 |                                     0 |                                    0 |                            0 |                0 |                        0 |             0 |           0 |                    0 |                   0 |
| IT |    2613 |                                 0 |                 0 |                  1910 |           703 |                                 0 |                                         0 |                                     0 |                                    0 |                            0 |                0 |                        0 |             0 |           0 |                    0 |                   0 |
| LT |    2800 |                                 0 |                 0 |                  1920 |           880 |                                 0 |                                         0 |                                     0 |                                    0 |                            0 |                0 |                        0 |             0 |           0 |                    0 |                   0 |
| NL |    3549 |                                 0 |                 0 |                  1905 |          1644 |                                 0 |                                         0 |                                     0 |                                    0 |                            0 |                0 |                        0 |             0 |           0 |                    0 |                   0 |
| PL |    3322 |                                 0 |                 0 |                  1909 |          1413 |                                 0 |                                         0 |                                     0 |                                    0 |                            0 |                0 |                        0 |             0 |           0 |                    0 |                   0 |
| PT |    3806 |                                 0 |                 0 |                  1897 |          1909 |                                 0 |                                         0 |                                     0 |                                    0 |                            0 |                0 |                        0 |             0 |           0 |                    0 |                   0 |
| RO |    1888 |                                 0 |                 0 |                  1888 |             0 |                                 0 |                                         0 |                                     0 |                                    0 |                            0 |                0 |                        0 |             0 |           0 |                    0 |                   0 |
| GA |    3069 |                                 0 |                 0 |                  1880 |          1189 |                                 0 |                                         0 |                                     0 |                                    0 |                            0 |                0 |                        0 |             0 |           0 |                    0 |                   0 |
| SK |    1922 |                                 0 |                 0 |                  1922 |             0 |                                 0 |                                         0 |                                     0 |                                    0 |                            0 |                0 |                        0 |             0 |           0 |                    0 |                   0 |
| SL |    1894 |                                 0 |                 0 |                  1894 |             0 |                                 0 |                                         0 |                                     0 |                                    0 |                            0 |                0 |                        0 |             0 |           0 |                    0 |                   0 |
| SV |    3160 |                                 0 |                 0 |                  1916 |          1244 |                                 0 |                                         0 |                                     0 |                                    0 |                            0 |                0 |                        0 |             0 |           0 |                    0 |                   0 |

Total across languages 210,176


### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->
Instruction fined tuned version of [Teuken-7B-base-v0.4](https://huggingface.co/openGPT-X/Teuken-7B-base-v0.4).
More information regarding the pre-training are available in our model preprint ["Teuken-7B-Base & Teuken-7B-Instruct: Towards European LLMs"](https://arxiv.org/abs/2410.03730).

#### Training Hyperparameters

- **Training regime:** bf16 mixed precision <!--fp32, fp16 mixed precision, , bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

Results on multilingual benchmarks for 21 European languages with instruction-tuned models
| Model                          | Avg.   | EU21-ARC | EU21-HeSw | EU21-TQA | EU21-MMLU |
|--------------------------------|--------|----------|-----------|----------|-----------|
| Meta-Llama-3.1-8B-Instruct     | **.563** | .563   | .579      | .532     | **.576**  |
| Mistral-7B-Instruct-v0.3       | .527   | .530     | .538      | **.548** | .491   |
| Salamandra-7B-Instruct         | .543   | **.595** | **.637**  | .482     | .459      |
| Aya-23-8B                      | .485   | .475     | .535      | .476     | .455      |
| Occiglot-7B-eu5-Instruct       | .475   | .484     | .519      | .471     | .428      |
| Pharia-1-LLM-7B-C-A            | .417   | .396     | .438      | .469     | .366      |
| Bloomz-7B1                     | .358   | .316     | .354      | .461     | .302      |
| **Teuken-7B-instruct-commercial-v0.4**            | .531   | .569     | .620      | .503     | .430      |

More information regarding the quality of our translated benchmarks are available in our Evaluation preprint ["Towards Multilingual LLM Evaluation for European Languages"](https://arxiv.org/abs/2410.08928).
More evaluation results regarding Teuken-7B-instruct-research-v0.4 are available in our model preprint  ["Teuken-7B-Base & Teuken-7B-Instruct: Towards European LLMs"](https://arxiv.org/abs/2410.03730).

The model was evaluated in 21 languages on ARC, GSM8K, HellaSwag, TruthfulQA, Translation and MMLU. Results can also be seen in the [European LLM Leaderboard](https://huggingface.co/spaces/openGPT-X/european-llm-leaderboard).

## Technical Specifications

### Model Architecture and Objective

| Hyper-Parameter            | Value    |
|----------------------------|----------|
| Training Objective         | CLM      |
| Activation Function        | SwiGLU   |
| Seq Length                 | 4096     |
| Position Embeddings        | Rotary   |
| Num Layers                 | 32       |
| Hidden Size                | 4096     |
| FFN Hidden Size            | 13440    |
| Num Attention Heads        | 32       |
| Head Dim                   | 128      |
| Group Query Attention      | yes      |
| Num Query Groups           | 2        |
| Normalization              | RMSNorm  |
| Learning rate              | 3e-4     |
| Min learning rate          | 3e-5     |
| Disable bias in linear     | yes      |
| Hidden dropout             | 0.0      |
| Attention dropout          | 0.0      |
| Optimizer                  | AdamW    |
| Beta1                      | 0.9      |
| Beta2                      | 0.95     |
| Data-type                  | bf16     |
| Recompute-activations      | yes      |
| Distributed-optimizers     | yes      |

### Compute Infrastructure

We trained our models on JUWELS Booster which consists of 936 compute nodes, each equipped with 4 NVIDIA A100 GPUs. The GPUs are hosted by AMD EPYC Rome CPUs. The compute nodes are connected with HDR-200 InfiniBand in a DragonFly+ topology. 

#### Hardware

The configuration of JUWELS Booster compute nodes is the following:

    CPU: AMD EPYC 7402 processor; 2 sockets, 24 cores per socket, SMT-2 (total: 2×24×2 = 96 threads) in NPS-4 1 configuration
    Memory: 512 GB DDR4-3200 RAM (of which at least 20 GB is taken by the system software stack, including the file system); 256 GB per socket; 8 memory channels per socket (2 channels per NUMA domain)
    GPU: 4 × NVIDIA A100 Tensor Core GPU with 40 GB; connected via NVLink3 to each other
    Network: 4 × Mellanox HDR200 InfiniBand ConnectX 6 (200 Gbit/s each), HCA
    Periphery: CPU, GPU, and network adapter are connected via 2 PCIe Gen 4 switches with 16 PCIe lanes going to each device (CPU socket: 2×16 lanes). PCIe switches are configured in synthetic mode.
#### Software

[Megatron-LM](https://github.com/OpenGPTX/Megatron-LM)

**BibTeX:**

If you find our model useful in your research, please consider citing our [preprint](https://arxiv.org/abs/2410.03730):
```
@misc{ali2024teuken7bbaseteuken7binstructeuropean,
      title={Teuken-7B-Base & Teuken-7B-Instruct: Towards European LLMs}, 
      author={Mehdi Ali and Michael Fromm and Klaudia Thellmann and Jan Ebert and Alexander Arno Weber and Richard Rutmann and Charvi Jain and Max Lübbering and Daniel Steinigen and Johannes Leveling and Katrin Klug and Jasper Schulze Buschhoff and Lena Jurkschat and Hammam Abdelwahab and Benny Jörg Stein and Karl-Heinz Sylla and Pavel Denisov and Nicolo' Brandizzi and Qasid Saleem and Anirban Bhowmick and Lennard Helmer and Chelsea John and Pedro Ortiz Suarez and Malte Ostendorff and Alex Jude and Lalith Manjunath and Samuel Weinbach and Carolin Penke and Oleg Filatov and Shima Asaadi and Fabio Barth and Rafet Sifa and Fabian Küch and Andreas Herten and René Jäkel and Georg Rehm and Stefan Kesselheim and Joachim Köhler and Nicolas Flores-Herr},
      year={2024},
      eprint={2410.03730},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.03730}, 
}
```

# Team
## Data Team
Anirban Bhowmick (IAIS), Nicolo Brandizzi (IAIS), Lennard Helmer (IAIS), Benny Jörg Stein (IAIS), Karl-Heinz Sylla (IAIS), Pavel Denisov (IAIS), Qasid Saleem (IAIS), Johannes Leveling (IAIS), Hammam Abdelwahab (IAIS), Luzian Hahn (IIS), Farzad Naderi (IIS), Md Saiful Islam (IIS), Alexander Schwirjow (IIS), Pedro Ortiz Suarez (ex. DFKI), Malte Ostendorff (ex. DFKI)
## Model-Training Team
### Core contributors
Mehdi Ali (IAIS), Michael Fromm (IAIS), Jan Ebert (FZJ), Chelsea John (FZJ), Lena Jurkschat (TUD), Alexander Weber (IAIS)
### Contributors:
Richard Rutmann (IAIS), Daniel Steinigen (IAIS), Lalith Manjunath (TUD), Carolin Penke (FZJ)
## Evaluation Team
### Core contributors
Klaudia Thellmann (TUD), Alex Jude (IAIS), Jasper Buschhoff (IAIS)
### Contributors:
Shima Assadi (IIS), Fabio Barth (DFKI)
## Management
Joachim Köhler (IAIS), Nicolas Flores-Herr (IAIS), Stefan Kesselheim (FZJ), Andreas Herten (FZJ), Georg Rehm (DFKI), René Jäkel (TUD), Fabian Küch (IIS), Nicole Hildebrandt (IAIS), Ines Wendler (IAIS)

We believe that collaboration is key to overcome the aforementioned limitations and thereby strengthening the European GenAI landscape. Because of this, the team invites researchers, developers, and AI enthusiasts to join and engage through various platforms. A Discord server has been created for community collaboration, offering a space for discussions on technical details, ideas, and direct interaction with developers. Additionally, resources like research publications and a European LLM Leaderboard provide insights into Teuken-7B’s performance and technical aspects. The OpenGPT-X team encourages ongoing engagement and collaboration as the project evolves.
Key links:
Discord: OpenGPT-X [Discord server](https://discord.com/invite/RvdHpGMvB3)
Research Papers: OpenGPT-X News [Research Papers](https://opengpt-x.de/en/news-en/)
LLM Leaderboard: European LLM Leaderboard [LLM Leaderboard](https://huggingface.co/spaces/openGPT-X/european-llm-leaderboard)

<div class="hf-card">
    <h2>Contact Information</h2>
    <p>You can reach out to the following model card contact:</p>
    <ul>
        <li>
            <a href="https://huggingface.co/openGPT-X" target="_blank">OpenGPT-X</a> 
            - <a href="contact@opengpt-x.de">contact@opengpt-x.de</a>
        </li>
    </ul>
</div>