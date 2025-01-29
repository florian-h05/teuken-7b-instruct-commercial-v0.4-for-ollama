from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import sentencepiece as spm
import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files, try_to_load_from_cache
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import TOKENIZER_CONFIG_FILE


REPO_ID = "openGPT-X/Teuken-7B-instruct-commercial-v0.4"

class HFGPTXTokenizer(PreTrainedTokenizer):
    """
    A custom tokenizer class that extends Hugging Face's PreTrainedTokenizer.
    It is specifically designed to work with SentencePiece models and integrates
    with Hugging Face's tokenizer utilities.
    """
    
    model_file_glob = "*tokenizer.json"
    vocab_files_names = {"tokenizer_file": "tokenizer.json"}
    decode_kwargs: List[str] = []

    def _encode(self, text: str, return_tokens: bool = False, is_continuation: bool = False):
        """
        Encode a given text using the tokenizer.
        
        Args:
            text (str): The text to encode.
            return_tokens (bool): If True, returns token strings instead of token IDs.
            is_continuation (bool): If True, uses a continuation tokenizer (if available).
        Returns:
            List[int] or List[str]: Encoded text as a list of token IDs or token strings.
        """
        assert self.tok is not None, "No tokenizer is currently loaded"

        # Variant with additional sp processor:
        tokenizer = self.continuation_tokenizer if is_continuation else self.tok

        if return_tokens:
            return tokenizer.encode_as_pieces(text)
        else:
            return tokenizer.encode(text)
    
    def create_list_of_special_tokens(self) -> List[str]:
        """
        Create a list of special tokens, including the BOS, EOS, PAD, EOD tokens,
        and 256 additional placeholder tokens.
        Returns:
            List[str]: List of special tokens.
        """
        return [self.bos_token, self.eos_token, self.pad_token, self.eod_token] + [
            f"<placeholder_tok_{i}>" for i in range(256)
        ]
    
    def find_tokenizer_config(self, config_path: Path, repo_id: str = None) -> Optional[Path]:
        if not os.path.isfile(config_path):
            config_path = try_to_load_from_cache(repo_id=repo_id, filename=Path(config_path).name)
            if not config_path:
                config_path = self._download_config_from_hub(repo_id=repo_id)

        return config_path

    
    def instantiate_from_file_or_name(self, model_file_or_name: str, repo_id: str = None):
        """
        Load the tokenizer model from a file or download it from a repository.

        Args:
            model_file_or_name (str): Path to the model file or the model name.
            repo_id (str, optional): Repository ID from which to download the model file.

        Returns:
            spm.SentencePieceProcessor: Loaded SentencePieceProcessor instance.

        Raises:
            ValueError: If repo_id is not provided when model_file_or_name is not a file.
            OSError: If the model file cannot be loaded or downloaded.
        """
        if not os.path.isfile(model_file_or_name):
            model_file_or_name = try_to_load_from_cache(repo_id=repo_id, filename=Path(model_file_or_name).name)
            if not model_file_or_name:
                model_file_or_name = self._download_model_from_hub(repo_id=repo_id)

        try:
            return spm.SentencePieceProcessor(model_file=model_file_or_name)
        except Exception as e:
            raise OSError(f"Failed to load tokenizer model: {str(e)}")

    def _download_model_from_hub(self, repo_id: str) -> Optional[str]:
        try:
            # List all files in the repo
            repo_files = list_repo_files(repo_id)

            # Find the tokenizer model file
            tokenizer_files = [f for f in repo_files if f.endswith('.model')]
            if not tokenizer_files:
                raise FileNotFoundError(f"No .model file found in repository {repo_id}")

            # Use the first .model file found
            model_file = tokenizer_files[0]
            print(f"Found tokenizer model file: {model_file}")

            # Download the file
            model_file_or_name = hf_hub_download(repo_id=repo_id, filename=model_file)
            print(f"Downloaded tokenizer model to: {model_file_or_name}")
        except Exception as e:
            raise OSError(f"Failed to download tokenizer model: {str(e)}")

        return model_file_or_name

    def _download_config_from_hub(self, repo_id: str):
        if repo_id is None:
            raise ValueError("repo_id must be provided if config_path is not a local file")

        try:
            # List all files in the repo
            repo_files = list_repo_files(repo_id)

            # Find the tokenizer config file
            tokenizer_files = [f for f in repo_files if f.endswith('tokenizer_config.json')]
            if not tokenizer_files:
                raise FileNotFoundError(f"No tokenizer_config.json file found in repository {repo_id}")

            # Use the first tokenizer_config.json file found
            tokenizer_config_file = tokenizer_files[0]
            print(f"Found tokenizer config file: {tokenizer_config_file}")

            # Download the file
            tokenizer_config_file_or_name = hf_hub_download(repo_id=repo_id, filename=tokenizer_config_file)
            print(f"Downloaded tokenizer config file to: {tokenizer_config_file_or_name}")
            return tokenizer_config_file_or_name
        except Exception as e:
            raise OSError(f"Failed to download tokenizer model: {str(e)}")    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the tokenizer.
        Args:
            model_path (Optional[str]): Path to the tokenizer model file.
            config_path (Optional[str]): Path to the tokenizer configuration file.
            **kwargs: Additional keyword arguments passed to the superclass.
        This method also ensures backward compatibility by setting
        `clean_up_tokenization_spaces` to False by default.
        """
        # Prevent cleanup of tokenization spaces to maintain backward compatibility
        self.clean_up_tokenization_spaces = kwargs.setdefault("clean_up_tokenization_spaces", False)
        self.vocab = None
        cp_path = kwargs.get("name_or_path", ".")
        if model_path is None:
            model_path = str(Path(cp_path) / self.vocab_files_names["tokenizer_file"])
        self.tok = self.instantiate_from_file_or_name(model_path, repo_id=REPO_ID)

        super().__init__(**kwargs)

        # Specify special tokens which we know the value of.
        # EOD from `tok` is used as what is called EOS in HuggingFace.
        # Since there is no corresponding mapping for EOS from `tok` in
        # HuggingFace, it is treated as an additional special token.
        # Same for all other special tokens.
        
        
        self.unk_token = "<unk>"
        self.eos_token = "</s>"
        self.bos_token = "<s>"
        self.pad_token = "<pad>"
        self.eod_token = "<eod>"
        
        self.additional_special_tokens = self.create_list_of_special_tokens()
    
        if config_path is None:
            config_path = str(Path(cp_path) / TOKENIZER_CONFIG_FILE)

        if os.path.isfile(config_path):
            self.tokenizer_config = self.load_json(Path(config_path))
        else: # Load from repo
            self.tokenizer_config = self.load_json(Path(self.find_tokenizer_config(Path(config_path), repo_id=REPO_ID)))

    @property
    def vocab_size(self) -> int:
        """
        Get the size of the tokenizer vocabulary.
        Returns:
            int: The size of the vocabulary.
        """
        return self.tok.GetPieceSize()

    def get_vocab(self) -> Dict[str, int]:
        """
        Get the vocabulary as a dictionary mapping token strings to their IDs.
        Returns:
            Dict[str, int]: Vocabulary mapping.
        """
        if self.vocab is None:
            self.vocab = {self.tok.IdToPiece(i): i for i in range(self.vocab_size)}
        return self.vocab

    def _tokenize(self, text: str, **kwargs) -> List[int]:
        """
        Tokenize the input text.
        Args:
            text (str): Text to tokenize.
            **kwargs: Additional keyword arguments.
        Returns:
            List[int]: List of token IDs.
        """
        return_tokens = kwargs.pop("return_tokens", True)
        return self._encode(text, return_tokens=return_tokens, **kwargs)

    def _convert_token_to_id(self, token: str) -> int:
        """
        Convert a token string to its corresponding ID.
        Args:
            token (str): The token to convert.
        Returns:
            int: The token's ID.
        Raises:
            ValueError: If the token is unknown and cannot be encoded to a single ID.
        """
        return self.tok.PieceToId(token)


    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        num_threads: Optional[int] = None,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        """
        Decode a list of token IDs into a string.
        Args:
            token_ids (Union[List[int], List[List[int]]]): List of token IDs or lists of token IDs.
            num_threads (Optional[int]): Number of threads to use for decoding.
        Returns:
            str: Decoded string.
        """
        if isinstance(token_ids, torch.Tensor):  # For PyTorch tensors
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, np.ndarray):  # For NumPy arrays
            token_ids = token_ids.tolist()
        
        output = self.tok.decode(input=token_ids, num_threads=num_threads)
        if skip_special_tokens:
            for substring in self.additional_special_tokens:
                output = output.replace(substring, "")
        
        if clean_up_tokenization_spaces:
            warnings.warn(
                "when cleaning up tokenization spaces, this will not behave "
                "like the original `GPTXTokenizer`., Please supply "
                "`clean_up_tokenization_spaces=False` for decoding."
            )
            output = self.clean_up_tokenization(output)
        
        return output

    
    def _convert_id_to_token(self, index: int) -> str:
        """
        Convert a token ID to its corresponding token string.
        Args:
            index (int): Token ID.
        Returns:
            str: Corresponding token string.
        """
        return self.tok.IdToPiece(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert a list of tokens into a single string.
        Args:
            tokens (List[str]): List of token strings.
        Returns:
            str: Concatenated string of tokens.
        """
        return self.tok.DecodePieces(tokens)

    def _tok_decode(self, token_ids: List[int], **kwargs: Any) -> str:
        """
        Internal method to decode token IDs with additional arguments.
        Args:
            token_ids (List[int]): List of token IDs.
            **kwargs: Additional arguments to pass to the decode method.
        Returns:
            str: Decoded string.
        This method also issues a warning if unsupported arguments are provided.
        """
        passed_kwargs = {key: value for (key, value) in kwargs.items() if key in self.decode_kwargs}
        if len(passed_kwargs) != len(kwargs):
            warnings.warn("silently ignoring some arguments to `decode` due to missing " "support from the tokenizer.")
        text = self.decode(token_ids, **passed_kwargs)
        return text
    
    def save_tokenizer(self, save_dir: str) -> None:
        if not os.path.isdir(save_dir):
            print(f"Vocabulary path ({save_dir}) should be a directory")
            return
        out_vocab_file = os.path.join(save_dir, "tokenizer.model")

        # if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
        #     copyfile(self.vocab_file, out_vocab_file)
        # elif not os.path.isfile(self.vocab_file):
        with open(out_vocab_file, "wb") as f:
            content_spiece_model = self.tok.serialized_model_proto()
            f.write(content_spiece_model)

        return (out_vocab_file,)
        
    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = True,
        **kwargs: Any,
    ) -> str:
        text = self._tok_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            **kwargs,
        )

        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            warnings.warn(
                "when cleaning up tokenization spaces, this will not behave "
                "like the original `GPTXTokenizer`., Please supply "
                "`clean_up_tokenization_spaces=False` for decoding."
            )
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text
        
    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        filename_prefix = filename_prefix + "-" if filename_prefix else ""
        save_directory = Path(save_directory)

        self._save_tokenizer_config(save_directory, filename_prefix)
        tokenizer_file_path = self._save_tokenizer(save_directory, filename_prefix)

        return (tokenizer_file_path,)

    def _save_tokenizer_config(
        self,
        save_directory: Path,
        filename_prefix: str,
    ) -> str:
        self.save_tokenizer_config(save_directory)
        old_tokenizer_config_path = save_directory / TOKENIZER_CONFIG_FILE
        assert old_tokenizer_config_path.is_file(), "tokenizer config path changed"
        new_tokenizer_config_path = save_directory / (filename_prefix + old_tokenizer_config_path.name)
        old_tokenizer_config_path.replace(new_tokenizer_config_path)
        return str(new_tokenizer_config_path)

    def _find_tokenizer_files(self, save_directory: Path) -> List[Path]:
        files = list(Path(save_directory).glob(self.model_file_glob))
        return files

    def _get_tokenizer_file(self, files: List[Path]):
        assert files, "no saved tokenizer file found"
        assert len(files) <= 1, "cannot handle multiple saved tokenizer files"
        return files[0]
    
    def _save_tokenizer(
        self,
        save_directory: Path,
        filename_prefix: str,
    ) -> str:
        self.save_tokenizer(str(save_directory))
        tokenizer_files = self._find_tokenizer_files(save_directory)
        old_tokenizer_file_path = self._get_tokenizer_file(tokenizer_files)
        assert old_tokenizer_file_path.is_file(), "could not access saved tokenizer file"
        new_tokenizer_file_path = save_directory / (filename_prefix + self.vocab_files_names["tokenizer_file"])
        old_tokenizer_file_path.replace(new_tokenizer_file_path)
        return str(new_tokenizer_file_path)
    
    def save_tokenizer_config(self, save_dir: Path) -> None:
        # convert Path to str
        for k in self.tokenizer_config:
            if isinstance(self.tokenizer_config[k], Path):
                self.tokenizer_config[k] = str(self.tokenizer_config[k])

        info_file = save_dir / "tokenizer_config.json"
        with info_file.open("w") as f:
            json.dump(self.tokenizer_config, f, indent=4)
            
    def load_json(self, path: Path) -> dict:
        with path.open("r") as f:
            return json.load(f)
        
class SPTokenizer(HFGPTXTokenizer):
    model_file_glob = "*tokenizer.model"
    vocab_files_names = {"tokenizer_file": "tokenizer.model"}
    decode_kwargs = ["num_threads"]
    # `is_continuation` does not work without this, but it doesn't
    # implement all APIs of `PreTrainedTokenizer`.
    def encode(self, text: str, **kwargs) -> List[int]:
        return_tokens = kwargs.pop('return_tokens', False)
        is_continuation = kwargs.pop('is_continuation', False)
        return self._encode(
            text,
            return_tokens=return_tokens,
            is_continuation=is_continuation,
        )
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eos_token = "</s>"
        self.eos_token_id = 2
        self.system_messages_by_lang = {  # translations by deepl / google translate
                "BG": "Чат между човек и асистент с изкуствен интелект. Асистентът дава полезни и учтиви отговори на въпросите на човека.",  # noqa
                "CS": "Chat mezi člověkem a asistentem s umělou inteligencí. Asistent poskytuje vstřícné a zdvořilé odpovědi na otázky člověka.",  # noqa
                "DA": "En chat mellem et menneske og en assistent med kunstig intelligens, som giver hjælpsomme og høflige svar på menneskets spørgsmål.",  # noqa
                "DE": "Ein Gespräch zwischen einem Menschen und einem Assistenten mit künstlicher Intelligenz. Der Assistent gibt hilfreiche und höfliche Antworten auf die Fragen des Menschen.",  # noqa
                "EL": "Μια συνομιλία μεταξύ ενός ανθρώπου και ενός βοηθού τεχνητής νοημοσύνης. Ο βοηθός δίνει χρήσιμες και ευγενικές απαντήσεις στις ερωτήσεις του ανθρώπου.",  # noqa
                "EN": "A chat between a human and an artificial intelligence assistant.The assistant gives helpful and polite answers to the human's questions.",  # noqa
                "ES": "Una conversación entre un humano y un asistente de inteligencia artificial. El asistente da respuestas útiles y amables a las preguntas del humano.",  # noqa
                "ET": "Inimese ja tehisintellekti assistendi vaheline vestlus. Assistent annab inimese küsimustele abivalmis ja viisakaid vastuseid.",  # noqa
                "FI": "Ihmisen ja tekoälyavustajan välinen keskustelu. Avustaja antaa avuliaita ja kohteliaita vastauksia ihmisen kysymyksiin.",  # noqa
                "FR": "Conversation entre un humain et un assistant doté d'une intelligence artificielle. L'assistant donne des réponses utiles et polies aux questions de l'homme.",  # noqa
                "GA": "Comhrá idir duine agus cúntóir hintleachta saorga. Tugann an cúntóir freagraí cabhracha dea-bhéasacha ar cheisteanna an duine.",  # noqa
                "HR": "Razgovor između čovjeka i pomoćnika umjetne inteligencije. Pomoćnik daje korisne i ljubazne odgovore na ljudska pitanja.",  # noqa
                "HU": "Egy ember és egy mesterséges intelligencia asszisztens közötti beszélgetés. Az asszisztens segítőkész és udvarias válaszokat ad az ember kérdéseire.",  # noqa
                "IT": "Una chat tra un umano e un assistente di intelligenza artificiale. L'assistente fornisce risposte utili ed educate alle domande dell'uomo.",  # noqa
                "LT": "Žmogaus ir dirbtinio intelekto asistento pokalbis. Asistentas naudingai ir mandagiai atsako į žmogaus klausimus.",  # noqa
                "LV": "Cilvēka un mākslīgā intelekta asistenta tērzēšana. Asistents sniedz noderīgas un pieklājīgas atbildes uz cilvēka jautājumiem.",  # noqa
                "MT": "Chat bejn bniedem u assistent ta' intelliġenza artifiċjali. L-assistent jagħti tweġibiet ta' għajnuna u edukat għall-mistoqsijiet tal-bniedem.",  # noqa
                "NL": "Een chat tussen een mens en een assistent met kunstmatige intelligentie. De assistent geeft behulpzame en beleefde antwoorden op de vragen van de mens.",  # noqa
                "PL": "Czat między człowiekiem a asystentem sztucznej inteligencji. Asystent udziela pomocnych i uprzejmych odpowiedzi na pytania człowieka.",  # noqa
                "PT": "Uma conversa entre um ser humano e um assistente de inteligência artificial. O assistente dá respostas úteis e educadas às perguntas do utilizador.",  # noqa
                "RO": "O conversație între un om și un asistent cu inteligență artificială. Asistentul oferă răspunsuri utile și politicoase la întrebările omului.",  # noqa
                "SK": "Rozhovor medzi človekom a asistentom s umelou inteligenciou. Asistent poskytuje užitočné a zdvorilé odpovede na otázky človeka.",  # noqa
                "SL": "Pogovor med človekom in pomočnikom z umetno inteligenco. Pomočnik človeku prijazno in vljudno odgovarja na njegova vprašanja.",  # noqa
                "SV": "En chatt mellan en människa och en assistent med artificiell intelligens. Assistenten ger hjälpsamma och artiga svar på människans frågor.",  # noqa
        }
        chat_template = "{%- for message in messages %}\n{%- if (message['role']|lower == 'user') != (loop.index0 % 2 == 0) %}\n{{- raise_exception('Roles must alternate User/Assistant/User/Assistant/...') }}\n{%- endif %}\n{%-if message['role']|lower == 'user' %}\n{{- message['role']|capitalize + ': ' + message['content'] + '\\n' }}\n{%- elif message['role']|lower == 'assistant' %}\n{{- message['role']|capitalize + ': ' + message['content'] + eos_token + '\\n' }}\n{%- else %}\n{{- raise_exception('Only user and assistant roles are supported!') }}\n {%- endif %}\n{%- endfor %}{%-if add_generation_prompt %}\n{{- 'Assistant: '}}\n{%- endif %}\n"
        self.chat_template = {
            lang: f"System: {sys_msg}" + "{{- '\\n'}}\n" + chat_template
            for lang, sys_msg in self.system_messages_by_lang.items()
        }
        self.chat_template['default'] = f"System: {self.system_messages_by_lang['EN']}" + "{{- '\\n'}}\n" + chat_template