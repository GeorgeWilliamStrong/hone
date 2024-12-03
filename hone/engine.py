from typing import Union, List
import os
import platformdirs
from textgrad.engine.base import EngineLM, CachedEngine
from instill.clients import init_pipeline_client


class ChatInstill(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str = "gpt-4o",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        namespace_id: str = None,
        pipeline_id: str = None,
        **kwargs
    ):
        """Initialize an Instill chat engine for TextGrad.

        Args:
            model_string (str): The model identifier (e.g. "gpt-4o-mini")
            system_prompt (str): Default system prompt for the model
            namespace_id (str): Instill namespace ID (can also be set via INSTILL_NAMESPACE_ID env var)
            pipeline_id (str): Instill pipeline ID (can also be set via INSTILL_PIPELINE_ID env var)
        """
        # Remove "instill-" prefix if present
        if model_string.startswith("instill-"):
            model_string = model_string.replace("instill-", "")

        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_instill_{model_string}.db")
        CachedEngine.__init__(self, cache_path=cache_path)

        self.model_string = model_string
        self.system_prompt = system_prompt
        self.namespace_id = namespace_id or os.getenv("INSTILL_NAMESPACE_ID")
        self.pipeline_id = pipeline_id or os.getenv("INSTILL_PIPELINE_ID")

        if not self.namespace_id or not self.pipeline_id:
            raise ValueError("namespace_id and pipeline_id must be provided either through constructor or environment variables")

        self.pipeline = init_pipeline_client(api_token=os.environ["INSTILL_API_TOKEN"])
        self.is_multimodal = False

    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt: str = None, **kwargs):
        """Generate text using the Instill pipeline.

        Args:
            content: Either a string prompt or a list of strings/bytes for multimodal input
            system_prompt: Optional system prompt to override the default
            **kwargs: Additional arguments passed to _generate_from_single_prompt
        """
        if isinstance(content, str):
            return self._generate_from_single_prompt(content, system_prompt=system_prompt, **kwargs)
        else:
            raise NotImplementedError("Multimodal input not supported for ChatInstill")

    def _generate_from_single_prompt(
        self, prompt: str, system_prompt: str = None, temperature=0, max_tokens=2000, top_p=0.99
    ):
        """Generate text from a single prompt using the Instill pipeline.

        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt override
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
        """
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        pipeline_response = self.pipeline.trigger(
            namespace_id=self.namespace_id,
            pipeline_id=self.pipeline_id,
            data=[{
                "input": prompt,
                "max-tokens": max_tokens,
                "model": self.model_string,
                "system-message": sys_prompt_arg,
                "temperature": temperature,
                "top-p": top_p
            }]
        )

        response_text = pipeline_response.outputs[0].fields['llm-output'].string_value
        self._save_cache(sys_prompt_arg + prompt, response_text)
        return response_text

    def __call__(self, prompt, **kwargs):
        """Convenience method to allow using the engine as a callable."""
        return self.generate(prompt, **kwargs)
