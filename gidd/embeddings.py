import torch
import transformers

def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging"""
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class TextEmbedder:
    """
    A class to embed a batch of texts using an embedding model (e.g., NVEmbed).
    The output embedding dimension is available as self.cond_dim.
    """

    def __init__(self, model_name, device="cuda"):
        """
        Args:
            model_name: Name or path of the embedding model (e.g., "NVEmbed", "sentence-transformers/all-MiniLM-L6-v2").
            device: Device to run the model on.
        """
        self.device = device

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = transformers.AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.model.eval()

        dummy_text = ["dummy"]
        outputs = self.__call__(dummy_text)
        self.cond_dim = outputs.shape[-1]

    @torch.no_grad()
    def __call__(self, texts):
        """
        Args:
            texts: List of strings (batch of texts).
        Returns:
            embeddings: torch.Tensor of shape (batch_size, embedding_dim)
        """
        return self._compute_embeddings(texts)

    def _compute_embeddings(self, texts):
        """
        Compute embeddings for a batch of texts using the underlying model.
        Args:
            texts: List of strings.
        Returns:
            List or tensor of embeddings.
        """
        # Try to use .encode() method first (for sentence-transformers models)
        if hasattr(self.model, 'encode'):
            try:
                return self.model.encode(texts)
            except Exception as e:
                print(f"Failed to use .encode() method: {e}. Falling back to manual computation.")
        
        # Fall back to manual tokenization and mean pooling
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        assert sentence_embeddings.shape[0] == len(texts)
        assert len(sentence_embeddings.shape) == 2

        return sentence_embeddings
    
    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self
