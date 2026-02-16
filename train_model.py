import torch
class Train():
    def __init__(self, config):
        self.model = AceStepConditionGenerationModel(config)
        self.model.train()
        self.model = self.model.to("cuda")
        self.model = self.model.bfloat16()
        self.audio_embed = AudioEmbedder() #embed audio 

    def train(text_tokens: torch.Tensor,
            text_mask: torch.Tensor,
            audio_feats: torch.Tensor,
            audio_mask: torch.Tensor,
            loss_mask: torch.Tensor,
            labels: torch.Tensor):
        pass    