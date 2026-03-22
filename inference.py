import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import warnings
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel, PreTrainedModel, PretrainedConfig
from pyannote.audio import Model as PyannoteModel

warnings.filterwarnings('ignore')


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class AttentivePooling(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(input_dim, 256), nn.Tanh(), nn.Linear(256, 1))

    def forward(self, x, mask=None):
        attn_weights = self.attention(x).squeeze(-1)
        if mask is not None:
            safe_min = torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights.masked_fill(mask == 0, safe_min)
        attn_weights = F.softmax(attn_weights, dim=-1)
        pooled_emb = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        return pooled_emb


class LanguageIdentificationHead(nn.Module):
    def __init__(self, input_dim: int = 1024, num_languages: int = 35):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_languages)
        )

    def forward(self, embeddings: torch.Tensor, labels=None) -> torch.Tensor:
        # Labels argument kept for compatibility with the forward pass signature, but ignored
        return self.head(embeddings)


class SpeakerIdentificationHead(nn.Module):
    def __init__(self, input_dim: int = 1024, num_speakers: int = 3566):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_speakers)
        )

    def forward(self, x):
        return self.head(x)


class PyannoteResNet34(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.model = PyannoteModel.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")

    def forward(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        backbone = self.model
        if hasattr(backbone, "model"): backbone = backbone.model
        if hasattr(backbone, "resnet"): backbone = backbone.resnet

        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = F.relu(x)
        if hasattr(backbone, "maxpool"): x = backbone.maxpool(x)

        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x_frames = backbone.layer4(x)

        x_flat = x_frames.mean(dim=2)
        mean = torch.mean(x_flat, dim=2)
        std = torch.std(x_flat, dim=2)
        global_emb = torch.cat([mean, std], dim=1)

        return x_flat, global_emb

class FrameLevelAcousticAdapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()

    def forward(self, acoustic_frames, target_seq_len):
        x = acoustic_frames.permute(0, 2, 1)
        x = self.proj(x)
        x = self.norm(x)
        x = self.activation(x)
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, size=target_seq_len, mode='linear', align_corners=False)
        return x.permute(0, 2, 1)

class GatedCrossModalFusion(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=8, batch_first=True, dropout=0.1)
        self.gate_net = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size), nn.Sigmoid())
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, semantic_emb, acoustic_emb, attention_mask=None):
        attn_output, _ = self.cross_attn(query=semantic_emb, key=acoustic_emb, value=acoustic_emb)
        gate_input = torch.cat([semantic_emb, attn_output], dim=-1)
        gate = self.gate_net(gate_input)
        fused = semantic_emb + (gate * attn_output)

        if attention_mask is not None:
            fused = fused * attention_mask.unsqueeze(-1).type_as(fused)
        return self.layer_norm(self.dropout(fused))


class M3LMSpeechEncoderConfig(PretrainedConfig):
    model_type = "M3LMSpeechEncoder"

    def __init__(self, hidden_size: int = 1024, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size



class M3LMSpeechEncoder(PreTrainedModel):
    config_class = M3LMSpeechEncoderConfig

    def __init__(self, config: M3LMSpeechEncoderConfig):
        super().__init__(config)
        self.config = config
        self.semantic_encoder = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
        self.acoustic_encoder = PyannoteResNet34(pretrained=True)
        self.acoustic_adapter = FrameLevelAcousticAdapter(input_dim=256, output_dim=config.hidden_size)
        self.fusion_layer = GatedCrossModalFusion(config.hidden_size)

    def forward(self, features: dict, attention_mask=None, tasks=None):
        input_features = features["input_features"]
        a_input_features = features["acoustic_ip_features"]

        bert_out = self.semantic_encoder(input_features, attention_mask=attention_mask, output_hidden_states=True)
        sem_h = bert_out.last_hidden_state

        x_frames, global_emb = self.acoustic_encoder(a_input_features)
        adapted_acoustic = self.acoustic_adapter(x_frames, target_seq_len=sem_h.shape[1])

        fused_emb = self.fusion_layer(sem_h, adapted_acoustic, attention_mask=attention_mask)

        outputs = {
            'embeddings': fused_emb,
            'semantic_hidden_states': bert_out.hidden_states,
            'embeddings_acoustic_pooled': global_emb
        }
        return outputs


class AdversarialLanguageAdapter(nn.Module):
    def __init__(self, num_languages: int, num_train_speakers: int, alpha: float = 1.0, device="cuda"):
        super().__init__()
        self.device = device
        config = M3LMSpeechEncoderConfig()
        self.encoder = M3LMSpeechEncoder(config).to(device)

        # Frozen Backbone
        self.encoder.eval()
        for param in self.encoder.parameters(): param.requires_grad = False

        # Task Adapters
        self.layer_weights = nn.Parameter(torch.zeros(24)).to(device)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024, nhead=8, dim_feedforward=2048, dropout=0.1, activation='gelu', batch_first=True
        )
        self.task_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1).to(device)
        self.residual_scale = nn.Parameter(torch.zeros(1)).to(device)

        self.pooler = AttentivePooling(input_dim=1024).to(device)
        self.lang_head = LanguageIdentificationHead(input_dim=1024, num_languages=num_languages).to(device)
        self.grl = GradientReversalLayer(alpha=alpha).to(device)
        self.spk_head = SpeakerIdentificationHead(input_dim=1024, num_speakers=num_train_speakers).to(device)

    def forward(self, input_features, acoustic_ip_features, attention_mask, lang_ids=None):
        with torch.no_grad():
            features = {"input_features": input_features, "acoustic_ip_features": acoustic_ip_features}
            out = self.encoder(features=features, attention_mask=attention_mask, tasks=[])
            hidden_states = out["semantic_hidden_states"][1:]
            stacked_states = torch.stack(hidden_states, dim=-1)

        norm_weights = F.softmax(self.layer_weights, dim=-1)
        routed_seq = (stacked_states * norm_weights).sum(dim=-1)

        padding_mask = (attention_mask == 0)
        transformer_out = self.task_transformer(routed_seq, src_key_padding_mask=padding_mask)
        adapted_seq = routed_seq + (self.residual_scale * transformer_out)

        lang_embedding = self.pooler(adapted_seq, attention_mask)
        lang_logits = self.lang_head(lang_embedding, labels=lang_ids)

        reversed_emb = self.grl(lang_embedding)
        spk_logits = self.spk_head(reversed_emb)

        return lang_logits, spk_logits, lang_embedding



feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
acoustic_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_single_file(audio_path, max_seconds=10.0):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio_np = waveform.squeeze().numpy()
    max_samples = int(max_seconds * 16000)

    if len(audio_np) > max_samples:
        start = (len(audio_np) - max_samples) // 2
        audio_np = audio_np[start: start + max_samples]

    min_samples = int(1.0 * 16000)
    if len(audio_np) < min_samples:
        audio_np = np.pad(audio_np, (0, min_samples - len(audio_np)))

    sem_out = feature_extractor(audio_np, sampling_rate=16000, return_tensors="pt")
    input_features = sem_out.input_features

    acoustic_features = acoustic_transform(torch.tensor(audio_np).unsqueeze(0))
    acoustic_features = torch.clamp(acoustic_features, min=1e-5).log().transpose(1, 2)

    attention_mask = torch.ones(1, input_features.shape[1], dtype=torch.long)

    return input_features.to(device), acoustic_features.to(device), attention_mask.to(device)


