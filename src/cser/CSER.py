import torch
import torch.nn as nn

class CSERM(nn.Module):
    """
    Continuous Speech Emotion Recognition Model (CSERM).
    This model uses a two-layer BiLSTM to process pre-extracted WavLM features
    and predict continuous emotion dimensions (Valence, Arousal, Dominance).
    """
    def __init__(self, bilstm_params: dict):
        """
        Initializes the CSERM model.

        Args:
            bilstm_params (dict): A dictionary containing parameters for the BiLSTM layers,
                                  including 'hidden_size' and 'num_layers'.
        """
        super(CSERM, self).__init__()

        # First BiLSTM layer
        self.bilstm_1 = nn.LSTM(
            input_size=1024,  # WavLM Large output feature dimension
            hidden_size=bilstm_params['hidden_size'],
            num_layers=bilstm_params['num_layers'],
            bidirectional=True,
            dropout=0.5,
            batch_first=True
        )

        # Final fully connected layer to predict Valence, Arousal, and Dominance
        self.output_layer = nn.Linear(bilstm_params['hidden_size'] * 2, 3)

    def forward(self, wav_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            wav_features (torch.Tensor): A tensor of pre-extracted and averaged audio features.
                                         Shape: (batch_size, num_seconds, feature_dim)

        Returns:
            torch.Tensor: The predicted VAD values for each second.
                          Shape: (batch_size, num_seconds, 3)
        """
        # Pass features through the BiLSTM layer
        lstm_out, _ = self.bilstm_1(wav_features)

        # Pass the BiLSTM output to the final linear layer
        output = self.output_layer(lstm_out)

        return output