# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import numpy as np
import torch

from vllm.model_executor.models.arctic_speculator import MLPSpeculator

class MLPProposer:
    
    def link_model(
        self,
        model: MLPSpeculator,
    ):
        self.model = model
        self.device = next(model.parameters()).device

    def propose(
        self,
        context_token_ids: np.ndarray,
        previous_hidden_states: torch.Tensor,
    ) -> Optional[np.ndarray]:
        input_ids = torch.tensor(context_token_ids, device=self.device)
        
        # next_tokens = self.model.generate_proposals(
        #     input_ids=input_ids,
        #     previous_hidden_states=previous_hidden_states,
        #     num_predict_tokens=3,
        # )

        next_tokens, more_tokens = self.model.generate_proposals(
            input_ids=input_ids,
            previous_hidden_states=previous_hidden_states,
            num_predict_tokens=3,
        )

        more_tokens_cpu = []
        for mt in more_tokens:
            more_tokens_cpu.append(mt.cpu().numpy())

        from typing import List

        batch_size = input_ids.size(0)
        all_seq_candidates : List[List[List[int]]] = []

        for b in range(batch_size):
            seq_candidates : List[List[int]] = []
            seq_candidates.append([more_tokens_cpu[0][b, 0], more_tokens_cpu[1][b, 0], more_tokens_cpu[2][b, 0]])
            seq_candidates.append([more_tokens_cpu[0][b, 1]])
            seq_candidates.append([more_tokens_cpu[0][b, 2]])
            #seq_candidates.append([more_tokens_cpu[0][b, 3]])
            seq_candidates.append([more_tokens_cpu[0][b, 0], more_tokens_cpu[1][b, 1]])
            seq_candidates.append([more_tokens_cpu[0][b, 0], more_tokens_cpu[1][b, 2]])
            #seq_candidates.append([more_tokens_cpu[0][b, 0], more_tokens_cpu[1][b, 3]])
            seq_candidates.append([more_tokens_cpu[0][b, 0], more_tokens_cpu[1][b, 0], more_tokens_cpu[2][b, 1]])
            seq_candidates.append([more_tokens_cpu[0][b, 0], more_tokens_cpu[1][b, 0], more_tokens_cpu[2][b, 2]])
            #seq_candidates.append([more_tokens_cpu[0][b, 0], more_tokens_cpu[1][b, 0], more_tokens_cpu[2][b, 3]])

            all_seq_candidates.append(seq_candidates)

        #print('seq_candidates', seq_candidates)

        #return next_tokens.cpu().numpy()
        return next_tokens.cpu().numpy(), all_seq_candidates
        
